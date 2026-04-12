from __future__ import annotations

import csv
import hashlib
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VALID_SPLITS = {"train", "val", "test"}


@dataclass(frozen=True)
class StandardizedExample:
    example_id: str
    task_id: str
    task_type: str
    prompt: str
    target: str
    split: str


@dataclass(frozen=True)
class ClientAssignment:
    example_id: str
    client_id: str
    heterogeneity_level: str


@dataclass(frozen=True)
class ClientExample:
    example_id: str
    task_id: str
    task_type: str
    prompt: str
    target: str
    split: str
    client_id: str
    heterogeneity_level: str


@dataclass(frozen=True)
class TokenizationConfig:
    max_prompt_tokens: int = 384
    max_target_tokens: int = 128
    add_eos_token: bool = True


class CausalClientDataset:
    def __init__(self, encoded_examples: list[dict[str, Any]]) -> None:
        self._encoded_examples = encoded_examples

    def __len__(self) -> int:
        return len(self._encoded_examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._encoded_examples[index]


def load_standardized_examples(path: Path) -> list[StandardizedExample]:
    rows = _read_csv_rows(
        path,
        required_columns=("example_id", "task_id", "task_type", "prompt", "target", "split"),
    )

    examples: list[StandardizedExample] = []
    seen_ids: set[str] = set()
    for row in rows:
        example_id = _required_cell(row, "example_id", path)
        if example_id in seen_ids:
            raise ValueError(f"Duplicate example_id found in {path}: {example_id}")
        seen_ids.add(example_id)

        split = _normalize_split(_required_cell(row, "split", path), path)
        examples.append(
            StandardizedExample(
                example_id=example_id,
                task_id=_required_cell(row, "task_id", path),
                task_type=_required_cell(row, "task_type", path),
                prompt=_required_cell(row, "prompt", path),
                target=_required_cell(row, "target", path),
                split=split,
            )
        )
    return examples


def load_client_assignments(path: Path) -> list[ClientAssignment]:
    rows = _read_csv_rows(
        path,
        required_columns=("example_id", "client_id", "heterogeneity_level"),
    )

    assignments: list[ClientAssignment] = []
    seen_ids: set[str] = set()
    for row in rows:
        example_id = _required_cell(row, "example_id", path)
        if example_id in seen_ids:
            raise ValueError(f"Duplicate example_id found in {path}: {example_id}")
        seen_ids.add(example_id)
        assignments.append(
            ClientAssignment(
                example_id=example_id,
                client_id=_required_cell(row, "client_id", path),
                heterogeneity_level=_required_cell(row, "heterogeneity_level", path),
            )
        )
    return assignments


def _deterministic_split(
    example_id: str,
    *,
    train_fraction: float = 0.70,
    val_fraction: float = 0.10,
) -> str:
    digest = hashlib.sha256(example_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train_fraction:
        return "train"
    if bucket < train_fraction + val_fraction:
        return "val"
    return "test"


def load_from_sqlite(
    db_path: Path,
    *,
    train_fraction: float = 0.70,
    val_fraction: float = 0.10,
) -> tuple[list[StandardizedExample], list[ClientAssignment]]:
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database does not exist: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Load task_type lookup from task_manifest
    cur.execute("SELECT task_name, task_type FROM task_manifest")
    task_type_map: dict[str, str] = {}
    for row in cur.fetchall():
        raw_type = row["task_type"] or "unknown"
        task_type_map[row["task_name"]] = row["task_name"] if raw_type == "unknown" else raw_type

    # Load examples, sorted by (task_name, example_id) for deterministic index assignment
    cur.execute("SELECT example_id, task_name, input, target FROM examples ORDER BY task_name, example_id")
    db_examples = cur.fetchall()

    # Assign sequential indices within each task → canonical_id = "{task_name}_{index}"
    # Also assign deterministic splits since the DB has no split column
    examples: list[StandardizedExample] = []
    canonical_id_map: dict[str, str] = {}  # canonical_id → original example_id
    current_task: str | None = None
    task_index = 0

    for row in db_examples:
        task_name = row["task_name"]
        if task_name != current_task:
            current_task = task_name
            task_index = 0
        canonical_id = f"{task_name}_{task_index}"
        task_index += 1

        original_id = row["example_id"]
        canonical_id_map[canonical_id] = original_id
        split = _deterministic_split(
            original_id,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
        )
        examples.append(
            StandardizedExample(
                example_id=canonical_id,
                task_id=task_name,
                task_type=task_type_map.get(task_name, task_name),
                prompt=row["input"],
                target=row["target"],
                split=split,
            )
        )

    # Load client assignments, keeping only those whose canonical_id maps to a real example
    canonical_ids = set(canonical_id_map.keys())
    cur.execute("SELECT example_id, client_id, heterogeneity_level FROM client_assignments")
    assignments: list[ClientAssignment] = []
    skipped = 0
    for row in cur.fetchall():
        assign_id = row["example_id"]
        if assign_id not in canonical_ids:
            skipped += 1
            continue
        assignments.append(
            ClientAssignment(
                example_id=assign_id,
                client_id=str(row["client_id"]),
                heterogeneity_level=row["heterogeneity_level"],
            )
        )

    conn.close()

    if skipped > 0:
        import warnings
        warnings.warn(
            f"Dropped {skipped} orphaned client assignments "
            f"(assignment IDs with no matching example in the DB).",
            stacklevel=2,
        )

    return examples, assignments


def merge_examples_with_assignments(
    examples: list[StandardizedExample],
    assignments: list[ClientAssignment],
    *,
    split: str | None = None,
    heterogeneity_level: str | None = None,
) -> list[ClientExample]:
    example_by_id = {example.example_id: example for example in examples}
    merged: list[ClientExample] = []

    for assignment in assignments:
        example = example_by_id.get(assignment.example_id)
        if example is None:
            continue

        if split is not None and example.split != _normalize_split(split, Path("<runtime>")):
            continue
        if heterogeneity_level is not None and assignment.heterogeneity_level != heterogeneity_level:
            continue

        merged.append(
            ClientExample(
                example_id=example.example_id,
                task_id=example.task_id,
                task_type=example.task_type,
                prompt=example.prompt,
                target=example.target,
                split=example.split,
                client_id=assignment.client_id,
                heterogeneity_level=assignment.heterogeneity_level,
            )
        )

    merged.sort(key=lambda item: (item.client_id, item.example_id))
    return merged


def group_examples_by_client(examples: list[ClientExample]) -> dict[str, list[ClientExample]]:
    grouped: dict[str, list[ClientExample]] = defaultdict(list)
    for example in examples:
        grouped[example.client_id].append(example)
    return dict(sorted(grouped.items()))


def tokenize_client_example(
    example: ClientExample,
    tokenizer: Any,
    config: TokenizationConfig,
) -> dict[str, Any]:
    prompt_ids = tokenizer(
        example.prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=config.max_prompt_tokens,
    )["input_ids"]
    target_ids = tokenizer(
        example.target,
        add_special_tokens=False,
        truncation=True,
        max_length=config.max_target_tokens,
    )["input_ids"]

    if config.add_eos_token and getattr(tokenizer, "eos_token_id", None) is not None:
        target_ids = [*target_ids, int(tokenizer.eos_token_id)]

    if not target_ids:
        raise ValueError(f"Target text for example {example.example_id!r} tokenized to an empty sequence.")

    input_ids = [*prompt_ids, *target_ids]
    labels = ([-100] * len(prompt_ids)) + target_ids
    attention_mask = [1] * len(input_ids)

    return {
        "example_id": example.example_id,
        "client_id": example.client_id,
        "task_id": example.task_id,
        "task_type": example.task_type,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_client_dataloaders(
    client_examples: dict[str, list[ClientExample]],
    *,
    tokenizer: Any,
    tokenization_config: TokenizationConfig,
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
) -> dict[str, Any]:
    torch_module = _require_torch()
    loaders: dict[str, Any] = {}
    for client_index, (client_id, examples) in enumerate(client_examples.items()):
        encoded_examples = [
            tokenize_client_example(example, tokenizer, tokenization_config)
            for example in examples
        ]
        dataset = CausalClientDataset(encoded_examples)
        generator = None
        if shuffle and seed is not None:
            generator = torch_module.Generator()
            generator.manual_seed(seed + client_index)
        loaders[client_id] = torch_module.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
            collate_fn=lambda batch, pad_token_id=tokenizer.pad_token_id: collate_causal_batch(
                batch,
                pad_token_id=pad_token_id,
            ),
        )
    return loaders


def collate_causal_batch(features: list[dict[str, Any]], *, pad_token_id: int | None) -> dict[str, Any]:
    torch_module = _require_torch()
    if pad_token_id is None:
        raise ValueError("Tokenizer must define a pad_token_id before batching.")

    input_ids = [
        torch_module.tensor(feature["input_ids"], dtype=torch_module.long) for feature in features
    ]
    attention_masks = [
        torch_module.tensor(feature["attention_mask"], dtype=torch_module.long) for feature in features
    ]
    labels = [
        torch_module.tensor(feature["labels"], dtype=torch_module.long) for feature in features
    ]

    pad_sequence = torch_module.nn.utils.rnn.pad_sequence
    batch = {
        "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id),
        "attention_mask": pad_sequence(attention_masks, batch_first=True, padding_value=0),
        "labels": pad_sequence(labels, batch_first=True, padding_value=-100),
        "example_ids": [feature["example_id"] for feature in features],
        "client_ids": [feature["client_id"] for feature in features],
        "task_ids": [feature["task_id"] for feature in features],
    }
    return batch


def render_client_data_summary(examples: list[ClientExample]) -> str:
    if not examples:
        return "Client Data Summary\n  Examples: 0"

    task_types = sorted({example.task_type for example in examples})
    client_counts: dict[str, int] = defaultdict(int)
    for example in examples:
        client_counts[example.client_id] += 1

    lines = [
        "Client Data Summary",
        f"  Examples: {len(examples)}",
        f"  Clients: {len(client_counts)}",
        f"  Task types: {', '.join(task_types)}",
        "  Client preview:",
    ]
    for client_id, count in list(sorted(client_counts.items()))[:8]:
        lines.append(f"    - {client_id}: {count} examples")
    return "\n".join(lines)


def _read_csv_rows(path: Path, *, required_columns: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file is missing a header row: {path}")

        missing_columns = [column for column in required_columns if column not in reader.fieldnames]
        if missing_columns:
            joined = ", ".join(missing_columns)
            raise ValueError(f"CSV file {path} is missing required columns: {joined}")

        return [dict(row) for row in reader]


def _required_cell(row: dict[str, str], column: str, path: Path) -> str:
    value = (row.get(column) or "").strip()
    if not value:
        raise ValueError(f"CSV file {path} has an empty value in required column {column!r}.")
    return value


def _normalize_split(split: str, path: Path) -> str:
    normalized = split.strip().lower()
    if normalized == "validation":
        normalized = "val"
    if normalized not in VALID_SPLITS:
        raise ValueError(f"Unsupported split {split!r} in {path}. Expected one of {sorted(VALID_SPLITS)}.")
    return normalized


def _require_torch() -> Any:
    try:
        return __import__("torch")
    except ImportError as exc:
        raise ImportError("torch is required for data batching.") from exc

