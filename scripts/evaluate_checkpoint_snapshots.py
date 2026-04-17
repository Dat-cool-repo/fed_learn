from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fed_learn.config import load_local_paths, load_model_config
from fed_learn.data_pipeline import (
    ClientExample,
    TokenizationConfig,
    build_client_dataloaders,
    group_examples_by_client,
    load_client_assignments,
    load_from_sqlite,
    load_standardized_examples,
    merge_examples_with_assignments,
)
from fed_learn.evaluation import evaluate_client_loaders
from fed_learn.local_train import infer_model_device
from fed_learn.modeling import build_model_bundle, seed_runtime
from fed_learn.peft_state import load_trainable_state


@dataclass(frozen=True)
class CheckpointSource:
    round_index: int
    checkpoint_kind: str
    path: Path


@dataclass(frozen=True)
class CheckpointEvalResult:
    run_name: str
    aggregation_method: str
    peft_method: str
    heterogeneity_level: str
    participation_fraction: float
    seed: int
    split: str
    round_index: int
    checkpoint_kind: str
    checkpoint_path: str
    example_count: int | None
    batch_count: int | None
    mean_loss: float | None
    rouge_l: float | None
    generated_example_count: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved federated checkpoint snapshots for one or more run directories, "
            "then optionally write ROUGE-L scores back into metrics.json."
        )
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=REPO_ROOT / "configs" / "models" / "qwen_0_5b_instruct.toml",
        help="Path to the shared model config.",
    )
    parser.add_argument(
        "--local-config",
        type=Path,
        default=REPO_ROOT / "configs" / "local.toml",
        help="Path to the local machine config.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=REPO_ROOT / "data" / "data" / "processed" / "federated_data.db",
        help="Path to the SQLite database (preferred over CSV paths).",
    )
    parser.add_argument(
        "--examples-path",
        type=Path,
        default=None,
        help="Path to the standardized examples CSV (only if not using --db-path).",
    )
    parser.add_argument(
        "--assignments-path",
        type=Path,
        default=None,
        help="Path to the client assignment CSV (only if not using --db-path).",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "federated",
        help="Root directory that contains per-run output folders.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        nargs="+",
        default=None,
        help="One or more explicit run directories to evaluate.",
    )
    parser.add_argument(
        "--run-name",
        nargs="+",
        default=None,
        help="One or more run directory names under --runs-root to evaluate.",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Evaluate every run directory under --runs-root that contains metrics.json.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
        help="Dataset split to evaluate on. Default: val.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--max-prompt-tokens", type=int, default=384)
    parser.add_argument("--max-target-tokens", type=int, default=128)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on the number of evaluation examples per run (useful for smoke tests).",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Evaluate only the newest available checkpoint for each run.",
    )
    parser.add_argument(
        "--include-base-model",
        action="store_true",
        help="Evaluate the untuned base model in addition to any checkpoints.",
    )
    parser.add_argument(
        "--base-model-only",
        action="store_true",
        help="Evaluate only the untuned base model and skip checkpoint loading.",
    )
    parser.add_argument(
        "--rouge-only",
        action="store_true",
        help="Skip the loss pass and compute only generation-based ROUGE-L.",
    )
    parser.add_argument(
        "--skip-rouge",
        action="store_true",
        help="Skip generation-based ROUGE-L and only compute loss.",
    )
    parser.add_argument(
        "--no-write-metrics",
        action="store_true",
        help="Do not write ROUGE-L results back into metrics.json.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional JSON filename for the per-run checkpoint eval summary. Defaults to checkpoint_eval_<split>.json.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.rouge_only and args.skip_rouge:
        raise SystemExit("Choose at most one of --rouge-only and --skip-rouge.")
    if args.base_model_only:
        args.include_base_model = True

    run_dirs = _resolve_run_dirs(args)
    examples, assignments = _load_examples_and_assignments(args)
    local_paths = load_local_paths(args.local_config, missing_ok=True)
    cache_dir = local_paths.preferred_cache_dir()
    model_config = load_model_config(args.model_config)
    shared_base_model_bundle = None

    for run_dir in run_dirs:
        metadata = _load_run_metadata(run_dir)
        eval_examples = merge_examples_with_assignments(
            examples,
            assignments,
            split=args.split,
            heterogeneity_level=metadata["heterogeneity_level"],
        )
        if args.max_examples is not None:
            eval_examples = eval_examples[: args.max_examples]
        if not eval_examples:
            raise SystemExit(
                f"No {args.split!r} examples matched heterogeneity={metadata['heterogeneity_level']!r} "
                f"for run directory {run_dir}."
            )

        grouped_examples = group_examples_by_client(eval_examples)
        if args.include_base_model and shared_base_model_bundle is None:
            shared_base_model_bundle = build_model_bundle(
                model_config,
                peft_method="fft",
                cache_dir=cache_dir,
            )

        model_bundle = None
        eval_tokenizer = shared_base_model_bundle.tokenizer if shared_base_model_bundle is not None else None
        if not args.base_model_only:
            seed_runtime(int(metadata["seed"]))
            model_bundle = build_model_bundle(
                model_config,
                peft_method=metadata["peft_method"],
                cache_dir=cache_dir,
            )
            eval_tokenizer = model_bundle.tokenizer

        if eval_tokenizer is None:
            raise SystemExit("Could not determine a tokenizer for evaluation.")

        eval_loaders = build_client_dataloaders(
            grouped_examples,
            tokenizer=eval_tokenizer,
            tokenization_config=TokenizationConfig(
                max_prompt_tokens=args.max_prompt_tokens,
                max_target_tokens=args.max_target_tokens,
            ),
            batch_size=args.eval_batch_size,
            shuffle=False,
        )

        checkpoint_sources: list[CheckpointSource] = []
        if not args.base_model_only:
            checkpoint_sources = _discover_checkpoint_sources(
                run_dir,
                latest_only=args.latest_only,
            )
        if not checkpoint_sources and not args.include_base_model:
            raise SystemExit(f"No checkpoints were found under {run_dir}.")

        print()
        print(f"Run: {metadata['run_name']}")
        print(
            f"Evaluating {len(checkpoint_sources) + (1 if args.include_base_model else 0)} source(s) on split={args.split!r} "
            f"with {len(eval_examples)} example(s)."
        )

        run_results: list[CheckpointEvalResult] = []
        if args.include_base_model:
            base_result = _evaluate_loaded_model(
                model=shared_base_model_bundle.model,
                tokenizer=shared_base_model_bundle.tokenizer,
                eval_loaders=eval_loaders,
                eval_examples=eval_examples,
                metadata=metadata,
                split=args.split,
                checkpoint_kind="base",
                checkpoint_path="<base_model>",
                round_index=0,
                rouge_only=args.rouge_only,
                skip_rouge=args.skip_rouge,
                max_prompt_tokens=args.max_prompt_tokens,
                max_target_tokens=args.max_target_tokens,
            )
            run_results.append(base_result)
            _print_eval_result(base_result)

        for source in checkpoint_sources:
            checkpoint = _load_checkpoint(source.path)
            load_trainable_state(model_bundle.model, checkpoint["global_state"])
            result = _evaluate_loaded_model(
                model=model_bundle.model,
                tokenizer=model_bundle.tokenizer,
                eval_loaders=eval_loaders,
                eval_examples=eval_examples,
                metadata=metadata,
                split=args.split,
                checkpoint_kind=source.checkpoint_kind,
                checkpoint_path=str(source.path.relative_to(run_dir)),
                round_index=source.round_index,
                rouge_only=args.rouge_only,
                skip_rouge=args.skip_rouge,
                max_prompt_tokens=args.max_prompt_tokens,
                max_target_tokens=args.max_target_tokens,
            )
            run_results.append(result)
            _print_eval_result(result)

        output_path = _write_checkpoint_eval_file(
            run_dir,
            run_results,
            split=args.split,
            output_name=args.output_name,
        )
        print(f"  Wrote checkpoint eval summary -> {output_path}")

        if not args.skip_rouge and not args.no_write_metrics:
            _write_rouge_scores_to_metrics(
                run_dir,
                run_results,
                split=args.split,
                checkpoint_eval_path=output_path,
            )
            print(f"  Updated ROUGE-L entries -> {run_dir / 'metrics.json'}")

    return 0


def _resolve_run_dirs(args: argparse.Namespace) -> list[Path]:
    if args.run_dir is not None:
        run_dirs = [path.resolve() for path in args.run_dir]
    elif args.run_name is not None:
        run_dirs = [(args.runs_root / name).resolve() for name in args.run_name]
    elif args.all_runs:
        run_dirs = sorted(
            path.resolve()
            for path in args.runs_root.iterdir()
            if path.is_dir() and (path / "metrics.json").exists()
        )
    else:
        raise SystemExit(
            "Select at least one run with --run-dir, --run-name, or --all-runs."
        )

    missing = [str(path) for path in run_dirs if not path.exists()]
    if missing:
        raise SystemExit("These run directories do not exist:\n  " + "\n  ".join(missing))
    return run_dirs


def _load_examples_and_assignments(
    args: argparse.Namespace,
) -> tuple[list[Any], list[Any]]:
    if args.db_path is not None and args.db_path.exists():
        return load_from_sqlite(args.db_path)
    if args.examples_path is not None and args.assignments_path is not None:
        return (
            load_standardized_examples(args.examples_path),
            load_client_assignments(args.assignments_path),
        )
    raise SystemExit(
        "Provide either an existing --db-path or both --examples-path and --assignments-path."
    )


def _load_run_metadata(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Run directory is missing metrics.json: {run_dir}")
    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    required_keys = (
        "run_name",
        "aggregation_method",
        "peft_method",
        "heterogeneity_level",
        "participation_fraction",
        "seed",
    )
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise KeyError(
            f"metrics.json in {run_dir} is missing required keys: {', '.join(missing)}"
        )
    return payload


def _discover_checkpoint_sources(
    run_dir: Path,
    *,
    latest_only: bool = False,
) -> list[CheckpointSource]:
    sources_by_round: dict[int, CheckpointSource] = {}
    snapshot_dir = run_dir / "checkpoints"

    for path in sorted(snapshot_dir.glob("round_*.pt")):
        match = re.fullmatch(r"round_(\d+)\.pt", path.name)
        if match is None:
            continue
        round_index = int(match.group(1))
        sources_by_round[round_index] = CheckpointSource(
            round_index=round_index,
            checkpoint_kind="snapshot",
            path=path,
        )

    rolling_path = run_dir / "checkpoint.pt"
    if rolling_path.exists():
        rolling_checkpoint = _load_checkpoint(rolling_path)
        round_index = int(rolling_checkpoint["round_index"])
        sources_by_round.setdefault(
            round_index,
            CheckpointSource(
                round_index=round_index,
                checkpoint_kind="rolling",
                path=rolling_path,
            ),
        )

    ordered_sources = [sources_by_round[round_index] for round_index in sorted(sources_by_round)]
    if latest_only and ordered_sources:
        return [ordered_sources[-1]]
    return ordered_sources


def _load_checkpoint(path: Path) -> dict[str, Any]:
    torch_module = __import__("torch")
    checkpoint = torch_module.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected checkpoint payload dict in {path}")
    if "global_state" not in checkpoint or "round_index" not in checkpoint:
        raise KeyError(f"Checkpoint at {path} is missing required keys.")
    return checkpoint


def _evaluate_loaded_model(
    *,
    model: Any,
    tokenizer: Any,
    eval_loaders: dict[str, Any],
    eval_examples: list[ClientExample],
    metadata: dict[str, Any],
    split: str,
    checkpoint_kind: str,
    checkpoint_path: str,
    round_index: int,
    rouge_only: bool,
    skip_rouge: bool,
    max_prompt_tokens: int,
    max_target_tokens: int,
) -> CheckpointEvalResult:
    example_count = None
    batch_count = None
    mean_loss = None
    if not rouge_only:
        loss_result = evaluate_client_loaders(
            model,
            eval_loaders,
            max_batches_per_client=None,
        )
        example_count = loss_result.example_count
        batch_count = loss_result.batch_count
        mean_loss = loss_result.mean_loss

    rouge_l = None
    generated_example_count = 0
    if not skip_rouge:
        rouge_l = evaluate_rouge_l(
            model,
            tokenizer,
            eval_examples,
            max_prompt_tokens=max_prompt_tokens,
            max_new_tokens=max_target_tokens,
        )
        generated_example_count = len(eval_examples)

    return CheckpointEvalResult(
        run_name=metadata["run_name"],
        aggregation_method=metadata["aggregation_method"],
        peft_method=metadata["peft_method"],
        heterogeneity_level=metadata["heterogeneity_level"],
        participation_fraction=float(metadata["participation_fraction"]),
        seed=int(metadata["seed"]),
        split=split,
        round_index=round_index,
        checkpoint_kind=checkpoint_kind,
        checkpoint_path=checkpoint_path,
        example_count=example_count,
        batch_count=batch_count,
        mean_loss=mean_loss,
        rouge_l=rouge_l,
        generated_example_count=generated_example_count,
    )


def _print_eval_result(result: CheckpointEvalResult) -> None:
    loss_label = "skipped" if result.mean_loss is None else f"{result.mean_loss:.4f}"
    rouge_label = "skipped" if result.rouge_l is None else f"{result.rouge_l:.4f}"
    round_label = "base" if result.checkpoint_kind == "base" else f"{result.round_index:>2}"
    print(
        f"  Round {round_label} ({result.checkpoint_kind:<8})  "
        f"loss={loss_label}  rouge_l={rouge_label}"
    )


def evaluate_rouge_l(
    model: Any,
    tokenizer: Any,
    examples: list[ClientExample],
    *,
    max_prompt_tokens: int,
    max_new_tokens: int,
) -> float:
    torch_module = __import__("torch")
    model.eval()
    model_device = infer_model_device(model)
    scorer = _build_rouge_scorer()

    scores: list[float] = []
    with torch_module.inference_mode():
        for example in examples:
            encoded = tokenizer(
                example.prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=max_prompt_tokens,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(model_device)
                for key, value in encoded.items()
            }
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=getattr(tokenizer, "eos_token_id", None),
            )
            prompt_length = int(encoded["input_ids"].shape[1])
            completion_ids = generated[0][prompt_length:]
            prediction = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            scores.append(_score_rouge_l(example.target, prediction, scorer=scorer))

    return statistics.fmean(scores) if scores else 0.0


def _build_rouge_scorer() -> Any:
    try:
        from rouge_score import rouge_scorer  # type: ignore

        return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    except ImportError:
        return None


def _score_rouge_l(reference: str, prediction: str, *, scorer: Any) -> float:
    if scorer is not None:
        return float(scorer.score(reference, prediction)["rougeL"].fmeasure)
    return _fallback_rouge_l_fmeasure(reference, prediction)


def _fallback_rouge_l_fmeasure(reference: str, prediction: str) -> float:
    reference_tokens = reference.split()
    prediction_tokens = prediction.split()
    if not reference_tokens or not prediction_tokens:
        return 0.0

    lcs_length = _lcs_length(reference_tokens, prediction_tokens)
    precision = lcs_length / len(prediction_tokens)
    recall = lcs_length / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for right_index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[right_index - 1] + 1)
            else:
                current.append(max(previous[right_index], current[-1]))
        previous = current
    return previous[-1]


def _write_checkpoint_eval_file(
    run_dir: Path,
    results: list[CheckpointEvalResult],
    *,
    split: str,
    output_name: str | None,
) -> Path:
    if output_name is None:
        output_name = f"checkpoint_eval_{split}.json"
    output_path = run_dir / output_name
    payload = {
        "run_name": results[0].run_name if results else run_dir.name,
        "split": split,
        "results": [asdict(result) for result in results],
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path


def _write_rouge_scores_to_metrics(
    run_dir: Path,
    results: list[CheckpointEvalResult],
    *,
    split: str,
    checkpoint_eval_path: Path,
) -> None:
    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    rounds = payload.get("rounds", [])
    rouge_values = payload.get("rouge_l_per_round")
    if not isinstance(rounds, list):
        raise ValueError(f"metrics.json in {run_dir} does not define a usable rounds array.")
    if not isinstance(rouge_values, list) or len(rouge_values) != len(rounds):
        rouge_values = [None] * len(rounds)

    rouge_by_round = {
        result.round_index: result.rouge_l
        for result in results
        if result.rouge_l is not None
    }
    for index, round_index in enumerate(rounds):
        if round_index in rouge_by_round:
            rouge_values[index] = rouge_by_round[round_index]

    payload["rouge_l_per_round"] = rouge_values
    payload["rouge_l_split"] = split
    payload["checkpoint_eval_file"] = checkpoint_eval_path.name

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
