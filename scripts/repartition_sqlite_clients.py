from __future__ import annotations

import argparse
import random
import sqlite3
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild the SQLite client assignments table with a new clients-per-group setting."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=REPO_ROOT / "data" / "data" / "processed" / "federated_data.db",
        help="Path to the SQLite database to repartition.",
    )
    parser.add_argument(
        "--clients-per-group",
        type=int,
        default=3,
        help="Number of clients per task group for the high-heterogeneity split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when shuffling per-task and per-group assignments.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repartition_database(
        args.db_path,
        clients_per_group=args.clients_per_group,
        seed=args.seed,
    )
    return 0


def repartition_database(
    db_path: Path,
    *,
    clients_per_group: int,
    seed: int,
) -> None:
    if clients_per_group <= 0:
        raise ValueError("clients_per_group must be positive.")
    if not db_path.exists():
        raise FileNotFoundError(f"Database does not exist: {db_path}")

    rng = random.Random(seed)
    connection = sqlite3.connect(str(db_path))
    try:
        task_groups = load_task_groups(connection)
        canonical_ids_by_task = load_canonical_ids_by_task(connection)

        high_clients = build_high_heterogeneity_assignments(
            canonical_ids_by_task,
            task_groups,
            clients_per_group=clients_per_group,
            rng=rng,
        )
        low_clients = build_low_heterogeneity_assignments(
            canonical_ids_by_task,
            num_clients=clients_per_group * len(task_groups),
            rng=rng,
        )

        write_client_assignments(connection, high_clients=high_clients, low_clients=low_clients)
        render_summary(connection)
    finally:
        connection.close()


def load_task_groups(connection: sqlite3.Connection) -> dict[str, list[str]]:
    rows = connection.execute(
        "SELECT task_name, task_type FROM task_manifest ORDER BY task_name"
    ).fetchall()
    if not rows:
        raise ValueError("task_manifest is empty.")

    grouped: dict[str, list[str]] = defaultdict(list)
    for task_name, task_type in rows:
        normalized_type = (task_type or "").strip()
        if not normalized_type or normalized_type.lower() == "unknown":
            raise ValueError(
                f"task_manifest contains missing or unknown task_type for task {task_name!r}."
            )
        grouped[normalized_type].append(task_name)
    return dict(grouped)


def load_canonical_ids_by_task(connection: sqlite3.Connection) -> dict[str, list[str]]:
    rows = connection.execute(
        "SELECT task_name, example_id FROM examples ORDER BY task_name, example_id"
    ).fetchall()
    if not rows:
        raise ValueError("examples table is empty.")

    canonical_ids_by_task: dict[str, list[str]] = defaultdict(list)
    for task_name, _original_example_id in rows:
        canonical_index = len(canonical_ids_by_task[task_name])
        canonical_ids_by_task[task_name].append(f"{task_name}_{canonical_index}")
    return dict(canonical_ids_by_task)


def build_high_heterogeneity_assignments(
    canonical_ids_by_task: dict[str, list[str]],
    task_groups: dict[str, list[str]],
    *,
    clients_per_group: int,
    rng: random.Random,
) -> list[list[str]]:
    client_data: list[list[str]] = []
    for task_type in sorted(task_groups):
        group_samples: list[str] = []
        for task_name in task_groups[task_type]:
            group_samples.extend(canonical_ids_by_task.get(task_name, ()))
        rng.shuffle(group_samples)
        chunks = [group_samples[index::clients_per_group] for index in range(clients_per_group)]
        client_data.extend(chunks)
    return client_data


def build_low_heterogeneity_assignments(
    canonical_ids_by_task: dict[str, list[str]],
    *,
    num_clients: int,
    rng: random.Random,
) -> list[list[str]]:
    client_data = [[] for _ in range(num_clients)]
    for task_name in sorted(canonical_ids_by_task):
        task_samples = list(canonical_ids_by_task[task_name])
        rng.shuffle(task_samples)
        for index, canonical_id in enumerate(task_samples):
            client_data[index % num_clients].append(canonical_id)
    return client_data


def write_client_assignments(
    connection: sqlite3.Connection,
    *,
    high_clients: list[list[str]],
    low_clients: list[list[str]],
) -> None:
    rows: list[tuple[str, int, str]] = []
    for heterogeneity_level, clients in (("high", high_clients), ("low", low_clients)):
        for client_id, samples in enumerate(clients):
            for canonical_id in samples:
                rows.append((canonical_id, client_id, heterogeneity_level))

    with connection:
        connection.execute("DELETE FROM client_assignments")
        connection.executemany(
            "INSERT INTO client_assignments (example_id, client_id, heterogeneity_level) VALUES (?, ?, ?)",
            rows,
        )


def render_summary(connection: sqlite3.Connection) -> None:
    client_counts = connection.execute(
        """
        SELECT heterogeneity_level, COUNT(DISTINCT client_id)
        FROM client_assignments
        GROUP BY heterogeneity_level
        ORDER BY heterogeneity_level
        """
    ).fetchall()
    size_rows = connection.execute(
        """
        SELECT heterogeneity_level, client_id, COUNT(*) AS example_count
        FROM client_assignments
        GROUP BY heterogeneity_level, client_id
        ORDER BY heterogeneity_level, client_id
        """
    ).fetchall()

    by_level: dict[str, list[int]] = defaultdict(list)
    for heterogeneity_level, _client_id, example_count in size_rows:
        by_level[heterogeneity_level].append(int(example_count))

    print("Repartition Summary")
    print(f"  Client counts: {client_counts}")
    for heterogeneity_level, counts in sorted(by_level.items()):
        print(
            "  "
            f"{heterogeneity_level}: clients={len(counts)}, min={min(counts)}, "
            f"max={max(counts)}, mean={sum(counts) / len(counts):.2f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
