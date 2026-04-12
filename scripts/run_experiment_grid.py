from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fed_learn.config import load_experiment_config, load_local_paths, load_model_config
from fed_learn.federated import build_experiment_grid, render_run_summary, run_experiment_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the federated experiment grid locally.",
        epilog=(
            "Chunking examples:\n"
            "  Person A: --chunk-index 0 --num-chunks 3\n"
            "  Person B: --chunk-index 1 --num-chunks 3\n"
            "  Person C: --chunk-index 2 --num-chunks 3\n\n"
            "Filter examples:\n"
            "  Only LoRA runs:       --only-peft lora\n"
            "  Only FedAvg + high:   --only-aggregation fedavg --only-heterogeneity high\n"
            "  Combine chunk + filter: --only-peft lora --chunk-index 0 --num-chunks 2\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=REPO_ROOT / "configs" / "experiments" / "pilot_superni.toml",
        help="Path to the shared experiment config.",
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
        "--assignments-template",
        default=None,
        help="Assignment path template with {heterogeneity} placeholder (only if not using --db-path).",
    )

    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-local-steps", type=int, default=None)
    parser.add_argument("--max-eval-batches-per-client", type=int, default=None)
    parser.add_argument("--fedprox-mu", type=float, default=0.01)
    parser.add_argument("--max-prompt-tokens", type=int, default=384)
    parser.add_argument("--max-target-tokens", type=int, default=128)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant",
                        help="Learning rate schedule across rounds.")
    parser.add_argument("--lr-min-factor", type=float, default=0.1,
                        help="Minimum LR as fraction of initial LR (for cosine schedule).")
    parser.add_argument(
        "--checkpoint-rounds",
        type=int,
        nargs="*",
        default=[5, 10],
        help="Round indices to save as checkpoint snapshots for later expensive evaluation.",
    )
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs" / "federated")
    parser.add_argument("--use-full-seeds", action="store_true")

    # ── Chunk-based splitting ──────────────────────────────────────────
    chunk_group = parser.add_argument_group(
        "chunking",
        "Split the full grid into N equal chunks so different machines can run in parallel.",
    )
    chunk_group.add_argument(
        "--chunk-index",
        type=int,
        default=None,
        help="Zero-based index of the chunk this machine should run (0 .. num-chunks-1).",
    )
    chunk_group.add_argument(
        "--num-chunks",
        type=int,
        default=None,
        help="Total number of chunks the grid is split into.",
    )

    # ── Dimension filters ──────────────────────────────────────────────
    filter_group = parser.add_argument_group(
        "filters",
        "Narrow the grid to specific dimension values. Filters are applied BEFORE chunking.",
    )
    filter_group.add_argument(
        "--only-peft",
        nargs="+",
        default=None,
        metavar="METHOD",
        help="Only run these PEFT methods (e.g. lora soft_prompt).",
    )
    filter_group.add_argument(
        "--only-aggregation",
        nargs="+",
        default=None,
        metavar="METHOD",
        help="Only run these aggregation methods (e.g. fedavg fedprox scaffold).",
    )
    filter_group.add_argument(
        "--only-heterogeneity",
        nargs="+",
        default=None,
        metavar="LEVEL",
        help="Only run these heterogeneity levels (e.g. low high).",
    )

    # ── Dry run ────────────────────────────────────────────────────────
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the run names that would execute, then exit without running.",
    )

    return parser


def _apply_filters(
    run_configs: list,
    *,
    only_peft: list[str] | None,
    only_aggregation: list[str] | None,
    only_heterogeneity: list[str] | None,
) -> list:
    filtered = run_configs
    if only_peft is not None:
        allowed = {m.strip().lower() for m in only_peft}
        filtered = [c for c in filtered if c.peft_method in allowed]
    if only_aggregation is not None:
        allowed = {m.strip().lower() for m in only_aggregation}
        filtered = [c for c in filtered if c.aggregation_method in allowed]
    if only_heterogeneity is not None:
        allowed = {m.strip().lower() for m in only_heterogeneity}
        filtered = [c for c in filtered if c.heterogeneity_level in allowed]
    return filtered


def _apply_chunking(
    run_configs: list,
    *,
    chunk_index: int,
    num_chunks: int,
) -> list:
    if chunk_index < 0 or chunk_index >= num_chunks:
        raise ValueError(
            f"chunk-index must be in [0, {num_chunks - 1}], got {chunk_index}."
        )
    # Deterministic: configs are already in a stable order from build_experiment_grid
    # (seeds → heterogeneity → aggregation → peft → participation)
    total = len(run_configs)
    base_size = total // num_chunks
    remainder = total % num_chunks
    # First `remainder` chunks get base_size+1, the rest get base_size
    if chunk_index < remainder:
        start = chunk_index * (base_size + 1)
        end = start + base_size + 1
    else:
        start = remainder * (base_size + 1) + (chunk_index - remainder) * base_size
        end = start + base_size
    return run_configs[start:end]


def main() -> int:
    args = build_parser().parse_args()

    # Validate chunk args come in pairs
    if (args.chunk_index is None) != (args.num_chunks is None):
        print("Error: --chunk-index and --num-chunks must both be provided or both omitted.")
        return 1

    experiment_config = load_experiment_config(args.experiment_config)
    model_config = load_model_config(args.model_config)
    local_paths = load_local_paths(args.local_config, missing_ok=True)
    run_configs = build_experiment_grid(
        experiment_config,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        max_local_steps=args.max_local_steps,
        max_eval_batches_per_client=args.max_eval_batches_per_client,
        fedprox_mu=args.fedprox_mu,
        max_prompt_tokens=args.max_prompt_tokens,
        max_target_tokens=args.max_target_tokens,
        lr_schedule=args.lr_schedule,
        lr_min_factor=args.lr_min_factor,
        checkpoint_rounds=tuple(sorted({round_index for round_index in args.checkpoint_rounds if round_index > 0})),
        use_full_seeds=args.use_full_seeds,
    )

    full_count = len(run_configs)

    # Apply dimension filters first (narrows the grid)
    run_configs = _apply_filters(
        run_configs,
        only_peft=args.only_peft,
        only_aggregation=args.only_aggregation,
        only_heterogeneity=args.only_heterogeneity,
    )
    filtered_count = len(run_configs)

    # Apply chunking second (splits the filtered grid)
    if args.chunk_index is not None:
        run_configs = _apply_chunking(
            run_configs,
            chunk_index=args.chunk_index,
            num_chunks=args.num_chunks,
        )

    chunk_label = ""
    if args.chunk_index is not None:
        chunk_label = f" (chunk {args.chunk_index}/{args.num_chunks})"
    filter_label = ""
    if filtered_count < full_count:
        filter_label = f" (filtered from {full_count})"

    print(f"Grid: {len(run_configs)} configs to run{chunk_label}{filter_label}")
    for i, rc in enumerate(run_configs):
        print(f"  [{i}] {rc.run_name}")

    if args.dry_run:
        print("\n--dry-run: exiting without running.")
        return 0

    if not run_configs:
        print("Nothing to run after filtering/chunking.")
        return 0

    print()
    results = run_experiment_grid(
        run_configs,
        model_config=model_config,
        local_paths=local_paths,
        db_path=args.db_path,
        examples_path=args.examples_path,
        assignments_template=args.assignments_template,
        output_root=args.output_root,
    )
    print()
    for result in results:
        print(render_run_summary(result))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
