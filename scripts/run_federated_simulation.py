from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fed_learn.config import load_local_paths, load_model_config
from fed_learn.federated import FederatedRunConfig, render_run_summary, run_federated_simulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one federated simulation configuration locally.")
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
    parser.add_argument("--aggregation-method", choices=["fedavg", "fedprox", "scaffold"], required=True)
    parser.add_argument("--peft-method", choices=["fft", "lora", "soft_prompt"], required=True)
    parser.add_argument("--heterogeneity-level", required=True)
    parser.add_argument("--participation-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--local-epochs", type=int, default=3)
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
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "federated",
        help="Base directory for per-run outputs.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Optional JSONL path for per-round logs. Defaults to <output-root>/<run-name>/round_logs.jsonl.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=None,
        help="Optional metrics JSON path. Defaults to <output-root>/<run-name>/metrics.json.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional path for rolling round-level checkpoint. Defaults to <output-root>/<run-name>/checkpoint.pt.",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="Optional directory for checkpoint snapshots. Defaults to <output-root>/<run-name>/checkpoints/.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    assignments_path = args.assignments_path
    run_config = FederatedRunConfig(
        aggregation_method=args.aggregation_method,
        peft_method=args.peft_method,
        heterogeneity_level=args.heterogeneity_level,
        participation_fraction=args.participation_fraction,
        seed=args.seed,
        rounds=args.rounds,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        local_epochs=args.local_epochs,
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
    )

    model_config = load_model_config(args.model_config)
    local_paths = load_local_paths(args.local_config, missing_ok=True)
    run_dir = args.output_root / run_config.run_name
    log_path = args.log_path or (run_dir / "round_logs.jsonl")
    metrics_path = args.metrics_path or (run_dir / "metrics.json")
    checkpoint_path = args.checkpoint_path or (run_dir / "checkpoint.pt")
    snapshot_dir = args.snapshot_dir or (run_dir / "checkpoints")
    result = run_federated_simulation(
        run_config,
        model_config=model_config,
        local_paths=local_paths,
        db_path=args.db_path,
        examples_path=args.examples_path,
        assignments_path=assignments_path,
        log_path=log_path,
        metrics_path=metrics_path,
        checkpoint_path=checkpoint_path,
        snapshot_dir=snapshot_dir,
    )
    print(render_run_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
