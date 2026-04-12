from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fed_learn.config import load_local_paths, load_model_config
from fed_learn.data_pipeline import (
    TokenizationConfig,
    build_client_dataloaders,
    group_examples_by_client,
    load_client_assignments,
    load_from_sqlite,
    load_standardized_examples,
    merge_examples_with_assignments,
    render_client_data_summary,
)
from fed_learn.local_train import LocalTrainConfig, render_local_train_result, train_local_client
from fed_learn.modeling import build_model_bundle, render_model_bundle_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a single local client training smoke test."
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
        "--peft-method",
        choices=["lora", "soft_prompt"],
        default="lora",
        help="Which PEFT branch to train.",
    )
    parser.add_argument(
        "--client-id",
        default=None,
        help="Specific client to train. Defaults to the first available client.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Which dataset split to use.",
    )
    parser.add_argument(
        "--heterogeneity-level",
        default=None,
        help="Optional heterogeneity level filter.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-prompt-tokens", type=int, default=384)
    parser.add_argument("--max-target-tokens", type=int, default=128)
    parser.add_argument("--no-shuffle", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.db_path is not None and args.db_path.exists():
        examples, assignments = load_from_sqlite(args.db_path)
    elif args.examples_path is not None and args.assignments_path is not None:
        examples = load_standardized_examples(args.examples_path)
        assignments = load_client_assignments(args.assignments_path)
    else:
        raise SystemExit(
            "Provide either --db-path (default) or both --examples-path and --assignments-path."
        )

    merged_examples = merge_examples_with_assignments(
        examples,
        assignments,
        split=args.split,
        heterogeneity_level=args.heterogeneity_level,
    )
    print(render_client_data_summary(merged_examples))

    client_examples = group_examples_by_client(merged_examples)
    if not client_examples:
        raise SystemExit("No client examples matched the selected files and filters.")

    client_id = args.client_id or next(iter(client_examples))
    if client_id not in client_examples:
        raise SystemExit(f"Client {client_id!r} was not found in the joined dataset.")

    local_paths = load_local_paths(args.local_config, missing_ok=True)
    cache_dir = local_paths.preferred_cache_dir()
    model_config = load_model_config(args.model_config)
    model_bundle = build_model_bundle(
        model_config,
        peft_method=args.peft_method,
        cache_dir=cache_dir,
    )
    print()
    print(render_model_bundle_summary(model_bundle))

    loaders = build_client_dataloaders(
        {client_id: client_examples[client_id]},
        tokenizer=model_bundle.tokenizer,
        tokenization_config=TokenizationConfig(
            max_prompt_tokens=args.max_prompt_tokens,
            max_target_tokens=args.max_target_tokens,
        ),
        batch_size=args.batch_size,
        shuffle=not args.no_shuffle,
    )
    dataloader = loaders[client_id]

    result = train_local_client(
        model_bundle.model,
        dataloader,
        client_id=client_id,
        config=LocalTrainConfig(
            local_epochs=args.local_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            max_steps=args.max_steps,
        ),
    )
    print()
    print(render_local_train_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
