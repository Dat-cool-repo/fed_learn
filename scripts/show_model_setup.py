from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fed_learn.config import load_local_paths, load_model_config
from fed_learn.modeling import (
    build_model_bundle,
    create_model_load_kwargs,
    create_tokenizer_load_kwargs,
    render_model_bundle_summary,
)
from fed_learn.peft_state import extract_trainable_state, render_trainable_state_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Show the model-side scaffold for federated PEFT before wiring in data."
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=REPO_ROOT / "configs" / "models" / "qwen_1_5b_instruct.toml",
        help="Path to the shared model config.",
    )
    parser.add_argument(
        "--local-config",
        type=Path,
        default=REPO_ROOT / "configs" / "local.toml",
        help="Path to the local machine config.",
    )
    parser.add_argument(
        "--peft-method",
        choices=["lora", "soft_prompt"],
        default="lora",
        help="Which PEFT branch to attach to the frozen base model.",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Actually instantiate the tokenizer and model. Without this flag, print a dry-run summary only.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model_config = load_model_config(args.model_config)
    local_paths = load_local_paths(args.local_config, missing_ok=True)
    cache_dir = local_paths.preferred_cache_dir()

    print("Model Setup Template")
    print(f"  Model config: {args.model_config}")
    print(f"  Base model: {model_config.model_id}")
    print(f"  Tokenizer: {model_config.tokenizer_id}")
    print(f"  PEFT method: {args.peft_method}")
    print(f"  Cache dir: {cache_dir or '<default>'}")
    print(f"  Tokenizer kwargs: {create_tokenizer_load_kwargs(model_config, cache_dir=cache_dir)}")
    print(f"  Model kwargs: {create_model_load_kwargs(model_config, cache_dir=cache_dir)}")

    if not args.load:
        print("  Load step: skipped (pass --load to instantiate the model stack)")
        return 0

    bundle = build_model_bundle(model_config, peft_method=args.peft_method, cache_dir=cache_dir)
    print()
    print(render_model_bundle_summary(bundle))

    trainable_state = extract_trainable_state(bundle.model)
    print()
    print(render_trainable_state_summary(trainable_state))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
