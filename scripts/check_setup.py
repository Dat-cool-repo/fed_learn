from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from datasets import load_dataset
from transformers import AutoTokenizer

from fed_learn.config import (
    load_experiment_config,
    load_local_paths,
    load_model_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate dataset, metadata path, and model access.")
    parser.add_argument("--skip-dataset", action="store_true", help="Skip the SuperNI access check.")
    parser.add_argument("--skip-model", action="store_true", help="Skip the tokenizer access check.")
    parser.add_argument(
        "--local-config",
        type=Path,
        default=REPO_ROOT / "configs" / "local.toml",
        help="Path to the local machine config.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    experiment = load_experiment_config(REPO_ROOT / "configs" / "experiments" / "pilot_superni.toml")
    model = load_model_config(REPO_ROOT / "configs" / "models" / "qwen_1_5b_instruct.toml")
    local_paths = load_local_paths(args.local_config)

    if local_paths.superni_metadata_repo is not None:
        if local_paths.superni_metadata_repo.exists():
            print(f"[ok] metadata repo: {local_paths.superni_metadata_repo}")
        else:
            print(f"[warn] metadata repo missing: {local_paths.superni_metadata_repo}")
    else:
        print("[warn] local metadata path not configured yet")

    if not args.skip_dataset:
        dataset = load_dataset(experiment.dataset_name, split="train", streaming=True)
        print(f"[ok] dataset stream: {experiment.dataset_name}")
        print(f"     features: {dataset.features}")

    if not args.skip_model:
        tokenizer = AutoTokenizer.from_pretrained(model.tokenizer_id)
        print(f"[ok] tokenizer: {model.tokenizer_id}")
        print(f"     vocab size: {tokenizer.vocab_size}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

