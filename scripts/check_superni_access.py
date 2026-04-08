from pathlib import Path
import sys

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fed_learn.config import load_experiment_config


def main() -> None:
    experiment = load_experiment_config(REPO_ROOT / "configs" / "experiments" / "pilot_superni.toml")
    dataset = load_dataset(experiment.dataset_name, split="train", streaming=True)
    print(dataset)


if __name__ == "__main__":
    main()
