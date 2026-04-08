from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fed_learn.config import load_experiment_config, load_local_paths, load_model_config
from fed_learn.plan import render_plan_summary


def main() -> None:
    experiment = load_experiment_config(REPO_ROOT / "configs" / "experiments" / "pilot_superni.toml")
    model = load_model_config(REPO_ROOT / "configs" / "models" / "qwen_1_5b_instruct.toml")
    local_paths = load_local_paths(REPO_ROOT / "configs" / "local.toml")
    print(render_plan_summary(experiment=experiment, model=model, paths=local_paths))


if __name__ == "__main__":
    main()

