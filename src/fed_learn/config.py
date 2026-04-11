from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    dataset_name: str
    benchmark_name: str
    selected_task_count: int
    samples_per_task: int
    train_split: float
    validation_split: float
    test_split: float
    num_clients: int
    rounds: int
    local_epochs: int
    token_budget_per_client: int
    participation_fractions: tuple[float, ...]
    aggregation_methods: tuple[str, ...]
    heterogeneity_levels: tuple[str, ...]
    peft_methods: tuple[str, ...]
    pilot_seeds: tuple[int, ...]
    full_seeds: tuple[int, ...]


@dataclass(frozen=True)
class LocalPaths:
    superni_metadata_repo: Path | None = None
    output_root: Path | None = None
    hf_home: Path | None = None
    model_cache_dir: Path | None = None

    def preferred_cache_dir(self) -> Path | None:
        return self.model_cache_dir or self.hf_home


@dataclass(frozen=True)
class LoraTuningConfig:
    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class SoftPromptTuningConfig:
    num_virtual_tokens: int


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    tokenizer_id: str
    torch_dtype: str
    device_map: str | None
    attn_implementation: str | None
    trust_remote_code: bool
    lora: LoraTuningConfig
    soft_prompt: SoftPromptTuningConfig

    def get_peft_config(self, peft_method: str) -> LoraTuningConfig | SoftPromptTuningConfig:
        normalized = peft_method.strip().lower()
        if normalized == "lora":
            return self.lora
        if normalized == "soft_prompt":
            return self.soft_prompt
        raise ValueError(f"Unsupported PEFT method: {peft_method}")


def load_experiment_config(path: Path) -> ExperimentConfig:
    payload = _read_toml(path)
    config = ExperimentConfig(
        name=_require_string(payload, "name", path),
        dataset_name=_require_string(payload, "dataset_name", path),
        benchmark_name=_require_string(payload, "benchmark_name", path),
        selected_task_count=_require_int(payload, "selected_task_count", path),
        samples_per_task=_require_int(payload, "samples_per_task", path),
        train_split=_require_float(payload, "train_split", path),
        validation_split=_require_float(payload, "validation_split", path),
        test_split=_require_float(payload, "test_split", path),
        num_clients=_require_int(payload, "num_clients", path),
        rounds=_require_int(payload, "rounds", path),
        local_epochs=_require_int(payload, "local_epochs", path),
        token_budget_per_client=_require_int(payload, "token_budget_per_client", path),
        participation_fractions=_tuple_of_floats(payload.get("participation_fractions")),
        aggregation_methods=_tuple_of_strings(payload.get("aggregation_methods")),
        heterogeneity_levels=_tuple_of_strings(payload.get("heterogeneity_levels")),
        peft_methods=_tuple_of_strings(payload.get("peft_methods")),
        pilot_seeds=_tuple_of_ints(payload.get("pilot_seeds")),
        full_seeds=_tuple_of_ints(payload.get("full_seeds")),
    )
    _validate_experiment_config(config)
    return config


def load_local_paths(path: Path, *, missing_ok: bool = False) -> LocalPaths:
    if not path.exists():
        if missing_ok:
            return LocalPaths()
        raise FileNotFoundError(f"Local config does not exist: {path}")

    payload = _read_toml(path)
    paths_payload = payload.get("paths")
    if isinstance(paths_payload, dict):
        source_payload = paths_payload
    else:
        source_payload = payload

    return LocalPaths(
        superni_metadata_repo=_optional_path(source_payload.get("superni_metadata_repo")),
        output_root=_optional_path(source_payload.get("output_root")),
        hf_home=_optional_path(source_payload.get("hf_home")),
        model_cache_dir=_optional_path(source_payload.get("model_cache_dir")),
    )


def load_model_config(path: Path) -> ModelConfig:
    payload = _read_toml(path)
    peft_payload = payload.get("peft")
    if not isinstance(peft_payload, dict):
        raise KeyError(f"Expected a [peft] table in {path}")

    lora_payload = peft_payload.get("lora")
    soft_prompt_payload = peft_payload.get("soft_prompt")
    if not isinstance(lora_payload, dict):
        raise KeyError(f"Expected a [peft.lora] table in {path}")
    if not isinstance(soft_prompt_payload, dict):
        raise KeyError(f"Expected a [peft.soft_prompt] table in {path}")

    return ModelConfig(
        model_id=_require_string(payload, "model_id", path),
        tokenizer_id=_require_string(payload, "tokenizer_id", path),
        torch_dtype=_require_string(payload, "torch_dtype", path),
        device_map=_optional_string(payload.get("device_map")),
        attn_implementation=_optional_string(payload.get("attn_implementation")),
        trust_remote_code=bool(payload.get("trust_remote_code", False)),
        lora=LoraTuningConfig(
            rank=_require_int(lora_payload, "rank", path),
            alpha=_require_int(lora_payload, "alpha", path),
            dropout=_require_float(lora_payload, "dropout", path),
            target_modules=_tuple_of_strings(lora_payload.get("target_modules")),
        ),
        soft_prompt=SoftPromptTuningConfig(
            num_virtual_tokens=_require_int(soft_prompt_payload, "num_virtual_tokens", path)
        ),
    )


def _read_toml(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Config does not exist: {path}")

    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a TOML table at the root of {path}")
    return payload


def _validate_experiment_config(config: ExperimentConfig) -> None:
    split_total = config.train_split + config.validation_split + config.test_split
    if abs(split_total - 1.0) > 1e-6:
        raise ValueError(
            "Expected train_split + validation_split + test_split to equal 1.0, "
            f"found {split_total:.6f}."
        )

    if config.selected_task_count <= 0:
        raise ValueError("selected_task_count must be positive.")
    if config.samples_per_task <= 0:
        raise ValueError("samples_per_task must be positive.")
    if config.num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    if config.rounds <= 0:
        raise ValueError("rounds must be positive.")
    if config.local_epochs <= 0:
        raise ValueError("local_epochs must be positive.")
    if config.token_budget_per_client <= 0:
        raise ValueError("token_budget_per_client must be positive.")
    if not config.aggregation_methods:
        raise ValueError("Expected at least one aggregation method.")
    if not config.heterogeneity_levels:
        raise ValueError("Expected at least one heterogeneity level.")
    if not config.peft_methods:
        raise ValueError("Expected at least one PEFT method.")
    if not config.participation_fractions:
        raise ValueError("Expected at least one participation fraction.")


def _require_string(payload: dict[str, object], key: str, path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise KeyError(f"Expected {key!r} to be a non-empty string in {path}")
    return value.strip()


def _optional_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _require_int(payload: dict[str, object], key: str, path: Path) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise KeyError(f"Expected {key!r} to be an integer in {path}")
    return value


def _require_float(payload: dict[str, object], key: str, path: Path) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)):
        raise KeyError(f"Expected {key!r} to be numeric in {path}")
    return float(value)


def _tuple_of_strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    normalized = [str(item).strip() for item in value if str(item).strip()]
    return tuple(normalized)


def _tuple_of_ints(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    items: list[int] = []
    for raw_value in value:
        if not isinstance(raw_value, int):
            raise ValueError(f"Expected integer seeds, found {raw_value!r}")
        items.append(raw_value)
    return tuple(items)


def _tuple_of_floats(value: object) -> tuple[float, ...]:
    if not isinstance(value, list):
        return ()
    items: list[float] = []
    for raw_value in value:
        if not isinstance(raw_value, (int, float)):
            raise ValueError(f"Expected numeric participation fractions, found {raw_value!r}")
        items.append(float(raw_value))
    return tuple(items)


def _optional_path(value: object) -> Path | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return Path(stripped)
