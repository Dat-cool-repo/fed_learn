from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import LoraTuningConfig, ModelConfig, SoftPromptTuningConfig


@dataclass
class ModelBundle:
    tokenizer: Any
    model: Any
    peft_method: str
    trainable_parameter_names: tuple[str, ...]
    trainable_parameter_count: int
    total_parameter_count: int

    @property
    def trainable_fraction(self) -> float:
        if self.total_parameter_count == 0:
            return 0.0
        return self.trainable_parameter_count / self.total_parameter_count


def create_tokenizer_load_kwargs(model_config: ModelConfig, *, cache_dir: Path | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": model_config.trust_remote_code,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return kwargs


def create_model_load_kwargs(model_config: ModelConfig, *, cache_dir: Path | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": model_config.trust_remote_code,
    }
    if model_config.torch_dtype:
        kwargs["torch_dtype"] = model_config.torch_dtype
    if model_config.device_map:
        kwargs["device_map"] = model_config.device_map
    if model_config.attn_implementation:
        kwargs["attn_implementation"] = model_config.attn_implementation
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return kwargs


def load_tokenizer(model_config: ModelConfig, *, cache_dir: Path | None = None) -> Any:
    transformers = _require_dependency("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_config.tokenizer_id,
        **create_tokenizer_load_kwargs(model_config, cache_dir=cache_dir),
    )
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "right"
    return tokenizer


def load_base_causal_lm(model_config: ModelConfig, *, cache_dir: Path | None = None) -> Any:
    transformers = _require_dependency("transformers")
    torch_module = _require_dependency("torch")
    model_load_kwargs = create_model_load_kwargs(model_config, cache_dir=cache_dir)
    if "torch_dtype" in model_load_kwargs:
        model_load_kwargs["torch_dtype"] = _resolve_torch_dtype(
            torch_module,
            str(model_load_kwargs["torch_dtype"]),
        )
    return transformers.AutoModelForCausalLM.from_pretrained(
        model_config.model_id,
        **model_load_kwargs,
    )


def attach_peft_adapter(base_model: Any, model_config: ModelConfig, peft_method: str) -> Any:
    normalized_method = peft_method.strip().lower()
    if normalized_method == "fft":
        base_model.requires_grad_(True)
        return base_model

    peft_module = _require_dependency("peft")
    base_model.requires_grad_(False)
    if normalized_method == "lora":
        peft_config = _build_lora_config(peft_module, model_config.lora)
    elif normalized_method == "soft_prompt":
        peft_config = _build_soft_prompt_config(peft_module, model_config.soft_prompt, model_config)
    else:
        raise ValueError(f"Unsupported PEFT method: {peft_method}")

    return peft_module.get_peft_model(base_model, peft_config)


def build_model_bundle(
    model_config: ModelConfig,
    *,
    peft_method: str,
    cache_dir: Path | None = None,
) -> ModelBundle:
    tokenizer = load_tokenizer(model_config, cache_dir=cache_dir)
    model = load_base_causal_lm(model_config, cache_dir=cache_dir)
    model = attach_peft_adapter(model, model_config, peft_method)
    trainable_names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    trainable_parameter_count, total_parameter_count = count_parameters(model)
    return ModelBundle(
        tokenizer=tokenizer,
        model=model,
        peft_method=peft_method,
        trainable_parameter_names=trainable_names,
        trainable_parameter_count=trainable_parameter_count,
        total_parameter_count=total_parameter_count,
    )


def count_parameters(model: Any) -> tuple[int, int]:
    trainable_parameter_count = 0
    total_parameter_count = 0
    for parameter in model.parameters():
        parameter_count = int(parameter.numel())
        total_parameter_count += parameter_count
        if parameter.requires_grad:
            trainable_parameter_count += parameter_count
    return trainable_parameter_count, total_parameter_count


def render_model_bundle_summary(bundle: ModelBundle) -> str:
    lines = [
        "Model Bundle Summary",
        f"  PEFT method: {bundle.peft_method}",
        f"  Trainable params: {bundle.trainable_parameter_count:,}",
        f"  Total params: {bundle.total_parameter_count:,}",
        f"  Trainable fraction: {bundle.trainable_fraction:.4%}",
    ]

    if bundle.trainable_parameter_names:
        lines.append("  Trainable parameter preview:")
        for name in bundle.trainable_parameter_names[:8]:
            lines.append(f"    - {name}")

    return "\n".join(lines)


def _build_lora_config(peft_module: Any, config: LoraTuningConfig) -> Any:
    return peft_module.LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(config.target_modules),
        bias="none",
        task_type=peft_module.TaskType.CAUSAL_LM,
    )


def _build_soft_prompt_config(
    peft_module: Any,
    config: SoftPromptTuningConfig,
    model_config: ModelConfig,
) -> Any:
    return peft_module.PromptTuningConfig(
        task_type=peft_module.TaskType.CAUSAL_LM,
        num_virtual_tokens=config.num_virtual_tokens,
        prompt_tuning_init=peft_module.PromptTuningInit.RANDOM,
        tokenizer_name_or_path=model_config.tokenizer_id,
    )


def _resolve_torch_dtype(torch_module: Any, dtype_name: str) -> Any:
    normalized_name = dtype_name.strip().lower()
    if not hasattr(torch_module, normalized_name):
        raise ValueError(f"torch does not define dtype {dtype_name!r}")
    return getattr(torch_module, normalized_name)


def _require_dependency(module_name: str) -> Any:
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise ImportError(
            f"{module_name!r} is required for model setup. Install the model stack first."
        ) from exc
