"""Project package for the federated PEFT empirical pipeline."""

from .config import ExperimentConfig, LocalPaths, ModelConfig
from .modeling import (
    ModelBundle,
    attach_peft_adapter,
    build_model_bundle,
    count_parameters,
    create_model_load_kwargs,
    create_tokenizer_load_kwargs,
    load_base_causal_lm,
    load_tokenizer,
    render_model_bundle_summary,
)
from .peft_state import (
    average_trainable_states,
    extract_trainable_state,
    load_trainable_state,
    render_trainable_state_summary,
)

__all__ = [
    "ExperimentConfig",
    "LocalPaths",
    "ModelBundle",
    "ModelConfig",
    "attach_peft_adapter",
    "average_trainable_states",
    "build_model_bundle",
    "count_parameters",
    "create_model_load_kwargs",
    "create_tokenizer_load_kwargs",
    "extract_trainable_state",
    "load_base_causal_lm",
    "load_tokenizer",
    "load_trainable_state",
    "render_model_bundle_summary",
    "render_trainable_state_summary",
]
