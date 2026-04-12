from __future__ import annotations

from typing import Any


def extract_trainable_state(model: Any, *, to_cpu: bool = True) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        tensor = parameter.detach().clone()
        if to_cpu:
            tensor = tensor.cpu()
        state[name] = tensor
    return state


def load_trainable_state(
    model: Any,
    state: dict[str, Any],
    *,
    strict: bool = True,
) -> None:
    torch_module = _require_torch()
    named_parameters = dict(model.named_parameters())

    missing_keys: list[str] = []
    unexpected_keys: list[str] = []

    for name, tensor in state.items():
        parameter = named_parameters.get(name)
        if parameter is None:
            unexpected_keys.append(name)
            continue
        if parameter.shape != tensor.shape:
            raise ValueError(
                f"Shape mismatch for {name}: expected {tuple(parameter.shape)}, found {tuple(tensor.shape)}."
            )
        with torch_module.no_grad():
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))

    if strict:
        for name, parameter in named_parameters.items():
            if parameter.requires_grad and name not in state:
                missing_keys.append(name)
        if missing_keys or unexpected_keys:
            raise KeyError(_render_key_mismatch(missing_keys, unexpected_keys))


def average_trainable_states(
    states: list[dict[str, Any]],
    *,
    weights: list[float] | None = None,
) -> dict[str, Any]:
    if not states:
        raise ValueError("Expected at least one trainable state to average.")

    _validate_state_keys(states)
    normalized_weights = _normalize_weights(len(states), weights)

    averaged_state: dict[str, Any] = {}
    for key in states[0]:
        accumulator = states[0][key].clone().mul_(normalized_weights[0])
        for index in range(1, len(states)):
            accumulator.add_(states[index][key], alpha=normalized_weights[index])
        averaged_state[key] = accumulator

    return averaged_state


def clone_trainable_state(state: dict[str, Any], *, to_cpu: bool = True) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for name, tensor in state.items():
        new_tensor = tensor.detach().clone()
        if to_cpu:
            new_tensor = new_tensor.cpu()
        cloned[name] = new_tensor
    return cloned


def zero_like_trainable_state(state: dict[str, Any], *, to_cpu: bool = True) -> dict[str, Any]:
    zeros: dict[str, Any] = {}
    for name, tensor in state.items():
        new_tensor = tensor.detach().clone().zero_()
        if to_cpu:
            new_tensor = new_tensor.cpu()
        zeros[name] = new_tensor
    return zeros


def add_trainable_states(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    _validate_state_keys([left, right])
    return {key: left[key].clone().add_(right[key]) for key in left}


def subtract_trainable_states(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    _validate_state_keys([left, right])
    return {key: left[key].clone().sub_(right[key]) for key in left}


def scale_trainable_state(state: dict[str, Any], factor: float) -> dict[str, Any]:
    return {key: tensor.clone().mul_(factor) for key, tensor in state.items()}


def trainable_state_l2_norm(state: dict[str, Any]) -> float:
    torch_module = _require_torch()
    if not state:
        return 0.0
    accumulator = torch_module.tensor(0.0)
    for tensor in state.values():
        accumulator = accumulator + tensor.detach().float().pow(2).sum()
    return float(torch_module.sqrt(accumulator).cpu())


def render_trainable_state_summary(state: dict[str, Any]) -> str:
    total_scalars = sum(int(tensor.numel()) for tensor in state.values())
    lines = [
        "Trainable State Summary",
        f"  Tensor count: {len(state)}",
        f"  Scalar count: {total_scalars:,}",
    ]

    preview_items = list(state.items())[:8]
    if preview_items:
        lines.append("  Tensor preview:")
        for name, tensor in preview_items:
            lines.append(f"    - {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype}")
    return "\n".join(lines)


def _normalize_weights(expected_count: int, weights: list[float] | None) -> list[float]:
    if weights is None:
        return [1.0 / expected_count] * expected_count
    if len(weights) != expected_count:
        raise ValueError(f"Expected {expected_count} weights, found {len(weights)}.")

    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("Expected positive aggregation weights.")
    return [weight / total_weight for weight in weights]


def _validate_state_keys(states: list[dict[str, Any]]) -> None:
    reference_keys = tuple(states[0].keys())
    reference_set = set(reference_keys)
    for index, state in enumerate(states[1:], start=1):
        current_set = set(state.keys())
        if current_set != reference_set:
            raise KeyError(
                f"Trainable state at index {index} does not match the reference key set."
            )


def _render_key_mismatch(missing_keys: list[str], unexpected_keys: list[str]) -> str:
    parts: list[str] = []
    if missing_keys:
        parts.append(f"missing keys: {', '.join(sorted(missing_keys))}")
    if unexpected_keys:
        parts.append(f"unexpected keys: {', '.join(sorted(unexpected_keys))}")
    return "; ".join(parts)


def _require_torch() -> Any:
    try:
        return __import__("torch")
    except ImportError as exc:
        raise ImportError("torch is required for PEFT state loading.") from exc
