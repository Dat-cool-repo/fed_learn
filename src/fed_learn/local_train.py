from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LocalTrainConfig:
    local_epochs: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    max_steps: int | None = None


@dataclass(frozen=True)
class LocalTrainResult:
    client_id: str
    example_count: int
    local_epochs: int
    steps_completed: int
    mean_loss: float
    loss_history: tuple[float, ...]
    mean_task_loss: float = 0.0
    task_loss_history: tuple[float, ...] = ()


def train_local_client(
    model: Any,
    dataloader: Any,
    *,
    client_id: str,
    config: LocalTrainConfig,
    fedprox_mu: float = 0.0,
    proximal_reference_state: dict[str, Any] | None = None,
    scaffold_server_control: dict[str, Any] | None = None,
    scaffold_client_control: dict[str, Any] | None = None,
) -> LocalTrainResult:
    torch_module = _require_torch()
    named_trainable_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    trainable_parameters = list(named_trainable_parameters.values())
    if not trainable_parameters:
        raise ValueError("Model has no trainable parameters.")

    optimizer = torch_module.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    model.train()
    model_device = infer_model_device(model)

    if proximal_reference_state is not None:
        proximal_reference_state = {
            name: tensor.to(device=model_device)
            for name, tensor in proximal_reference_state.items()
        }
    if scaffold_server_control is not None:
        scaffold_server_control = {
            name: tensor.to(device=model_device)
            for name, tensor in scaffold_server_control.items()
        }
    if scaffold_client_control is not None:
        scaffold_client_control = {
            name: tensor.to(device=model_device)
            for name, tensor in scaffold_client_control.items()
        }

    loss_history: list[float] = []
    task_loss_history: list[float] = []
    steps_completed = 0
    example_count = 0

    for _epoch_index in range(config.local_epochs):
        for batch in dataloader:
            optimizer.zero_grad(set_to_none=True)
            tensor_batch = _move_batch_to_device(batch, device=model_device)
            outputs = model(
                input_ids=tensor_batch["input_ids"],
                attention_mask=tensor_batch["attention_mask"],
                labels=tensor_batch["labels"],
            )
            task_loss = outputs.loss
            task_loss_history.append(float(task_loss.detach().cpu()))
            loss = task_loss
            if fedprox_mu > 0.0:
                if proximal_reference_state is None:
                    raise ValueError("FedProx requires proximal_reference_state when fedprox_mu > 0.")
                loss = loss + _compute_fedprox_penalty(
                    named_trainable_parameters,
                    proximal_reference_state,
                    fedprox_mu,
                )
            loss.backward()
            if scaffold_server_control is not None or scaffold_client_control is not None:
                _apply_scaffold_gradient_correction(
                    named_trainable_parameters,
                    server_control=scaffold_server_control or {},
                    client_control=scaffold_client_control or {},
                )

            if config.max_grad_norm > 0:
                torch_module.nn.utils.clip_grad_norm_(trainable_parameters, config.max_grad_norm)

            optimizer.step()
            loss_history.append(float(loss.detach().cpu()))
            steps_completed += 1
            example_count += int(tensor_batch["input_ids"].shape[0])

            if config.max_steps is not None and steps_completed >= config.max_steps:
                break
        if config.max_steps is not None and steps_completed >= config.max_steps:
            break

    mean_loss = statistics.fmean(loss_history) if loss_history else 0.0
    mean_task_loss = statistics.fmean(task_loss_history) if task_loss_history else 0.0
    return LocalTrainResult(
        client_id=client_id,
        example_count=example_count,
        local_epochs=config.local_epochs,
        steps_completed=steps_completed,
        mean_loss=mean_loss,
        loss_history=tuple(loss_history),
        mean_task_loss=mean_task_loss,
        task_loss_history=tuple(task_loss_history),
    )


def render_local_train_result(result: LocalTrainResult) -> str:
    lines = [
        "Local Train Result",
        f"  Client: {result.client_id}",
        f"  Example passes: {result.example_count}",
        f"  Local epochs: {result.local_epochs}",
        f"  Steps completed: {result.steps_completed}",
        f"  Mean loss: {result.mean_loss:.4f}",
    ]
    if result.loss_history:
        preview = ", ".join(f"{loss:.4f}" for loss in result.loss_history[:8])
        lines.append(f"  Loss preview: {preview}")
    return "\n".join(lines)


def infer_model_device(model: Any) -> Any:
    for parameter in model.parameters():
        if parameter.requires_grad:
            return parameter.device
    for parameter in model.parameters():
        return parameter.device
    raise ValueError("Model does not expose any parameters.")


def _compute_fedprox_penalty(
    named_trainable_parameters: dict[str, Any],
    reference_state: dict[str, Any],
    fedprox_mu: float,
) -> Any:
    torch_module = _require_torch()
    penalty = None
    for name, parameter in named_trainable_parameters.items():
        reference_tensor = reference_state.get(name)
        if reference_tensor is None:
            raise KeyError(f"FedProx reference state is missing trainable tensor {name!r}.")
        difference = parameter - reference_tensor.to(dtype=parameter.dtype)
        term = difference.pow(2).sum()
        penalty = term if penalty is None else penalty + term
    if penalty is None:
        penalty = torch_module.tensor(0.0)
    return 0.5 * fedprox_mu * penalty


def _apply_scaffold_gradient_correction(
    named_trainable_parameters: dict[str, Any],
    *,
    server_control: dict[str, Any],
    client_control: dict[str, Any],
) -> None:
    for name, parameter in named_trainable_parameters.items():
        if parameter.grad is None:
            continue
        server_tensor = server_control.get(name)
        client_tensor = client_control.get(name)
        if server_tensor is None or client_tensor is None:
            raise KeyError(
                f"SCAFFOLD control state is missing trainable tensor {name!r}."
            )
        parameter.grad.add_(
            server_tensor.to(dtype=parameter.grad.dtype)
        )
        parameter.grad.sub_(
            client_tensor.to(dtype=parameter.grad.dtype)
        )


def _move_batch_to_device(batch: dict[str, Any], *, device: Any) -> dict[str, Any]:
    torch_module = _require_torch()
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch_module.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _require_torch() -> Any:
    try:
        return __import__("torch")
    except ImportError as exc:
        raise ImportError("torch is required for local training.") from exc
