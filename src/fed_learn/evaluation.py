from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

from .local_train import infer_model_device


@dataclass(frozen=True)
class EvaluationResult:
    example_count: int
    batch_count: int
    mean_loss: float


def evaluate_dataloader(
    model: Any,
    dataloader: Any,
    *,
    max_batches: int | None = None,
) -> EvaluationResult:
    torch_module = _require_torch()
    model.eval()
    model_device = infer_model_device(model)

    losses: list[float] = []
    example_count = 0
    batch_count = 0
    with torch_module.no_grad():
        for batch in dataloader:
            tensor_batch = _move_batch_to_device(batch, device=model_device)
            outputs = model(
                input_ids=tensor_batch["input_ids"],
                attention_mask=tensor_batch["attention_mask"],
                labels=tensor_batch["labels"],
            )
            losses.append(float(outputs.loss.detach().cpu()))
            batch_count += 1
            example_count += int(tensor_batch["input_ids"].shape[0])
            if max_batches is not None and batch_count >= max_batches:
                break

    mean_loss = statistics.fmean(losses) if losses else 0.0
    return EvaluationResult(
        example_count=example_count,
        batch_count=batch_count,
        mean_loss=mean_loss,
    )


def evaluate_client_loaders(
    model: Any,
    loaders: dict[str, Any],
    *,
    max_batches_per_client: int | None = None,
) -> EvaluationResult:
    if not loaders:
        return EvaluationResult(example_count=0, batch_count=0, mean_loss=0.0)

    weighted_loss_total = 0.0
    total_examples = 0
    total_batches = 0
    for dataloader in loaders.values():
        result = evaluate_dataloader(model, dataloader, max_batches=max_batches_per_client)
        weighted_loss_total += result.mean_loss * result.example_count
        total_examples += result.example_count
        total_batches += result.batch_count

    mean_loss = weighted_loss_total / total_examples if total_examples > 0 else 0.0
    return EvaluationResult(
        example_count=total_examples,
        batch_count=total_batches,
        mean_loss=mean_loss,
    )


def render_evaluation_result(result: EvaluationResult, *, label: str = "Evaluation Result") -> str:
    return "\n".join(
        [
            label,
            f"  Examples: {result.example_count}",
            f"  Batches: {result.batch_count}",
            f"  Mean loss: {result.mean_loss:.4f}",
        ]
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
        raise ImportError("torch is required for evaluation.") from exc
