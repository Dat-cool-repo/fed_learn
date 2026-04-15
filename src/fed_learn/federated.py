from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ExperimentConfig, LocalPaths, ModelConfig
from .data_pipeline import (
    TokenizationConfig,
    build_client_dataloaders,
    group_examples_by_client,
    load_client_assignments,
    load_from_sqlite,
    load_standardized_examples,
    merge_examples_with_assignments,
)
from .evaluation import EvaluationResult, evaluate_client_loaders
from .local_train import LocalTrainConfig, LocalTrainResult, train_local_client
from .modeling import ModelBundle, build_model_bundle, seed_runtime
from .peft_state import (
    add_trainable_states,
    average_trainable_states,
    clone_trainable_state,
    extract_trainable_state,
    load_trainable_state,
    scale_trainable_state,
    subtract_trainable_states,
    trainable_state_l2_norm,
    zero_like_trainable_state,
)


@dataclass(frozen=True)
class FederatedRunConfig:
    aggregation_method: str
    peft_method: str
    heterogeneity_level: str
    participation_fraction: float
    seed: int
    rounds: int
    batch_size: int = 1
    eval_batch_size: int = 1
    local_epochs: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    max_local_steps: int | None = None
    max_eval_batches_per_client: int | None = None
    fedprox_mu: float = 0.01
    max_prompt_tokens: int = 384
    max_target_tokens: int = 128
    lr_schedule: str = "constant"  # "constant" or "cosine"
    lr_min_factor: float = 0.1  # minimum LR as fraction of initial LR for cosine
    checkpoint_rounds: tuple[int, ...] = ()

    @property
    def run_name(self) -> str:
        participation = int(self.participation_fraction * 100)
        return (
            f"{self.aggregation_method}-"
            f"{self.peft_method}-"
            f"{self.heterogeneity_level}-"
            f"p{participation}-"
            f"s{self.seed}"
        )


@dataclass(frozen=True)
class ClientRoundResult:
    client_id: str
    example_count: int
    train_result: LocalTrainResult
    update_norm: float


@dataclass(frozen=True)
class RoundLog:
    round_index: int
    selected_clients: tuple[str, ...]
    train_loss_mean: float
    train_task_loss_mean: float
    validation_loss: float
    update_norms: tuple[float, ...]
    update_norm_mean: float
    update_norm_max: float
    cosine_disagreements: tuple[float, ...]
    cosine_disagreement_mean: float
    cosine_disagreement_max: float
    total_client_examples: int
    total_local_steps: int


@dataclass(frozen=True)
class FederatedRunResult:
    run_config: FederatedRunConfig
    round_logs: tuple[RoundLog, ...]
    final_state: dict[str, Any]


def build_experiment_grid(
    experiment_config: ExperimentConfig,
    *,
    rounds: int | None = None,
    batch_size: int = 1,
    eval_batch_size: int = 1,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    max_local_steps: int | None = None,
    max_eval_batches_per_client: int | None = None,
    fedprox_mu: float = 0.01,
    max_prompt_tokens: int = 384,
    max_target_tokens: int = 128,
    lr_schedule: str = "constant",
    lr_min_factor: float = 0.1,
    checkpoint_rounds: tuple[int, ...] = (),
    use_full_seeds: bool = False,
) -> list[FederatedRunConfig]:
    seeds = experiment_config.full_seeds if use_full_seeds else experiment_config.pilot_seeds
    effective_rounds = rounds or experiment_config.rounds
    run_configs: list[FederatedRunConfig] = []
    for seed in seeds:
        for heterogeneity_level in experiment_config.heterogeneity_levels:
            for aggregation_method in experiment_config.aggregation_methods:
                for peft_method in experiment_config.peft_methods:
                    for participation_fraction in experiment_config.participation_fractions:
                        run_configs.append(
                            FederatedRunConfig(
                                aggregation_method=aggregation_method,
                                peft_method=peft_method,
                                heterogeneity_level=heterogeneity_level,
                                participation_fraction=participation_fraction,
                                seed=seed,
                                rounds=effective_rounds,
                                batch_size=batch_size,
                                eval_batch_size=eval_batch_size,
                                local_epochs=experiment_config.local_epochs,
                                learning_rate=learning_rate,
                                weight_decay=weight_decay,
                                max_grad_norm=max_grad_norm,
                                max_local_steps=max_local_steps,
                                max_eval_batches_per_client=max_eval_batches_per_client,
                                fedprox_mu=fedprox_mu,
                                max_prompt_tokens=max_prompt_tokens,
                                max_target_tokens=max_target_tokens,
                                lr_schedule=lr_schedule,
                                lr_min_factor=lr_min_factor,
                                checkpoint_rounds=checkpoint_rounds,
                            )
                        )
    return run_configs


def run_federated_simulation(
    run_config: FederatedRunConfig,
    *,
    model_config: ModelConfig,
    local_paths: LocalPaths,
    examples_path: Path | None = None,
    assignments_path: Path | None = None,
    db_path: Path | None = None,
    log_path: Path | None = None,
    metrics_path: Path | None = None,
    model_bundle: ModelBundle | None = None,
    checkpoint_path: Path | None = None,
    snapshot_dir: Path | None = None,
) -> FederatedRunResult:
    seed_runtime(run_config.seed)

    if model_bundle is None or model_bundle.peft_method != run_config.peft_method:
        cache_dir = local_paths.preferred_cache_dir()
        model_bundle = build_model_bundle(
            model_config,
            peft_method=run_config.peft_method,
            cache_dir=cache_dir,
        )

    train_loaders, val_loaders, train_example_counts = _prepare_dataloaders(
        examples_path=examples_path,
        assignments_path=assignments_path,
        db_path=db_path,
        heterogeneity_level=run_config.heterogeneity_level,
        tokenizer=model_bundle.tokenizer,
        tokenization_config=TokenizationConfig(
            max_prompt_tokens=run_config.max_prompt_tokens,
            max_target_tokens=run_config.max_target_tokens,
        ),
        batch_size=run_config.batch_size,
        eval_batch_size=run_config.eval_batch_size,
        seed=run_config.seed,
    )
    if not train_loaders:
        raise ValueError("No train client loaders were built from the provided data source.")

    client_ids = tuple(sorted(train_loaders))
    global_state = extract_trainable_state(model_bundle.model)
    server_control = zero_like_trainable_state(global_state)
    client_controls = {client_id: zero_like_trainable_state(global_state) for client_id in client_ids}

    rng = random.Random(run_config.seed)
    round_logs: list[RoundLog] = []
    start_round = 1
    checkpoint_round_set = set(run_config.checkpoint_rounds)

    resumed = _load_checkpoint(checkpoint_path) if checkpoint_path is not None else None
    if resumed is not None:
        global_state = resumed["global_state"]
        server_control = resumed["server_control"]
        client_controls = resumed["client_controls"]
        rng.setstate(resumed["rng_state"])
        round_logs = list(resumed["round_logs"])
        start_round = resumed["round_index"] + 1
        load_trainable_state(model_bundle.model, global_state)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if resumed is None and log_path.exists():
            log_path.unlink()
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        if resumed is None and metrics_path.exists():
            metrics_path.unlink()
    if snapshot_dir is not None:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    for round_index in range(start_round, run_config.rounds + 1):
        selected_clients = sample_clients_for_round(
            client_ids,
            participation_fraction=run_config.participation_fraction,
            rng=rng,
        )
        client_results: list[ClientRoundResult] = []
        local_states: list[dict[str, Any]] = []
        local_weights: list[float] = []
        update_states: list[dict[str, Any]] = []
        scaffold_deltas: list[dict[str, Any]] = []

        round_lr = _compute_round_learning_rate(
            base_lr=run_config.learning_rate,
            round_index=round_index,
            total_rounds=run_config.rounds,
            schedule=run_config.lr_schedule,
            min_factor=run_config.lr_min_factor,
        )

        for client_id in selected_clients:
            load_trainable_state(model_bundle.model, global_state)
            train_result = train_local_client(
                model_bundle.model,
                train_loaders[client_id],
                client_id=client_id,
                config=LocalTrainConfig(
                    local_epochs=run_config.local_epochs,
                    learning_rate=round_lr,
                    weight_decay=run_config.weight_decay,
                    max_grad_norm=run_config.max_grad_norm,
                    max_steps=run_config.max_local_steps,
                ),
                fedprox_mu=run_config.fedprox_mu if run_config.aggregation_method == "fedprox" else 0.0,
                proximal_reference_state=global_state if run_config.aggregation_method == "fedprox" else None,
                scaffold_server_control=server_control if run_config.aggregation_method == "scaffold" else None,
                scaffold_client_control=client_controls[client_id] if run_config.aggregation_method == "scaffold" else None,
            )

            local_state = extract_trainable_state(model_bundle.model)
            local_states.append(local_state)

            update_state = subtract_trainable_states(local_state, global_state)
            update_states.append(update_state)
            update_norm = trainable_state_l2_norm(update_state)
            local_weights.append(float(train_example_counts[client_id]))
            client_results.append(
                ClientRoundResult(
                    client_id=client_id,
                    example_count=train_example_counts[client_id],
                    train_result=train_result,
                    update_norm=update_norm,
                )
            )

            if run_config.aggregation_method == "scaffold":
                if train_result.steps_completed == 0:
                    continue
                control_delta = _compute_scaffold_control_delta(
                    global_state=global_state,
                    local_state=local_state,
                    server_control=server_control,
                    client_control=client_controls[client_id],
                    learning_rate=round_lr,
                    local_steps=train_result.steps_completed,
                )
                scaffold_deltas.append(control_delta)
                client_controls[client_id] = add_trainable_states(
                    client_controls[client_id],
                    control_delta,
                )

        global_state = average_trainable_states(local_states, weights=local_weights)
        load_trainable_state(model_bundle.model, global_state)

        if run_config.aggregation_method == "scaffold" and scaffold_deltas:
            mean_delta = average_trainable_states(scaffold_deltas)
            server_control = add_trainable_states(
                server_control,
                scale_trainable_state(mean_delta, len(selected_clients) / len(client_ids)),
            )

        cosine_disagreements = _compute_cosine_disagreements(update_states)
        evaluation = evaluate_client_loaders(
            model_bundle.model,
            val_loaders,
            max_batches_per_client=run_config.max_eval_batches_per_client,
        )
        round_log = _summarize_round(
            round_index=round_index,
            selected_clients=selected_clients,
            client_results=client_results,
            evaluation=evaluation,
            cosine_disagreements=cosine_disagreements,
        )
        round_logs.append(round_log)
        if log_path is not None:
            append_round_log(log_path, run_config=run_config, round_log=round_log)
        if metrics_path is not None:
            write_metrics_file(metrics_path, run_config=run_config, round_logs=round_logs)
        if checkpoint_path is not None:
            _save_checkpoint(
                checkpoint_path,
                round_index=round_index,
                global_state=global_state,
                server_control=server_control,
                client_controls=client_controls,
                rng_state=rng.getstate(),
                round_logs=round_logs,
            )
        if snapshot_dir is not None and round_index in checkpoint_round_set:
            snapshot_path = snapshot_dir / f"round_{round_index:04d}.pt"
            _save_checkpoint(
                snapshot_path,
                round_index=round_index,
                global_state=global_state,
                server_control=server_control,
                client_controls=client_controls,
                rng_state=rng.getstate(),
                round_logs=round_logs,
            )

    return FederatedRunResult(
        run_config=run_config,
        round_logs=tuple(round_logs),
        final_state=global_state,
    )


def run_experiment_grid(
    run_configs: list[FederatedRunConfig],
    *,
    model_config: ModelConfig,
    local_paths: LocalPaths,
    examples_path: Path | None = None,
    assignments_template: str | None = None,
    db_path: Path | None = None,
    output_root: Path | None = None,
) -> list[FederatedRunResult]:
    results: list[FederatedRunResult] = []
    sorted_configs = sorted(run_configs, key=lambda c: c.peft_method)
    for run_config in sorted_configs:
        assignments_path = None
        if assignments_template is not None:
            assignments_path = Path(assignments_template.format(heterogeneity=run_config.heterogeneity_level))
        log_path = None
        metrics_path = None
        ckpt_path = None
        snapshot_dir = None
        if output_root is not None:
            run_dir = output_root / run_config.run_name
            log_path = run_dir / "round_logs.jsonl"
            metrics_path = run_dir / "metrics.json"
            ckpt_path = run_dir / "checkpoint.pt"
            snapshot_dir = run_dir / "checkpoints"
        result = run_federated_simulation(
            run_config,
            model_config=model_config,
            local_paths=local_paths,
            examples_path=examples_path,
            assignments_path=assignments_path,
            db_path=db_path,
            log_path=log_path,
            metrics_path=metrics_path,
            checkpoint_path=ckpt_path,
            snapshot_dir=snapshot_dir,
        )
        results.append(result)
    return results


def sample_clients_for_round(
    client_ids: tuple[str, ...],
    *,
    participation_fraction: float,
    rng: random.Random,
) -> tuple[str, ...]:
    if not client_ids:
        raise ValueError("Expected at least one client id.")
    if participation_fraction <= 0 or participation_fraction > 1:
        raise ValueError("participation_fraction must be in the interval (0, 1].")

    sample_count = max(1, math.ceil(len(client_ids) * participation_fraction))
    sample_count = min(sample_count, len(client_ids))
    sampled = rng.sample(list(client_ids), k=sample_count)
    return tuple(sorted(sampled))


def append_round_log(path: Path, *, run_config: FederatedRunConfig, round_log: RoundLog) -> None:
    record = {
        "run_name": run_config.run_name,
        "aggregation_method": run_config.aggregation_method,
        "peft_method": run_config.peft_method,
        "heterogeneity_level": run_config.heterogeneity_level,
        "participation_fraction": run_config.participation_fraction,
        "seed": run_config.seed,
        "round_index": round_log.round_index,
        "selected_clients": list(round_log.selected_clients),
        "train_loss_mean": round_log.train_loss_mean,
        "train_task_loss_mean": round_log.train_task_loss_mean,
        "validation_loss": round_log.validation_loss,
        "update_norms": list(round_log.update_norms),
        "update_norm_mean": round_log.update_norm_mean,
        "update_norm_max": round_log.update_norm_max,
        "cosine_disagreements": list(round_log.cosine_disagreements),
        "cosine_disagreement_mean": round_log.cosine_disagreement_mean,
        "cosine_disagreement_max": round_log.cosine_disagreement_max,
        "total_client_examples": round_log.total_client_examples,
        "total_local_steps": round_log.total_local_steps,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def write_metrics_file(
    path: Path,
    *,
    run_config: FederatedRunConfig,
    round_logs: list[RoundLog],
) -> None:
    payload = {
        "run_name": run_config.run_name,
        "aggregation_method": run_config.aggregation_method,
        "peft_method": run_config.peft_method,
        "heterogeneity_level": run_config.heterogeneity_level,
        "participation_fraction": run_config.participation_fraction,
        "seed": run_config.seed,
        "checkpoint_rounds": list(run_config.checkpoint_rounds),
        "rounds": [round_log.round_index for round_log in round_logs],
        "rouge_l_per_round": [None for _ in round_logs],
        "train_loss_per_round": [round_log.train_loss_mean for round_log in round_logs],
        "train_task_loss_per_round": [round_log.train_task_loss_mean for round_log in round_logs],
        "validation_loss_per_round": [round_log.validation_loss for round_log in round_logs],
        "update_norms_per_round": [list(round_log.update_norms) for round_log in round_logs],
        "cosine_disagreement_per_round": [
            list(round_log.cosine_disagreements) for round_log in round_logs
        ],
        "selected_clients_per_round": [list(round_log.selected_clients) for round_log in round_logs],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def render_run_summary(result: FederatedRunResult) -> str:
    lines = [
        "Federated Run Summary",
        f"  Run: {result.run_config.run_name}",
        f"  Rounds: {len(result.round_logs)}",
    ]
    if result.round_logs:
        final_round = result.round_logs[-1]
        lines.extend(
            [
                f"  Final train loss (total): {final_round.train_loss_mean:.4f}",
                f"  Final train loss (task only): {final_round.train_task_loss_mean:.4f}",
                f"  Final val loss: {final_round.validation_loss:.4f}",
                f"  Final update mean norm: {final_round.update_norm_mean:.4f}",
                f"  Final cosine disagreement mean: {final_round.cosine_disagreement_mean:.4f}",
            ]
        )
    return "\n".join(lines)


def _save_checkpoint(
    path: Path,
    *,
    round_index: int,
    global_state: dict[str, Any],
    server_control: dict[str, Any],
    client_controls: dict[str, dict[str, Any]],
    rng_state: tuple[Any, ...],
    round_logs: list[RoundLog],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch_module = __import__("torch")
    checkpoint = {
        "round_index": round_index,
        "global_state": global_state,
        "server_control": server_control,
        "client_controls": client_controls,
        "rng_state": rng_state,
        "round_logs": round_logs,
    }
    torch_module.save(checkpoint, str(path))


def _load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    torch_module = __import__("torch")
    return torch_module.load(str(path), map_location="cpu", weights_only=False)


def _compute_round_learning_rate(
    base_lr: float,
    round_index: int,
    total_rounds: int,
    schedule: str,
    min_factor: float,
) -> float:
    if schedule == "constant" or total_rounds <= 1:
        return base_lr
    if schedule == "cosine":
        progress = (round_index - 1) / (total_rounds - 1)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return base_lr * (min_factor + (1.0 - min_factor) * cosine_decay)
    raise ValueError(f"Unknown lr_schedule: {schedule!r}. Expected 'constant' or 'cosine'.")


def _compute_cosine_disagreements(update_states: list[dict[str, Any]]) -> list[float]:
    torch_module = __import__("torch")
    if not update_states:
        return []

    flattened_updates = [_flatten_trainable_state(state) for state in update_states]
    if len(flattened_updates) == 1:
        return [0.0]

    mean_update = flattened_updates[0].clone()
    for flattened in flattened_updates[1:]:
        mean_update.add_(flattened)
    mean_update.div_(len(flattened_updates))

    mean_norm = float(mean_update.norm().cpu())
    disagreements: list[float] = []
    for flattened in flattened_updates:
        update_norm = float(flattened.norm().cpu())
        if update_norm <= 1e-12 or mean_norm <= 1e-12:
            disagreements.append(0.0)
            continue
        cosine = torch_module.dot(flattened, mean_update) / (flattened.norm() * mean_update.norm())
        clipped = float(torch_module.clamp(cosine, min=-1.0, max=1.0).cpu())
        disagreements.append(1.0 - clipped)
    return disagreements


def _flatten_trainable_state(state: dict[str, Any]) -> Any:
    torch_module = __import__("torch")
    if not state:
        return torch_module.zeros(0, dtype=torch_module.float32)
    return torch_module.cat([
        tensor.detach().float().reshape(-1).cpu()
        for tensor in state.values()
    ])


def _prepare_dataloaders(
    *,
    examples_path: Path | None = None,
    assignments_path: Path | None = None,
    db_path: Path | None = None,
    heterogeneity_level: str,
    tokenizer: Any,
    tokenization_config: TokenizationConfig,
    batch_size: int,
    eval_batch_size: int,
    seed: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    if db_path is not None:
        examples, assignments = load_from_sqlite(db_path)
    elif examples_path is not None and assignments_path is not None:
        examples = load_standardized_examples(examples_path)
        assignments = load_client_assignments(assignments_path)
    else:
        raise ValueError("Provide either db_path or both examples_path and assignments_path.")

    train_examples = merge_examples_with_assignments(
        examples,
        assignments,
        split="train",
        heterogeneity_level=heterogeneity_level,
    )
    val_examples = merge_examples_with_assignments(
        examples,
        assignments,
        split="val",
        heterogeneity_level=heterogeneity_level,
    )

    train_grouped = group_examples_by_client(train_examples)
    val_grouped = group_examples_by_client(val_examples)
    train_loaders = build_client_dataloaders(
        train_grouped,
        tokenizer=tokenizer,
        tokenization_config=tokenization_config,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    val_loaders = build_client_dataloaders(
        val_grouped,
        tokenizer=tokenizer,
        tokenization_config=tokenization_config,
        batch_size=eval_batch_size,
        shuffle=False,
    )
    train_example_counts = {
        client_id: len(client_examples)
        for client_id, client_examples in train_grouped.items()
    }
    return train_loaders, val_loaders, train_example_counts


def _compute_scaffold_control_delta(
    *,
    global_state: dict[str, Any],
    local_state: dict[str, Any],
    server_control: dict[str, Any],
    client_control: dict[str, Any],
    learning_rate: float,
    local_steps: int,
) -> dict[str, Any]:
    effective_denominator = max(learning_rate * local_steps, 1e-12)
    model_delta = scale_trainable_state(
        subtract_trainable_states(global_state, local_state),
        1.0 / effective_denominator,
    )
    adjusted_control = subtract_trainable_states(client_control, server_control)
    new_client_control = add_trainable_states(adjusted_control, model_delta)
    return subtract_trainable_states(new_client_control, client_control)


def _summarize_round(
    *,
    round_index: int,
    selected_clients: tuple[str, ...],
    client_results: list[ClientRoundResult],
    evaluation: EvaluationResult,
    cosine_disagreements: list[float],
) -> RoundLog:
    total_examples = sum(result.example_count for result in client_results)
    total_steps = sum(result.train_result.steps_completed for result in client_results)
    weighted_train_loss = 0.0
    weighted_task_loss = 0.0
    for result in client_results:
        weighted_train_loss += result.train_result.mean_loss * result.example_count
        weighted_task_loss += result.train_result.mean_task_loss * result.example_count
    train_loss_mean = weighted_train_loss / total_examples if total_examples > 0 else 0.0
    train_task_loss_mean = weighted_task_loss / total_examples if total_examples > 0 else 0.0

    update_norms = [result.update_norm for result in client_results]
    update_norm_mean = sum(update_norms) / len(update_norms) if update_norms else 0.0
    update_norm_max = max(update_norms) if update_norms else 0.0
    cosine_disagreement_mean = (
        sum(cosine_disagreements) / len(cosine_disagreements) if cosine_disagreements else 0.0
    )
    cosine_disagreement_max = max(cosine_disagreements) if cosine_disagreements else 0.0
    return RoundLog(
        round_index=round_index,
        selected_clients=selected_clients,
        train_loss_mean=train_loss_mean,
        train_task_loss_mean=train_task_loss_mean,
        validation_loss=evaluation.mean_loss,
        update_norms=tuple(update_norms),
        update_norm_mean=update_norm_mean,
        update_norm_max=update_norm_max,
        cosine_disagreements=tuple(cosine_disagreements),
        cosine_disagreement_mean=cosine_disagreement_mean,
        cosine_disagreement_max=cosine_disagreement_max,
        total_client_examples=total_examples,
        total_local_steps=total_steps,
    )
