# Experiment Data Setup

This project keeps large raw datasets and model weights outside the repository, but it
tracks the experiment dataset definition inside the repo.

## Goal

Before training begins, we want four reproducible artifacts:

1. a metadata catalog built from the cloned SuperNI task JSON files
2. a pilot task manifest listing the exact SuperNI tasks selected for experiments
3. a standardized example CSV with plain-text `prompt` and `target`
4. one client assignment CSV per heterogeneity setting

## Files used for the data setup

- `configs/experiments/pilot_superni.toml`
- `configs/local.toml`
- `data/manifests/superni_task_catalog.csv`
- `data/manifests/pilot_superni_tasks.csv`
- `data/processed/standardized_examples.csv`
- `data/processed/client_assignments_low.csv`
- `data/processed/client_assignments_high.csv`

## Workflow

1. Clone the SuperNI metadata repo outside this repository.
2. Set `superni_metadata_repo` in `configs/local.toml`.
3. Build a local catalog from the metadata repo.
4. Review the generated catalog and choose the first pilot tasks.
5. Fill `data/manifests/pilot_superni_tasks.csv`.
6. Build `data/processed/standardized_examples.csv` with the schema below.
7. Build client assignment CSVs for each heterogeneity setting.
8. Run the local training smoke test against one client.

## Pilot selection guidance

Try to choose:

- `10-12` tasks total
- `5-8` task types
- tasks with enough examples to support your target per-task sample budget
- a task mix that will support mild, medium, and hard heterogeneity partitions

## Manifest columns

- `task_id`: the task JSON stem from SuperNI metadata
- `task_type`: the high-level bucket you want to use for heterogeneity control
- `samples_target`: the target number of examples to pull for that task
- `notes`: optional comments for the team

## Standardized example schema

`data/processed/standardized_examples.csv`

- `example_id`
- `task_id`
- `task_type`
- `prompt`
- `target`
- `split`

## Client assignment schema

`data/processed/client_assignments_low.csv`
or
`data/processed/client_assignments_high.csv`

- `example_id`
- `client_id`
- `heterogeneity_level`

## Smoke test command

```powershell
python scripts/run_local_client_train.py --peft-method lora --max-steps 2
```

## Federated simulation commands

Single configuration:

```powershell
python scripts/run_federated_simulation.py --aggregation-method fedavg --peft-method lora --heterogeneity-level low --participation-fraction 0.3 --seed 7
```

Reduced pilot grid:

```powershell
python scripts/run_experiment_grid.py
```
