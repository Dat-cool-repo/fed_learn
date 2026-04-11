# Experiment Data Setup

This project keeps large raw datasets and model weights outside the repository, but it
tracks the experiment dataset definition inside the repo.

## Goal

Before training begins, we want three reproducible artifacts:

1. a metadata catalog built from the cloned SuperNI task JSON files
2. a pilot task manifest listing the exact SuperNI tasks selected for experiments
3. a shared experiment config that says how many tasks and samples each run expects

## Files used for the data setup

- `configs/experiments/pilot_superni.toml`
- `configs/local.toml`
- `data/manifests/superni_task_catalog.csv`
- `data/manifests/pilot_superni_tasks.csv`

## Workflow

1. Clone the SuperNI metadata repo outside this repository.
2. Set `superni_metadata_repo` in `configs/local.toml`.
3. Build a local catalog from the metadata repo.
4. Review the generated catalog and choose the first 10-12 pilot tasks.
5. Fill `data/manifests/pilot_superni_tasks.csv`.
6. Run the data summary script to verify the selected task count and task-type spread.

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

