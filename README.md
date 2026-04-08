# Federated PEFT Research Project

This repository is organized for the empirical phase of the CIS4930 research project on
drift-corrected federated instruction tuning with PEFT.

## Layout

```text
fed_learn/
  configs/              Experiment, model, and local path configuration files
  data/                 Local manifests and dataset notes
  docs/                 Setup notes and project documentation
  outputs/              Generated results, logs, and artifacts
  scripts/              Smoke tests and utility scripts
  src/fed_learn/        Main project package
  tests/                Unit and integration tests
```

## Current starting point

- `configs/experiments/pilot_superni.toml` captures the first experiment defaults.
- `configs/models/qwen_1_5b_instruct.toml` captures the temporary base model setup.
- `scripts/check_setup.py` validates dataset, metadata-path, and tokenizer access.
- `scripts/show_pilot_plan.py` prints the proposal-backed pilot plan.
- The team is using `Qwen/Qwen2.5-1.5B-Instruct` as the temporary base model.

## Local machine setup

Copy `configs/local.example.toml` to `configs/local.toml` and fill in your machine-specific
paths. Large assets stay outside the repo, for example:

- SuperNI metadata: `C:\Users\<name>\datasets\natural-instructions-meta`
- Hugging Face cache: default cache or a custom external directory
- Output artifacts: inside `outputs/` or another non-versioned folder

## Useful commands

```powershell
python scripts/show_pilot_plan.py
python scripts/check_superni_access.py
python scripts/check_setup.py --skip-model
python scripts/check_setup.py
```

## Immediate next steps

1. Choose the initial SuperNI task subset.
2. Define mild, medium, and hard client mixtures.
3. Build the dataset catalog and client partitioning code.
