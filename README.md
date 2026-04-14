# Federated PEFT Research Project

This repository now includes the first end-to-end local data-to-model path for federated
PEFT experiments.

## Current Layout

```text
fed_learn/
  configs/                    Shared model and local path configuration files
  data/processed/             Local standardized examples and client assignment CSVs
  docs/                       Notes and timeline references
  scripts/                    Model setup and local training smoke tests
  src/fed_learn/              Model, data, and local training utilities
```

## Current Pipeline

- `configs/models/qwen_0_5b_instruct.toml` is the default base model and PEFT config for the reduced pilot.
- `configs/models/qwen_1_5b_instruct.toml` is kept as the heavier backup option.
- `configs/experiments/pilot_superni.toml` now reflects the reduced pilot:
  `5 tasks`, `15 clients`, `10 rounds`, `1 seed`, `2 heterogeneity levels`.
- `src/fed_learn/config.py` loads typed config objects from TOML files.
- `src/fed_learn/modeling.py` loads the tokenizer/base model and configures `fft`,
  `LoRA`, or `soft_prompt`.
- `src/fed_learn/data_pipeline.py` loads standardized CSV examples, joins client
  assignments, and tokenizes prompt/target pairs for Qwen causal-LM training.
- `src/fed_learn/local_train.py` runs a single local client training loop over the
  tokenized batches.
- `src/fed_learn/federated.py` simulates federated rounds, client sampling, `FedAvg`,
  `FedProx`, `SCAFFOLD`, per-round evaluation, JSONL logging, metrics export, and
  checkpoint snapshots for later expensive evaluation.
- `src/fed_learn/peft_state.py` extracts, reloads, and averages whichever trainable
  tensors the selected adaptation method exposes.
- `scripts/show_model_setup.py` demonstrates the model workflow.
- `scripts/run_local_client_train.py` smoke-tests one client from CSV input through a
  local optimizer step loop.
- `scripts/run_federated_simulation.py` runs one federated configuration.
- `scripts/run_experiment_grid.py` runs the reduced experiment grid.

## Useful Commands

```powershell
python scripts/show_model_setup.py --peft-method lora
python scripts/show_model_setup.py --peft-method fft
python scripts/show_model_setup.py --peft-method soft_prompt
python scripts/show_model_setup.py --peft-method lora --load
python scripts/run_local_client_train.py --peft-method lora --max-steps 2
python scripts/repartition_sqlite_clients.py --clients-per-group 3
python scripts/run_federated_simulation.py --aggregation-method fedavg --peft-method fft --heterogeneity-level low
python scripts/run_experiment_grid.py
```

## Run Outputs

Each federated run writes to:

```text
outputs/federated/<run-name>/
  round_logs.jsonl
  metrics.json
  checkpoint.pt
  checkpoints/
    round_0005.pt
    round_0010.pt
```

- `round_logs.jsonl` stores per-round loss, selected clients, update norms, and cosine disagreement.
- `metrics.json` stores per-round arrays that are easy to plot or enrich later with `ROUGE-L`.
- `checkpoint.pt` is the rolling resume checkpoint.
- `checkpoints/round_XXXX.pt` stores selected snapshot rounds for later expensive evaluation.

## Next Step

Once the processed CSVs and evaluation details arrive, run one smoke-test configuration,
then iterate on metrics and benchmark integration.
