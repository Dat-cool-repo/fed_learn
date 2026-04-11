# Federated PEFT Research Project

This repository is currently focused on the model-side scaffold for federated PEFT
experiments.

## Current Layout

```text
fed_learn/
  configs/                    Shared model and local path configuration files
  docs/                       Notes and timeline references
  scripts/show_model_setup.py Model-only dry-run and load example
  src/fed_learn/              Model config, loading, and PEFT state utilities
```

## Model Scaffold

- `configs/models/qwen_1_5b_instruct.toml` defines the temporary base model and PEFT settings.
- `src/fed_learn/config.py` loads typed config objects from TOML files.
- `src/fed_learn/modeling.py` loads the tokenizer/base model and attaches `LoRA` or
  `soft_prompt`.
- `src/fed_learn/peft_state.py` extracts, reloads, and averages trainable PEFT tensors.
- `scripts/show_model_setup.py` demonstrates the intended model workflow before wiring in
  data or local training.

## Useful Commands

```powershell
python scripts/show_model_setup.py --peft-method lora
python scripts/show_model_setup.py --peft-method soft_prompt
python scripts/show_model_setup.py --peft-method lora --load
```

## Next Step

Add the first local training loop around the model scaffold so client-side updates can be
plugged into `FedAvg`, `FedProx`, and `SCAFFOLD`.
