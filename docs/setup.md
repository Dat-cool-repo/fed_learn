# Team Setup

## Environment

Create and activate the conda environment:

```powershell
conda create -n strats python=3.11 -y
conda activate strats
pip install --upgrade pip
pip install "huggingface_hub<1.0" "transformers>=4.43,<5" datasets accelerate peft torch
```

## Dataset

Clone SuperNI metadata outside the repository:

```powershell
git clone https://github.com/allenai/natural-instructions C:\Users\<YOUR_NAME>\datasets\natural-instructions-meta
```

Verify SuperNI access:

```powershell
python scripts/check_superni_access.py
python scripts/build_superni_catalog.py
python scripts/show_data_setup.py
```

Create local path config:

```powershell
Copy-Item configs\local.example.toml configs\local.toml
```

Then edit `configs\local.toml` with the correct metadata path for your machine.

## Model

Use Qwen for the current setup:

- default: `Qwen/Qwen2.5-0.5B-Instruct`
- backup/heavier option: `Qwen/Qwen2.5-1.5B-Instruct`

Verify tokenizer access:

```powershell
python scripts/show_model_setup.py --peft-method lora
python scripts/show_model_setup.py --peft-method lora --load
python scripts/run_local_client_train.py --peft-method lora --max-steps 2
```

## Completion checklist

Setup is ready when each teammate can confirm:

1. `conda activate strats` works.
2. SuperNI streaming loads successfully.
3. The metadata repo exists outside the project.
4. The Qwen tokenizer loads without error.
5. `python scripts/show_model_setup.py --peft-method lora` prints the shared model defaults.
6. `python scripts/show_model_setup.py --peft-method lora --load` successfully loads the reduced Qwen stack.
7. `python scripts/run_local_client_train.py --peft-method lora --max-steps 2` can train one simulated client from CSV input.
