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

- `Qwen/Qwen2.5-1.5B-Instruct`

Verify tokenizer access:

```powershell
python scripts/check_setup.py
```

## Completion checklist

Setup is ready when each teammate can confirm:

1. `conda activate strats` works.
2. SuperNI streaming loads successfully.
3. The metadata repo exists outside the project.
4. The Qwen tokenizer loads without error.
5. `python scripts/show_pilot_plan.py` prints the shared pilot defaults.
6. `python scripts/build_superni_catalog.py` creates the local SuperNI catalog.
