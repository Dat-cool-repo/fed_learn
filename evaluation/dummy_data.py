import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from itertools import product
from scipy.stats import pearsonr
from rouge_score import rouge_scorer


# ── Experiment grid ──────────────────────────────────────────────────────────
SEEDS        = [1, 2]
REGIMES      = ["mild", "hard"]
METHODS      = ["fedavg", "fedprox", "scaffold"]
PEFT_METHODS = ["lora", "softprompt"]

# Agreed results folder schema (sync with B before Day 3)
# results/seed_{s}/regime_{r}/method_{m}/peft_{p}/metrics.json
RESULTS_ROOT = Path("results")

# Training rounds (set to match B's checkpoint cadence)
MAX_ROUNDS = 50   # adjust to actual run; plots will truncate gracefully

# Drift metric: regression threshold ε
REGRESSION_EPS = 0.01

# ── Color palette (consistent across all figures) ────────────────────────────
METHOD_COLORS = {
    "fedavg":   "#E15A41",   # warm red
    "fedprox":  "#4A90D9",   # steel blue
    "scaffold": "#27AE60",   # emerald
}
PEFT_LINESTYLES = {
    "lora":       "-",
    "softprompt": "--",
}
REGIME_TITLES = {
    "mild":   "Mild heterogeneity",
    "medium": "Medium heterogeneity",
    "hard":   "Hard heterogeneity",
}

print("Config set.")
print(f"Total experiments in grid: {len(SEEDS)*len(REGIMES)*len(METHODS)*len(PEFT_METHODS)}")

def make_dummy_metrics(
    n_rounds: int = 20,
    n_clients: int = 10,
    method: str = "fedavg",
    regime: str = "mild",
    peft: str = "lora",
    seed: int = 1,
 ) -> dict:
    rng = np.random.default_rng(seed * 100 + hash(method + regime + peft) % 997)

    regime_penalty = {"mild": 0.0, "medium": -0.03, "hard": -0.08}[regime]
    method_bonus   = {"fedavg": 0.0, "fedprox": 0.03, "scaffold": 0.05}[method]
    peft_offset    = {"lora": 0.0, "softprompt": -0.02}[peft]

    base_rouge = 0.25 + method_bonus + regime_penalty + peft_offset
    noise_scale = 0.02

    # ROUGE-L: starts lower, climbs with noise
    progress = np.linspace(0, 1, n_rounds) ** 0.6
    rouge_l = base_rouge * progress + rng.normal(0, noise_scale, n_rounds)
    rouge_l = np.clip(rouge_l, 0, 1).tolist()

    # Update norms: decay over rounds (drift should reduce with correction)
    norm_base = {"fedavg": 0.8, "fedprox": 0.5, "scaffold": 0.4}[method]
    norm_decay = np.exp(-np.linspace(0, 2, n_rounds))
    update_norms = []
    for t in range(n_rounds):
        client_norms = (norm_base * norm_decay[t]
                        + rng.normal(0, 0.05, n_clients)).clip(0).tolist()
        update_norms.append(client_norms)

    # Cosine disagreement: should be lower for correction methods
    cd_base = {"fedavg": 0.6, "fedprox": 0.35, "scaffold": 0.25}[method]
    cosine_dis = []
    for t in range(n_rounds):
        cd = (cd_base * norm_decay[t]
              + rng.normal(0, 0.04, n_clients)).clip(0, 1).tolist()
        cosine_dis.append(cd)

    return {
        "rounds": list(range(1, n_rounds + 1)),
        "rouge_l_per_round": rouge_l,
        "update_norms_per_round": update_norms,
        "cosine_disagreement_per_round": cosine_dis,
    }


def write_dummy_results(root: Path = RESULTS_ROOT, n_rounds: int = 20):
    """Write dummy metrics.json files into the full results folder tree."""
    root.mkdir(parents=True, exist_ok=True)
    for seed, regime, method, peft in product(SEEDS, REGIMES, METHODS, PEFT_METHODS):
        path = root / f"seed_{seed}" / f"regime_{regime}" / f"method_{method}" / f"peft_{peft}"
        path.mkdir(parents=True, exist_ok=True)
        metrics = make_dummy_metrics(n_rounds, method=method, regime=regime,
                                     peft=peft, seed=seed)
        with open(path / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
    print(f"Dummy results written to '{root}/'")
    print(f"Files: {len(list(root.rglob('metrics.json')))} metrics.json")