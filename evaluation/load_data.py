import os
import json
import numpy as np
from collections import defaultdict

def load_experiment_folder(folder_path):
    data = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            filepath = os.path.join(folder_path, filename)

            with open(filepath, "r") as f:
                run_data = json.load(f)
                data.append(run_data)

    return data

import pandas as pd

def runs_to_dataframe(runs):
    rows = []

    for run in runs:
        for i, round_num in enumerate(run["rounds"]):
            update_vals = run["update_norms_per_round"][i]
            cos_vals = run["cosine_disagreement_per_round"][i]

            # Add ROUGE-L score to data
            rouge_val = run["rouge_l_per_round"][i]

            row = {
                "run_name": run["run_name"],
                "aggregation_method": run["aggregation_method"],
                "peft_method": run["peft_method"],
                "heterogeneity_level": run["heterogeneity_level"],
                "participation_fraction": run["participation_fraction"],
                "seed": run["seed"],
                "round": round_num,

                "train_loss": run["train_loss_per_round"][i],
                "val_loss": run["validation_loss_per_round"][i],
                "rouge_l": rouge_val,

                "update_norm_mean": np.mean(update_vals),
                
                "cosine_disagreement_mean": np.mean(cos_vals),
                "cosine_disagreement_std": np.std(cos_vals),
            }

            rows.append(row)

    return pd.DataFrame(rows)

def print_all_run_names(runs):
    groups = defaultdict(list)

    for run in runs:
        key = (
            run["seed"],
            run["aggregation_method"],
            run["peft_method"],
            run["heterogeneity_level"],
            run["participation_fraction"],
        )
        groups[key].append(run["run_name"])

    print("--- RUNS LOADED ---")

    for key in sorted(groups.keys(), key=lambda k: (k[0], k[1], k[2], k[3])):
        for run_name in groups[key]:
            print(f"  {run_name}")