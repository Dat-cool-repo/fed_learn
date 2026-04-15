import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HYPERPARAMS = [
    "aggregation_method",
    "peft_method",
    "heterogeneity_level",
]

METRICS = [
    "train_loss",
    "val_loss",
    "update_norm_mean",
    "cosine_disagreement_mean",
    "cosine_disagreement_std"
]

LABEL_MAP = {
    "train_loss": "Training Loss",
    "val_loss": "Validation Loss",
    "update_norm_mean": "Update Norm",
    "cosine_disagreement_mean": "Cosine Disagreement",
    "cosine_disagreement_std": "Cosine Disagreement Std",
    "soft_prompt": "Soft Prompt",
    "lora": "LoRA",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "scaffold": "SCAFFOLD",
    "high": "High Heterogeneity",
    "low": "Low Heterogeneity"
}

def aggregate_across_seeds(df):
    group_cols = [
        "aggregation_method",
        "peft_method",
        "heterogeneity_level",
        "participation_fraction",
        "round"
    ]

    agg_df = df.groupby(group_cols, as_index=False).mean(numeric_only=True)
    agg_df = agg_df.drop(columns=["seed"])

    return agg_df

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_seed_variance(
    df,
    metric="val_loss",
    group_by="aggregation_method",
    round_col="round",
    seed_col="seed",
    title=None
):
    grouped = df.groupby([group_by, round_col])

    stats = grouped[metric].agg(["mean", "std"]).reset_index()

    plt.figure(figsize=(10, 6))

    for method in stats[group_by].unique():
        subset = stats[stats[group_by] == method].sort_values(round_col)

        x = subset[round_col]
        y = subset["mean"]
        std = subset["std"].fillna(0)

        plt.plot(x, y, label=f"{LABEL_MAP.get(method, method)}")
        plt.fill_between(
            x,
            y - std,
            y + std,
            alpha=0.2
        )

    plt.xlabel("Round")
    plt.ylabel(LABEL_MAP.get(metric, metric))
    plt.title(title or f"{LABEL_MAP.get(metric, metric)}: Mean ± Std Across Seeds")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_regressions(
    df,
    metric="val_loss",
    group_by="aggregation_method",
    round_col="round",
    seed_col="seed",
    run_col="run_name",
):
    """
    Regression frequency (solid) + magnitude (dashed)
    Color = method
    """

    results = []

    for (run_name, seed), sub in df.groupby([run_col, seed_col]):

        sub = sub.sort_values(round_col)

        for i in range(len(sub) - 1):
            t = sub.iloc[i][round_col]
            t_next = sub.iloc[i + 1][round_col]

            if t_next != t + 1:
                continue

            diff = sub.iloc[i + 1][metric] - sub.iloc[i][metric]

            results.append({
                "run_name": run_name,
                "seed": seed,
                group_by: sub.iloc[0][group_by],
                "round": t_next,
                "regression": diff < 0,
                "magnitude": abs(diff) if diff < 0 else 0.0
            })

    reg_df = pd.DataFrame(results)

    if reg_df.empty:
        print("No regression data found.")
        return reg_df

    agg = (
        reg_df.groupby([group_by, "round"])
        .agg(
            regression_freq=("regression", "mean"),
            regression_mag=("magnitude", "mean")
        )
        .reset_index()
    )

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    methods = agg[group_by].unique()
    colors = plt.cm.tab10.colors

    color_map = {m: colors[i % len(colors)] for i, m in enumerate(methods)}

    # ---- plot ----
    for method in methods:
        sub = agg[agg[group_by] == method].sort_values("round")
        c = color_map[method]

        # frequency (solid)
        ax1.plot(
            sub["round"],
            sub["regression_freq"],
            color=c,
            linestyle="-",
        )

        # magnitude (dashed)
        ax2.plot(
            sub["round"],
            sub["regression_mag"],
            color=c,
            linestyle="--",
            alpha=0.8
        )

    ax1.set_xlabel("Round")
    ax1.set_ylabel("Regression Frequency")
    ax2.set_ylabel("Regression Magnitude")
    ax1.set_title(f"Regression Dynamics: {LABEL_MAP.get(metric, metric)}")
    ax1.grid(True, alpha=0.3)

    # ---- CUSTOM LEGEND (correct encoding) ----
    legend_elements = []

    # method colors
    for method in methods:
        legend_elements.append(
            Line2D([0], [0], color=color_map[method], lw=2, label=LABEL_MAP.get(method, method))
        )

    # linestyle meaning
    legend_elements.append(Line2D([0], [0], color="black", lw=2, linestyle="-", label="Frequency"))
    legend_elements.append(Line2D([0], [0], color="black", lw=2, linestyle="--", label="Magnitude"))

    ax1.legend(handles=legend_elements, loc="upper right")

    plt.show()

    return reg_df, agg

def plot_by_factor(df, factor):
    df = df.copy()
    df["round"] = df["round"].astype(int)

    # fully marginalize everything except (factor, round)
    agg = (
        df.groupby([factor, "round"], as_index=False)
          .mean(numeric_only=True)
    )

    n_metrics = len(METRICS)
    n_cols = 2
    n_rows = math.ceil(n_metrics / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), sharex=True)

    axes = np.atleast_1d(axes).flatten()

    if len(METRICS) == 1:
        axes = [axes]

    for ax, metric in zip(axes, METRICS):
        for val, g in agg.groupby(factor):
            g = g.sort_values("round")
            ax.plot(g["round"], g[metric], label=LABEL_MAP.get(val, val))

        ax.set_title(LABEL_MAP.get(metric, metric))
        ax.set_xlabel("Round")
        ax.set_ylabel(LABEL_MAP.get(metric, metric))
        ax.legend()

    for ax in axes[len(METRICS):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def plot_all(df):
    for factor in HYPERPARAMS:
        plot_by_factor(df, factor)