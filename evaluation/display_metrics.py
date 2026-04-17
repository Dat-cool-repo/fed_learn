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
    "val_loss": "Test Loss",
    "update_norm_mean": "Update Norm",
    "cosine_disagreement_mean": "Cosine Disagreement",
    "cosine_disagreement_std": "Cosine Disagreement Std",
    "soft_prompt": "Soft Prompt",
    "lora": "LoRA",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "scaffold": "SCAFFOLD",
    "high": "High Heterogeneity",
    "low": "Low Heterogeneity",
    "heterogeneity_level": "Heterogeneity Level",
    "aggregation_method": "Aggregation Method",
    "peft_method": "PEFT Method",
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
    round_col="round",
    seed_col="seed",
    title=None
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    groupings = [
        ("aggregation_method", axes[0]),
        ("peft_method", axes[1]),
    ]

    for group_by, ax in groupings:
        # Only exclude scaffold from the PEFT plot
        if group_by == "peft_method":
            plot_df = df[df["aggregation_method"] != "scaffold"].copy()
        else:
            plot_df = df.copy()
        
        grouped = plot_df.groupby([group_by, round_col])
        stats = grouped[metric].agg(["mean", "std"]).reset_index()

        for method in stats[group_by].unique():
            subset = stats[stats[group_by] == method].sort_values(round_col)

            x = subset[round_col]
            y = subset["mean"]
            std = subset["std"].fillna(0)

            ax.plot(
                x,
                y,
                label=LABEL_MAP.get(method, method)
            )
            ax.fill_between(
                x,
                y - std,
                y + std,
                alpha=0.2
            )
        if group_by == "peft_method":
            ax.set_title("PEFT Method (Excluding SCAFFOLD)")
        else:
            ax.set_title(LABEL_MAP.get(group_by, group_by))

        ax.set_xlabel("Round")
        ax.grid(True, alpha=0.3)

        ax.legend(loc="best", fontsize=9)

    axes[0].set_ylabel(LABEL_MAP.get(metric, metric))

    fig.suptitle(
        title or f"{LABEL_MAP.get(metric, metric)}: Mean ± Std Across Seeds",
        fontsize=14
    )

    plt.tight_layout()
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
    Top plot: Regression frequency
    Bottom plot: Regression magnitude
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

    # ---- TWO SUBPLOTS ----
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(10, 8),
        sharex=True
    )

    methods = agg[group_by].unique()
    colors = plt.cm.tab10.colors
    color_map = {m: colors[i % len(colors)] for i, m in enumerate(methods)}

    for method in methods:
        sub = agg[agg[group_by] == method].sort_values("round")
        c = color_map[method]

        # ---- TOP: FREQUENCY ----
        ax1.plot(
            sub["round"],
            sub["regression_freq"],
            color=c,
            linestyle="-",
            label=LABEL_MAP.get(method, method)
        )

        # ---- BOTTOM: MAGNITUDE ----
        ax2.plot(
            sub["round"],
            sub["regression_mag"],
            color=c,
            linestyle="-",
            label=LABEL_MAP.get(method, method)
        )

    # ---- LABELING ----
    ax1.set_ylabel("Regression Frequency")
    ax1.set_title("Regression Frequency")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Round")
    ax2.set_ylabel("Regression Magnitude")
    ax2.set_title("Regression Magnitude")
    ax2.grid(True, alpha=0.3)

    # ---- SINGLE LEGEND (top plot only) ----
    ax1.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
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

    for metric in METRICS:
        plt.figure(figsize=(6, 4))

        for val, g in agg.groupby(factor):
            g = g.sort_values("round")
            plt.plot(g["round"], g[metric], label=LABEL_MAP.get(val, val))

        plt.title(LABEL_MAP.get(metric, metric))
        plt.xlabel("Round")
        plt.ylabel(LABEL_MAP.get(metric, metric))
        plt.legend()

        # axis limits
        if metric == "val_loss":
            plt.ylim(0, 14) # 8
        elif metric == "train_loss":
            plt.ylim(0, 12) # 5
        elif metric == "update_norm_mean":
            plt.ylim(0, 600) # 20
        elif metric == "cosine_disagreement_mean":
            plt.ylim(0, 0.8) # 0.6
        elif metric == "cosine_disagreement_std":
            plt.ylim(0, 0.6) # 0.25

        plt.tight_layout()
        plt.show()

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def plot_cosine_disagreement_by_factor(df, factor):
    df = df.copy()
    df["round"] = df["round"].astype(int)

    metric = "cosine_disagreement_mean"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, het_level in zip(axes, ["low", "high"]):
        sub = df[df["heterogeneity_level"] == het_level].copy()

        agg = (
            sub.groupby([factor, "round"], as_index=False)
               .mean(numeric_only=True)
        )

        for val, g in agg.groupby(factor):
            g = g.sort_values("round")
            ax.plot(
                g["round"],
                g[metric],
                label=LABEL_MAP.get(val, val)
            )

        ax.set_title(f"Heterogeneity: {het_level}")
        ax.set_xlabel("Round")
        ax.set_ylim(0, 0.8)

    axes[0].set_ylabel(LABEL_MAP.get(metric, metric))

    # collect legend handles/labels from first axis
    handles, labels = axes[0].get_legend_handles_labels()

    # figure-level legend in top-right corner
    fig.legend(handles, labels, loc="upper right")

    # overall title
    fig.suptitle("Cosine Disagreement")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def get_final_rouge(df, factor):
    df = df.copy()
    df["round"] = df["round"].astype(int)

    # get last round per factor
    final = df.loc[df.groupby(factor)["round"].idxmax()]

    return final[[factor, "rouge_l"]]

def plot_rouge_bar(df, factor):
    agg = get_final_rouge(df, factor)

    plt.figure(figsize=(6, 4))

    x = agg[factor]
    y = agg["rouge_l"]

    plt.bar(x, y)

    plt.title(f"ROUGE-L by {LABEL_MAP.get(factor, factor)}")
    plt.xlabel(LABEL_MAP.get(factor, factor))
    plt.ylabel("ROUGE-L")
    plt.ylim(0, 1)

    # nicer labels if you have mappings
    if factor in LABEL_MAP:
        plt.xticks(ticks=range(len(x)), labels=[LABEL_MAP.get(v, v) for v in x])

    plt.tight_layout()
    plt.show()

def plot_all(df):
    for factor in HYPERPARAMS:
        plot_by_factor(df, factor)
        plot_rouge_bar(df, factor)

def plot_all_rouge(df):
    df = df.copy()
    df["round"] = df["round"].astype(int)

    fig, axes = plt.subplots(1, len(HYPERPARAMS), figsize=(15, 4))
    fig.suptitle("ROUGE-L", fontsize=16)

    for ax, factor in zip(axes, HYPERPARAMS):
        # get final round per factor
        final = df.loc[df.groupby(factor)["round"].idxmax()]

        x = final[factor]
        y = final["rouge_l"]

        ax.bar(range(len(x)), y)

        ax.set_title(LABEL_MAP.get(factor, factor))
        ax.set_xlabel(LABEL_MAP.get(factor, factor))
        ax.set_ylabel("ROUGE-L")
        ax.set_ylim(0, 1)

        # nicer labels
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels([LABEL_MAP.get(v, v) for v in x], rotation=20)

    plt.tight_layout()
    plt.show()

def plot_convergence_without_scaffold(df):
    df = df.copy()
    df["round"] = df["round"].astype(int)

    # REMOVE only scaffold rows
    df = df[df["aggregation_method"] != "scaffold"]

    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    axes = [ax1, ax2]

    axes[0].axhline(y=11.61, color="red", linestyle="--", label="Base Model")
    axes[1].axhline(y=11.61, color="red", linestyle="--", label="Base Model")

    fig.suptitle("Convergence Behavior Excluding SCAFFOLD", fontsize=16)

    for ax, factor in zip(axes, HYPERPARAMS):
        if (factor == "heterogeneity_level"):
            continue

        agg = (
            df.groupby([factor, "round"], as_index=False)
              .mean(numeric_only=True)
        )

        for val, g in agg.groupby(factor):
            g = g.sort_values("round")
            ax.plot(
                g["round"],
                g["val_loss"],
                label=LABEL_MAP.get(val, val)
            )

        ax.set_title(f"Test Loss by {LABEL_MAP.get(factor, factor)}")
        ax.set_xlabel("Round")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    axes[0].set_ylabel("Test Loss")

    plt.show()

def plot_targeted_convergence_comparisons(df):
    df = df.copy()
    df["round"] = df["round"].astype(int)

    # remove scaffold
    df = df[df["aggregation_method"] != "scaffold"]

    # ---- helper for aggregation ----
    def get_agg(sub_df, group_col):
        return (
            sub_df.groupby([group_col, "round"], as_index=False)
                  .mean(numeric_only=True)
        )

    # =========================================================
    # FIGURE 1: LoRA vs Soft Prompt (fixed aggregation)
    # =========================================================
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig1.suptitle("LoRA vs Soft Prompt (by Aggregation Method)", fontsize=16)

    for ax, method in zip([ax1, ax2], ["fedavg", "fedprox"]):
        sub = df[df["aggregation_method"] == method]
        agg = get_agg(sub, "peft_method")

        ax.axhline(y=11.61, color="red", linestyle="--", label="Base Model")

        for val, g in agg.groupby("peft_method"):
            g = g.sort_values("round")
            ax.plot(
                g["round"],
                g["val_loss"],
                label=LABEL_MAP.get(val, val)
            )

        ax.set_title(f"{LABEL_MAP.get(method, method)}")
        ax.set_xlabel("Round")
        ax.grid(True, alpha=0.3)
        ax.legend()

    ax1.set_ylabel("Test Loss")

    plt.tight_layout()
    plt.show()

    # =========================================================
    # FIGURE 2: FedAvg vs FedProx (fixed PEFT method)
    # =========================================================
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig2.suptitle("FedAvg vs FedProx (by PEFT Method)", fontsize=16)

    for ax, peft in zip([ax3, ax4], ["lora", "soft_prompt"]):
        sub = df[df["peft_method"] == peft]
        agg = get_agg(sub, "aggregation_method")

        ax.axhline(y=11.61, color="red", linestyle="--", label="Base Model")

        for val, g in agg.groupby("aggregation_method"):
            g = g.sort_values("round")
            ax.plot(
                g["round"],
                g["val_loss"],
                label=LABEL_MAP.get(val, val)
            )

        ax.set_title(f"{LABEL_MAP.get(peft, peft)}")
        ax.set_xlabel("Round")
        ax.grid(True, alpha=0.3)
        ax.legend()

    ax3.set_ylabel("Test Loss")

    plt.tight_layout()
    plt.show()



def plot_aggregation_methods(df):
    df = df.copy()
    df["round"] = df["round"].astype(int)

    fig, ax = plt.subplots(figsize=(8, 5))

    agg = (
        df.groupby(["aggregation_method", "round"], as_index=False)
          .mean(numeric_only=True)
    )

    for val, g in agg.groupby("aggregation_method"):
        g = g.sort_values("round")
        ax.plot(
            g["round"],
            g["val_loss"],
            label=LABEL_MAP.get(val, val)
        )

    plt.axhline(y=11.61, color="red", linestyle="--", label="Base Model")

    ax.set_title("Test Loss by Aggregation Method")
    ax.set_xlabel("Round")
    ax.set_ylabel("Test Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_rouge_l_heatmaps(df, rouge_col="rouge_l"):
    df = df.copy()

    # aggregate
    agg = (
        df.groupby(["heterogeneity_level", "peft_method", "aggregation_method"])
          .agg({rouge_col: "mean"})
          .reset_index()
    )

    heterogeneity_levels = sorted(agg["heterogeneity_level"].unique())

    fig, axes = plt.subplots(
        1, len(heterogeneity_levels),
        figsize=(6 * len(heterogeneity_levels), 5),
        squeeze=False
    )

    fig.suptitle("Average ROUGE-L Scores Across Experimental Configurations", fontsize=16)
    vmin = agg[rouge_col].min()
    vmax = agg[rouge_col].max()

    for ax, h in zip(axes[0], heterogeneity_levels):
        sub = agg[agg["heterogeneity_level"] == h]

        pivot = sub.pivot(
            index="peft_method",
            columns="aggregation_method",
            values=rouge_col
        )

        data = pivot.values
        im = ax.imshow(
            data,
            cmap="viridis",
            aspect="auto",
            vmin=vmin,
            vmax=vmax
        )

        # ticks
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))

        ax.set_xticklabels([LABEL_MAP.get(col, col) for col in pivot.columns])
        ax.set_yticklabels([LABEL_MAP.get(idx, idx) for idx in pivot.index])

        ax.set_title(f"Heterogeneity: {h}")
        ax.set_xlabel("Aggregation Method")
        ax.set_ylabel("PEFT Method")

        # annotate values
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                ax.text(
                    j, i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    color="white" if val < data.max() * 0.6 else "black"
                )

        # colorbar per subplot
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.subplots_adjust(wspace=0.7, top=0.85)
    plt.show()
