"""
GradCAM branch attribution analysis plots.

Reads best_model_{split}_gradcam_3branch.csv for train / val_id / val_ood
and writes individual PNGs to ./plots/.

Plots
-----
1. mean_branch_by_split.png        — stacked bar: mean ptv3/ae/sinr per split
2. branch_by_species_heatmap.png   — heatmap: mean branch proportion per species x split
3. branch_by_dataset.png           — grouped bar: mean per source dataset (val_ood)
4. branch_distributions.png        — violin: distribution of each branch per split
5. correct_vs_wrong.png            — boxplot: branch proportions correct vs misclassified
6. confidence_vs_ptv3.png          — scatter: confidence vs ptv3_prop, coloured by correct/wrong
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── paths ─────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "plots")
os.makedirs(OUT_DIR, exist_ok=True)

SPLITS = ["train", "val_id", "val_ood"]
SPLIT_LABELS = {"train": "Train", "val_id": "Val (ID)", "val_ood": "Val (OOD)"}

BRANCH_COLS = ["ptv3_prop", "ae_prop", "sinr_prop"]
BRANCH_LABELS = ["PTv3", "AE context", "SINR context"]
BRANCH_COLORS = ["#4C72B0", "#DD8452", "#55A868"]

SPECIES_ORDER = [
    "Abies", "Acer", "Alnus", "Betula", "Carpinus",
    "Fagus", "Larix", "Picea", "Pinus", "Quercus",
]

DPI = 150


# ── data loading ──────────────────────────────────────────────────────────────
def load_all():
    dfs = []
    for split in SPLITS:
        path = os.path.join(HERE, f"best_model_{split}_gradcam_3branch.csv")
        df = pd.read_csv(path)
        df["split"] = split
        df["species"] = df["sample_name"].str.split("/").str[0]
        df["dataset"] = df["sample_name"].str.split("/").str[1].str.split("_").str[0]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# ── plot 1: mean branch proportions per split (stacked bar) ──────────────────
def plot_mean_by_split(df):
    means = (
        df.groupby("split")[BRANCH_COLS].mean().reindex(SPLITS)
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    bottom = np.zeros(len(SPLITS))
    x = np.arange(len(SPLITS))
    for col, label, color in zip(BRANCH_COLS, BRANCH_LABELS, BRANCH_COLORS):
        vals = means[col].values
        bars = ax.bar(x, vals, bottom=bottom, label=label, color=color, edgecolor="white", linewidth=0.5)
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0.04:
                ax.text(i, b + v / 2, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABELS[s] for s in SPLITS])
    ax.set_ylabel("Mean attribution proportion")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Mean branch attribution by split")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "01_mean_branch_by_split.png"), dpi=DPI)
    plt.close(fig)
    print("Saved 01_mean_branch_by_split.png")


# ── plot 2: heatmap per species (one column per branch x split) ───────────────
def plot_species_heatmap(df):
    # build a multi-column df: species × (branch, split)
    rows = []
    for split in SPLITS:
        sub = df[df["split"] == split]
        grp = sub.groupby("species")[BRANCH_COLS].mean()
        grp.columns = [f"{l}\n({SPLIT_LABELS[split]})" for l in BRANCH_LABELS]
        rows.append(grp)
    combined = pd.concat(rows, axis=1).reindex(SPECIES_ORDER)

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(
        combined,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Mean attribution proportion", "shrink": 0.7},
        annot_kws={"size": 7},
    )
    ax.set_title("Mean branch attribution per species and split")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "02_branch_by_species_heatmap.png"), dpi=DPI)
    plt.close(fig)
    print("Saved 02_branch_by_species_heatmap.png")


# ── plot 3: mean per source dataset (val_ood) ─────────────────────────────────
def plot_by_dataset(df):
    sub = df[df["split"] == "val_ood"]
    grp = sub.groupby("dataset")[BRANCH_COLS].mean().sort_values("ptv3_prop", ascending=False)

    datasets = grp.index.tolist()
    x = np.arange(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (col, label, color) in enumerate(zip(BRANCH_COLS, BRANCH_LABELS, BRANCH_COLORS)):
        ax.bar(x + (i - 1) * width, grp[col].values, width=width,
               label=label, color=color, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=25, ha="right")
    ax.set_ylabel("Mean attribution proportion")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)
    ax.set_title("Mean branch attribution per dataset (val OOD)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "03_branch_by_dataset.png"), dpi=DPI)
    plt.close(fig)
    print("Saved 03_branch_by_dataset.png")


# ── plot 4: violin distributions per split ───────────────────────────────────
def plot_distributions(df):
    # melt to long form
    long = df.melt(
        id_vars=["split"],
        value_vars=BRANCH_COLS,
        var_name="branch",
        value_name="proportion",
    )
    long["branch"] = long["branch"].map(dict(zip(BRANCH_COLS, BRANCH_LABELS)))
    long["split_label"] = long["split"].map(SPLIT_LABELS)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, split in zip(axes, SPLITS):
        sub = long[long["split"] == split]
        sns.violinplot(
            data=sub, x="branch", y="proportion",
            hue="branch", palette=dict(zip(BRANCH_LABELS, BRANCH_COLORS)),
            ax=ax, inner="box", cut=0, linewidth=0.8, legend=False,
        )
        ax.set_title(SPLIT_LABELS[split])
        ax.set_xlabel("")
        ax.set_ylabel("Attribution proportion" if ax == axes[0] else "")
        ax.set_ylim(-0.02, 1.05)
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle("Distribution of branch attributions", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "04_branch_distributions.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved 04_branch_distributions.png")


# ── plot 5: correct vs misclassified ─────────────────────────────────────────
def plot_correct_vs_wrong(df):
    long = df.melt(
        id_vars=["split", "correct"],
        value_vars=BRANCH_COLS,
        var_name="branch",
        value_name="proportion",
    )
    long["branch"] = long["branch"].map(dict(zip(BRANCH_COLS, BRANCH_LABELS)))
    long["outcome"] = long["correct"].map({1: "Correct", 0: "Wrong"})
    long["split_label"] = long["split"].map(SPLIT_LABELS)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    palette = {"Correct": "#2ca02c", "Wrong": "#d62728"}

    for ax, split in zip(axes, SPLITS):
        sub = long[long["split"] == split]
        sns.boxplot(
            data=sub, x="branch", y="proportion",
            hue="outcome", palette=palette,
            ax=ax, linewidth=0.8, fliersize=2,
            hue_order=["Correct", "Wrong"],
        )
        ax.set_title(SPLIT_LABELS[split])
        ax.set_xlabel("")
        ax.set_ylabel("Attribution proportion" if ax == axes[0] else "")
        ax.set_ylim(-0.02, 1.05)
        ax.tick_params(axis="x", labelsize=8)
        if ax != axes[-1]:
            ax.get_legend().remove()
        else:
            ax.legend(title="", fontsize=8, loc="upper right")

    fig.suptitle("Branch attributions: correct vs misclassified", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "05_correct_vs_wrong.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved 05_correct_vs_wrong.png")


# ── plot 6: confidence vs ptv3_prop scatter ───────────────────────────────────
def plot_confidence_vs_ptv3(df):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, sharex=True)
    palette = {1: "#2ca02c", 0: "#d62728"}
    labels = {1: "Correct", 0: "Wrong"}

    for ax, split in zip(axes, SPLITS):
        sub = df[df["split"] == split]
        for outcome in [1, 0]:
            s = sub[sub["correct"] == outcome]
            ax.scatter(
                s["ptv3_prop"], s["confidence"],
                c=palette[outcome], label=labels[outcome],
                alpha=0.35, s=8, linewidths=0,
            )
        # regression line
        x = sub["ptv3_prop"].values
        y = sub["confidence"].values
        m, b = np.polyfit(x, y, 1)
        xr = np.linspace(x.min(), x.max(), 100)
        ax.plot(xr, m * xr + b, color="black", linewidth=1.2, linestyle="--")

        ax.set_title(SPLIT_LABELS[split])
        ax.set_xlabel("PTv3 proportion")
        ax.set_ylabel("Confidence" if ax == axes[0] else "")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)

        corr = np.corrcoef(x, y)[0, 1]
        ax.text(0.05, 0.93, f"r = {corr:.2f}", transform=ax.transAxes, fontsize=8)

        if ax == axes[-1]:
            handles = [mpatches.Patch(color=palette[k], label=labels[k]) for k in [1, 0]]
            ax.legend(handles=handles, fontsize=8, loc="lower right")

    fig.suptitle("Confidence vs PTv3 attribution proportion", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "06_confidence_vs_ptv3.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved 06_confidence_vs_ptv3.png")


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_all()
    print(f"Loaded {len(df)} samples across {df['split'].nunique()} splits")
    print(f"Output directory: {OUT_DIR}\n")

    plot_mean_by_split(df)
    plot_species_heatmap(df)
    plot_by_dataset(df)
    plot_distributions(df)
    plot_correct_vs_wrong(df)
    plot_confidence_vs_ptv3(df)

    print("\nAll done.")
