"""
GradCAM Feature Stream Attribution — Analysis & Visualization

Loads per-sample attributions CSV from gradcam_attribution.py,
generates 5 plots + markdown report in output directory.

Runs on host (no docker needed).
"""

import argparse
import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Consistent style
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE_CORRECT = {1: "#2ecc71", 0: "#e74c3c"}
LABEL_CORRECT = {1: "Correct", 0: "Incorrect"}

CLASS_NAMES = [
    "Acer", "Alnus", "Betula", "Carpinus", "Fagus",
    "Larix", "Picea", "Pinus", "Quercus", "Tilia",
]


def plot_ae_by_genus(df, output_dir):
    """1. Violin/box plot of relative_ae per genus, sorted by median."""
    order = df.groupby("genus")["relative_ae"].median().sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=df, x="genus", y="relative_ae", order=order,
        color="lightblue", fliersize=2, ax=ax,
    )
    sns.stripplot(
        data=df, x="genus", y="relative_ae", order=order,
        color="steelblue", alpha=0.15, size=2, jitter=True, ax=ax,
    )
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Equal contribution")
    ax.set_xlabel("Genus")
    ax.set_ylabel("Relative AE Contribution")
    ax.set_title("AlphaEarth Context Contribution by Genus (GradCAM)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "ae_contribution_by_genus.png"), dpi=150)
    plt.close(fig)


def plot_correct_vs_incorrect(df, output_dir):
    """2. Split violin per genus, colored by correct/incorrect."""
    order = df.groupby("genus")["relative_ae"].median().sort_values(ascending=False).index
    df_plot = df.copy()
    df_plot["Prediction"] = df_plot["correct"].map(LABEL_CORRECT)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(
        data=df_plot, x="genus", y="relative_ae", hue="Prediction",
        order=order, split=True, inner="quart",
        palette={"Correct": "#2ecc71", "Incorrect": "#e74c3c"},
        ax=ax,
    )
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Genus")
    ax.set_ylabel("Relative AE Contribution")
    ax.set_title("AE Contribution: Correct vs Incorrect Predictions")
    ax.legend(title="Prediction")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "correct_vs_incorrect.png"), dpi=150)
    plt.close(fig)


def plot_ae_by_district(df, output_dir):
    """3. Box plot per district with median annotations."""
    order = df.groupby("district")["relative_ae"].median().sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = sns.boxplot(
        data=df, x="district", y="relative_ae", order=order,
        color="lightsalmon", fliersize=2, ax=ax,
    )
    # Annotate medians
    medians = df.groupby("district")["relative_ae"].median()
    for i, district in enumerate(order):
        ax.text(i, medians[district] + 0.01, f"{medians[district]:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("District")
    ax.set_ylabel("Relative AE Contribution")
    ax.set_title("AlphaEarth Context Contribution by District")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "ae_contribution_by_district.png"), dpi=150)
    plt.close(fig)


def plot_heatmap(df, output_dir):
    """4. Genus x district heatmap of mean relative_ae."""
    pivot = df.pivot_table(
        values="relative_ae", index="genus", columns="district", aggfunc="mean"
    )
    # Sort genus by overall mean
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="YlOrRd",
        vmin=0, vmax=max(0.5, pivot.max().max() + 0.05),
        linewidths=0.5, ax=ax,
    )
    ax.set_title("Mean Relative AE Contribution (Genus × District)")
    ax.set_ylabel("Genus")
    ax.set_xlabel("District")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "heatmap_genus_district.png"), dpi=150)
    plt.close(fig)


def plot_overall_distribution(df, output_dir):
    """5. Histogram of relative_ae, split by correct/incorrect."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for corr_val, label in LABEL_CORRECT.items():
        subset = df[df["correct"] == corr_val]["relative_ae"]
        ax.hist(
            subset, bins=50, alpha=0.6, label=f"{label} (n={len(subset)})",
            color=PALETTE_CORRECT[corr_val], edgecolor="white",
        )
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.7, label="Equal contribution")
    ax.set_xlabel("Relative AE Contribution")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of AE Attribution Across All Samples")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "overall_distribution.png"), dpi=150)
    plt.close(fig)


def generate_report(df, output_dir):
    """Generate markdown report with summary tables and figure references."""
    n_total = len(df)
    n_correct = df["correct"].sum()
    acc = n_correct / n_total * 100
    mean_ae = df["relative_ae"].mean()
    median_ae = df["relative_ae"].median()

    # Per-genus table
    genus_stats = df.groupby("genus").agg(
        n=("relative_ae", "count"),
        accuracy=("correct", "mean"),
        mean_ae=("relative_ae", "mean"),
        median_ae=("relative_ae", "median"),
        std_ae=("relative_ae", "std"),
    ).sort_values("median_ae", ascending=False)
    genus_stats["accuracy"] = (genus_stats["accuracy"] * 100).round(1)
    genus_stats = genus_stats.round(4)

    # Per-district table
    district_stats = df.groupby("district").agg(
        n=("relative_ae", "count"),
        accuracy=("correct", "mean"),
        mean_ae=("relative_ae", "mean"),
        median_ae=("relative_ae", "median"),
    ).sort_values("median_ae", ascending=False)
    district_stats["accuracy"] = (district_stats["accuracy"] * 100).round(1)
    district_stats = district_stats.round(4)

    # Correct vs incorrect
    corr_stats = df.groupby("correct").agg(
        n=("relative_ae", "count"),
        mean_ae=("relative_ae", "mean"),
        median_ae=("relative_ae", "median"),
    ).round(4)
    corr_stats.index = corr_stats.index.map({0: "Incorrect", 1: "Correct"})

    # Key findings
    top_ae_genus = genus_stats.index[0]
    top_ae_val = genus_stats.loc[top_ae_genus, "median_ae"]
    low_ae_genus = genus_stats.index[-1]
    low_ae_val = genus_stats.loc[low_ae_genus, "median_ae"]

    ae_dominated = (df["relative_ae"] > 0.5).sum()
    ptv3_dominated = (df["relative_ae"] <= 0.5).sum()

    incorrect = df[df["correct"] == 0]
    correct = df[df["correct"] == 1]

    report = textwrap.dedent(f"""\
    # GradCAM Feature Stream Attribution Report

    ## Overview

    Analysis of AlphaEarth (AE) context contribution vs PTv3 point cloud features
    in the projected fusion model, using GradCAM at the fusion point.

    - **Total samples**: {n_total} (6-fold cross-validation)
    - **Overall accuracy**: {acc:.1f}%
    - **Mean relative AE**: {mean_ae:.4f}
    - **Median relative AE**: {median_ae:.4f}
    - **PTv3-dominated samples** (rel_ae ≤ 0.5): {ptv3_dominated} ({ptv3_dominated/n_total*100:.1f}%)
    - **AE-dominated samples** (rel_ae > 0.5): {ae_dominated} ({ae_dominated/n_total*100:.1f}%)

    ## Attribution by Genus

    | Genus | N | Accuracy (%) | Mean AE | Median AE | Std AE |
    |-------|---|-------------|---------|-----------|--------|
    """)

    for genus, row in genus_stats.iterrows():
        report += (
            f"| {genus} | {int(row['n'])} | {row['accuracy']} | "
            f"{row['mean_ae']:.4f} | {row['median_ae']:.4f} | {row['std_ae']:.4f} |\n"
        )

    report += textwrap.dedent(f"""
    ![AE Contribution by Genus](ae_contribution_by_genus.png)

    ## Correct vs Incorrect Predictions

    | Prediction | N | Mean AE | Median AE |
    |-----------|---|---------|-----------|
    """)

    for label, row in corr_stats.iterrows():
        report += f"| {label} | {int(row['n'])} | {row['mean_ae']:.4f} | {row['median_ae']:.4f} |\n"

    report += textwrap.dedent(f"""
    ![Correct vs Incorrect](correct_vs_incorrect.png)

    ## Attribution by District

    | District | N | Accuracy (%) | Mean AE | Median AE |
    |----------|---|-------------|---------|-----------|
    """)

    for district, row in district_stats.iterrows():
        report += (
            f"| {district} | {int(row['n'])} | {row['accuracy']} | "
            f"{row['mean_ae']:.4f} | {row['median_ae']:.4f} |\n"
        )

    report += textwrap.dedent(f"""
    ![AE Contribution by District](ae_contribution_by_district.png)

    ## Genus × District Heatmap

    ![Heatmap](heatmap_genus_district.png)

    ## Overall Distribution

    ![Distribution](overall_distribution.png)

    ## Key Findings

    1. **PTv3 dominates**: {ptv3_dominated/n_total*100:.1f}% of samples have relative AE ≤ 0.5, confirming point geometry is the primary signal.
    2. **Highest AE reliance**: *{top_ae_genus}* (median rel_ae = {top_ae_val:.4f}) relies most on satellite context.
    3. **Lowest AE reliance**: *{low_ae_genus}* (median rel_ae = {low_ae_val:.4f}) relies least on satellite context.
    4. **Correct vs incorrect**: Correctly classified samples have mean AE = {correct['relative_ae'].mean():.4f} vs {incorrect['relative_ae'].mean():.4f} for incorrect predictions.
    """)

    report_path = os.path.join(output_dir, "gradcam_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report written to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze GradCAM attributions")
    parser.add_argument(
        "--input",
        default="Pointcept/gradcam_attributions.csv",
        help="Input CSV from gradcam_attribution.py",
    )
    parser.add_argument(
        "--output_dir",
        default="results/gradcam",
        help="Output directory for plots and report",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Folds: {sorted(df['fold'].unique())}")
    print(f"Districts: {sorted(df['district'].unique())}")
    print(f"Genera: {sorted(df['genus'].unique())}")
    print(f"Accuracy: {df['correct'].mean()*100:.1f}%")
    print(f"Mean relative AE: {df['relative_ae'].mean():.4f}")

    print("\nGenerating plots...")
    plot_ae_by_genus(df, args.output_dir)
    print("  1/5 ae_contribution_by_genus.png")

    plot_correct_vs_incorrect(df, args.output_dir)
    print("  2/5 correct_vs_incorrect.png")

    plot_ae_by_district(df, args.output_dir)
    print("  3/5 ae_contribution_by_district.png")

    plot_heatmap(df, args.output_dir)
    print("  4/5 heatmap_genus_district.png")

    plot_overall_distribution(df, args.output_dir)
    print("  5/5 overall_distribution.png")

    print("\nGenerating report...")
    generate_report(df, args.output_dir)

    print(f"\nDone! All outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
