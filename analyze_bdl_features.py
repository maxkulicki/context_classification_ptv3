"""Analyze BDL categorical features available for fusion.

Produces distribution plots and a summary report for site_type,
soil_subtype_cd, moisture_cd, and species_cd across our 271 plots.

Usage:
    python analyze_bdl_features.py [--output_dir results/bdl_features]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BDL_PATH = "/home/makskulicki/tree_species_context_classification/data/plots_bdl_fused.csv"
PC_MAPPING_PATH = "Pointcept/data/treescanpl/sample_plotid_mapping.csv"

FEATURES = [
    ("site_type", "Forest Site Type"),
    ("soil_subtype_cd", "Soil Subtype"),
    ("moisture_cd", "Moisture Regime"),
    ("species_cd", "Dominant Species"),
]


def load_data():
    bdl = pd.read_csv(BDL_PATH, sep=";", low_memory=False)
    bdl = bdl.rename(columns=lambda c: c.strip())
    pc = pd.read_csv(PC_MAPPING_PATH)
    plot_ids = set(pc["plot_id"].unique())
    bdl = bdl[bdl["num"].isin(plot_ids)]
    # One row per plot
    plot_df = bdl.groupby("num").first().reset_index()
    return plot_df


def plot_distribution(values, title, out_path):
    """Horizontal bar chart of value counts."""
    counts = values.value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(counts) + 1.5)))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(counts)))
    bars = ax.barh(range(len(counts)), counts.values, color=colors)

    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index, fontsize=9)
    ax.set_xlabel("Number of plots")
    ax.set_title(title, fontsize=13, fontweight="bold")

    for i, (val, count) in enumerate(zip(counts.values, counts.values)):
        pct = count / counts.sum() * 100
        ax.text(val + 0.5, i, f"{count} ({pct:.1f}%)", va="center", fontsize=8)

    ax.set_xlim(0, counts.max() * 1.3)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_report(out_dir, plot_df):
    lines = []
    w = lines.append

    w("# BDL Categorical Features for Fusion\n")
    w("Distribution of BDL plot-level features across our 271 point cloud plots. "
      "These features describe the forest subdivision where each plot is located.\n")

    total_onehot = 0

    for col, label in FEATURES:
        counts = plot_df[col].value_counts().sort_values(ascending=False)
        total_onehot += len(counts)

        w(f"## {label} (`{col}`)\n")
        w(f"![{label}]({col}.png)\n")

        w(f"**{len(counts)} unique values** across 271 plots.\n")
        w(f"| Value | Plots | % |")
        w(f"|-------|------:|----:|")
        for val, count in counts.items():
            pct = count / len(plot_df) * 100
            w(f"| {val} | {count} | {pct:.1f}% |")
        w("")

        # Rare values summary
        rare = counts[counts < 5]
        if len(rare) > 0:
            w(f"*{len(rare)} values appear in fewer than 5 plots "
              f"({rare.sum()} plots total, {rare.sum()/len(plot_df)*100:.1f}%).*\n")

    w("## Summary\n")
    w(f"| Feature | Unique values | One-hot dims |")
    w(f"|---------|--------------|-------------|")
    for col, label in FEATURES:
        n = plot_df[col].nunique()
        w(f"| {label} (`{col}`) | {n} | {n} |")
    w(f"| **Total** | | **{total_onehot}** |")
    w("")

    # With grouping
    w("### After grouping rare values (< 5 plots) into 'Other'\n")
    grouped_total = 0
    w(f"| Feature | Raw | Grouped (incl. Other) |")
    w(f"|---------|-----|----------------------|")
    for col, label in FEATURES:
        raw = plot_df[col].nunique()
        counts = plot_df[col].value_counts()
        kept = (counts >= 5).sum()
        has_other = (counts < 5).sum() > 0
        grouped = kept + (1 if has_other else 0)
        grouped_total += grouped
        w(f"| `{col}` | {raw} | {grouped} |")
    w(f"| **Total** | **{total_onehot}** | **{grouped_total}** |")
    w("")

    (out_dir / "bdl_features_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results/bdl_features")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    plot_df = load_data()
    print(f"  {len(plot_df)} plots")

    print("Generating plots...")
    for col, label in FEATURES:
        plot_distribution(plot_df[col], f"{label} ({col})", out_dir / f"{col}.png")
        print(f"  {col}: {plot_df[col].nunique()} unique values")

    print("Generating report...")
    generate_report(out_dir, plot_df)

    print(f"Done! All outputs in {out_dir}/")


if __name__ == "__main__":
    main()
