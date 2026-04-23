"""Compare k-fold cross-validation results across three experiments:
baseline (point cloud only), projected fusion, and direct fusion.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXPERIMENTS = {
    "Baseline": "results/baseline-kfold/kfold_summary.csv",
    "Projected": "results/context-projected-kfold/kfold_summary.csv",
    "Direct": "results/context-direct-kfold/kfold_summary.csv",
}

EXP_COLORS = {
    "Baseline": "#7f7f7f",
    "Projected": "#2ca02c",
    "Direct": "#1f77b4",
}

CLASS_NAMES = [
    "Acer", "Alnus", "Betula", "Carpinus", "Fagus",
    "Larix", "Picea", "Pinus", "Quercus", "Tilia",
]

DISTRICTS = ["Gorlice", "Herby", "Katrynka", "Milicz", "Piensk", "Suprasl"]

METRIC_KEYS = ["allAcc", "mAcc", "macro_f1", "weighted_f1"]
METRIC_LABELS = ["Overall Acc", "Mean Acc", "Macro F1", "Weighted F1"]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_csv(path: str):
    """Parse kfold_summary.csv → fold rows, aggregated row, per-class rows."""
    folds = []
    aggregated = {}
    per_class = {}

    with open(path) as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Fold rows (lines 1-6, after header)
    header = rows[0]
    for row in rows[1:7]:
        folds.append({header[i]: row[i] for i in range(len(header))})

    # Find aggregated row
    for row in rows:
        if row and row[0] == "aggregated":
            aggregated = {header[i]: row[i] for i in range(len(header))}
            break

    # Find per-class section
    for i, row in enumerate(rows):
        if row and row[0] == "class":
            cls_header = row
            for crow in rows[i + 1:]:
                if not crow or not crow[0]:
                    break
                per_class[crow[0]] = {cls_header[j]: crow[j] for j in range(len(cls_header))}
            break

    return folds, aggregated, per_class


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_overall_metrics(all_agg: dict, out_dir: Path):
    """Grouped bar chart of aggregate metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    exp_names = list(all_agg.keys())
    n_metrics = len(METRIC_KEYS)
    n_exps = len(exp_names)
    x = np.arange(n_metrics)
    w = 0.25

    for i, name in enumerate(exp_names):
        vals = [float(all_agg[name].get(k, 0)) for k in METRIC_KEYS]
        offset = (i - (n_exps - 1) / 2) * w
        bars = ax.bar(x + offset, vals, w, label=name, color=EXP_COLORS[name])
        for j, v in enumerate(vals):
            ax.text(x[j] + offset, v + 0.008, f"{v:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("K-Fold Aggregated Metrics: Baseline vs Projected vs Direct Fusion")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "overall_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_f1(all_cls: dict, out_dir: Path):
    """Grouped bar chart of per-class F1."""
    fig, ax = plt.subplots(figsize=(14, 6))
    exp_names = list(all_cls.keys())
    n_classes = len(CLASS_NAMES)
    n_exps = len(exp_names)
    x = np.arange(n_classes)
    w = 0.25

    for i, name in enumerate(exp_names):
        vals = [float(all_cls[name].get(c, {}).get("f1", 0)) for c in CLASS_NAMES]
        offset = (i - (n_exps - 1) / 2) * w
        ax.bar(x + offset, vals, w, label=name, color=EXP_COLORS[name])

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Class F1: Baseline vs Projected vs Direct Fusion")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "per_class_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_fold_accuracy(all_folds: dict, out_dir: Path):
    """Grouped bar chart of allAcc per district."""
    fig, ax = plt.subplots(figsize=(12, 6))
    exp_names = list(all_folds.keys())
    n_districts = len(DISTRICTS)
    n_exps = len(exp_names)
    x = np.arange(n_districts)
    w = 0.25

    for i, name in enumerate(exp_names):
        vals = [float(all_folds[name][j]["allAcc"]) for j in range(n_districts)]
        offset = (i - (n_exps - 1) / 2) * w
        ax.bar(x + offset, vals, w, label=name, color=EXP_COLORS[name])

    ax.set_xticks(x)
    ax.set_xticklabels(DISTRICTS, rotation=45, ha="right")
    ax.set_ylabel("Overall Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Fold Overall Accuracy by District")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "per_fold_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Report ───────────────────────────────────────────────────────────────────

def generate_report(out_dir: Path, all_agg, all_cls, all_folds):
    lines = []
    exp_names = list(all_agg.keys())

    lines.append("# K-Fold Comparison: Baseline vs Projected vs Direct Fusion\n")
    lines.append("District-level 6-fold cross-validation (10 genera, Abies excluded). "
                 "6373 samples total across 271 plots.\n")
    lines.append("- **Baseline**: PTv3 point cloud features only")
    lines.append("- **Projected**: PTv3 (512d) + AlphaEarth (64d) projected to shared 128d, then concatenated (256d)")
    lines.append("- **Direct**: PTv3 (512d) + AlphaEarth (64d) concatenated raw (576d)")
    lines.append("")

    # Overall metrics
    lines.append("## 1. Aggregated Metrics\n")
    lines.append("![Overall Metrics](overall_metrics.png)\n")
    header = "| Metric | " + " | ".join(exp_names) + " |"
    sep = "|--------|" + "|".join(["-------"] * len(exp_names)) + "|"
    lines.append(header)
    lines.append(sep)
    for k, label in zip(METRIC_KEYS, METRIC_LABELS):
        vals = [f"{float(all_agg[name].get(k, 0)):.3f}" for name in exp_names]
        # Bold the best
        fvals = [float(all_agg[name].get(k, 0)) for name in exp_names]
        best_idx = int(np.argmax(fvals))
        vals[best_idx] = f"**{vals[best_idx]}**"
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    lines.append("")

    # Per-class F1
    lines.append("## 2. Per-Class F1\n")
    lines.append("![Per-Class F1](per_class_f1.png)\n")
    header = "| Genus | " + " | ".join(exp_names) + " |"
    sep = "|-------|" + "|".join(["-------"] * len(exp_names)) + "|"
    lines.append(header)
    lines.append(sep)
    for c in CLASS_NAMES:
        vals = [f"{float(all_cls[name].get(c, {}).get('f1', 0)):.3f}" for name in exp_names]
        fvals = [float(all_cls[name].get(c, {}).get("f1", 0)) for name in exp_names]
        best_idx = int(np.argmax(fvals))
        vals[best_idx] = f"**{vals[best_idx]}**"
        lines.append(f"| {c} | " + " | ".join(vals) + " |")
    lines.append("")

    # Per-fold accuracy
    lines.append("## 3. Per-Fold Overall Accuracy\n")
    lines.append("![Per-Fold Accuracy](per_fold_accuracy.png)\n")
    header = "| District | " + " | ".join(exp_names) + " |"
    sep = "|----------|" + "|".join(["-------"] * len(exp_names)) + "|"
    lines.append(header)
    lines.append(sep)
    for j, d in enumerate(DISTRICTS):
        vals = [f"{float(all_folds[name][j]['allAcc']):.3f}" for name in exp_names]
        fvals = [float(all_folds[name][j]["allAcc"]) for name in exp_names]
        best_idx = int(np.argmax(fvals))
        vals[best_idx] = f"**{vals[best_idx]}**"
        lines.append(f"| {d} | " + " | ".join(vals) + " |")
    lines.append("")

    (out_dir / "kfold_comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare k-fold experiments")
    parser.add_argument("--output_dir", type=str, default="results/kfold-comparison")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_folds = {}
    all_agg = {}
    all_cls = {}

    for name, path in EXPERIMENTS.items():
        print(f"Loading {name}: {path}")
        folds, agg, cls = load_csv(path)
        all_folds[name] = folds
        all_agg[name] = agg
        all_cls[name] = cls

    print("\nGenerating plots...")
    plot_overall_metrics(all_agg, out_dir)
    plot_per_class_f1(all_cls, out_dir)
    plot_per_fold_accuracy(all_folds, out_dir)

    print("Generating report...")
    generate_report(out_dir, all_agg, all_cls, all_folds)

    print(f"Done! All outputs in {out_dir}/")


if __name__ == "__main__":
    main()
