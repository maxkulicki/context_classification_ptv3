"""Generate comparison report: PTv3 baseline vs projected vs direct AE fusion.

Reads k-fold summary CSVs, confusion matrices, and plot-level train logs.
Generates plots and a self-contained markdown report in results/report/.

Usage:
    python generate_report.py [--output_dir results/report]
"""

import argparse
import csv
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Constants ────────────────────────────────────────────────────────────────

DISTRICTS = ["Gorlice", "Herby", "Katrynka", "Milicz", "Piensk", "Suprasl"]
CLASS_NAMES = [
    "Acer", "Alnus", "Betula", "Carpinus", "Fagus",
    "Larix", "Picea", "Pinus", "Quercus", "Tilia",
]

EXPERIMENTS = {
    "Baseline": {
        "csv": "results/baseline-kfold/kfold_summary.csv",
        "cm_prefix": "cls-ptv3-v1m1-0-base-finetune-kfold",
    },
    "Projected": {
        "csv": "results/context-projected-kfold/kfold_summary.csv",
        "cm_prefix": "cls-ptv3-v1m1-0-base-context-projected-kfold",
    },
    "Direct": {
        "csv": "results/context-direct-kfold/kfold_summary.csv",
        "cm_prefix": "cls-ptv3-v1m1-0-base-context-direct-kfold",
    },
}

EXP_COLORS = {
    "Baseline": "#7f7f7f",
    "Projected": "#2ca02c",
    "Direct": "#1f77b4",
}

EXP_BASE = "Pointcept/exp/treescanpl"

METRIC_KEYS = ["allAcc", "mAcc", "macro_f1", "weighted_f1"]
METRIC_LABELS = ["Overall Acc", "Mean Acc", "Macro F1", "Weighted F1"]

# Plot-level split experiments (11 classes incl. Abies)
PLOT_CLASS_NAMES = [
    "Abies", "Acer", "Alnus", "Betula", "Carpinus", "Fagus",
    "Larix", "Picea", "Pinus", "Quercus", "Tilia",
]

PLOT_EXPERIMENTS = {
    "Baseline": "cls-ptv3-v1m1-0-base-finetune",
    "Projected": "cls-ptv3-v1m1-0-base-context-projected",
    "Direct": "cls-ptv3-v1m1-0-base-context-direct",
}


# ── Data loading ─────────────────────────────────────────────────────────────

def parse_log_metrics(log_path, class_names):
    """Parse best mAcc/allAcc and per-class recall from train.log."""
    best_macc, best_allacc = 0.0, 0.0
    per_class_recall = {}

    with open(log_path) as f:
        for line in f:
            m = re.search(r"Current best record.*mAcc:\s*([\d.]+)\s+allAcc:\s*([\d.]+)", line)
            if m:
                macc, allacc = float(m.group(1)), float(m.group(2))
                if allacc > best_allacc:
                    best_macc, best_allacc = macc, allacc

            m = re.search(r"Class_\d+\s+-\s+(\w+)\s+Result:\s+iou/accuracy\s+([\d.]+)", line)
            if m:
                per_class_recall[m.group(1)] = float(m.group(2))

    recall_arr = np.array([per_class_recall.get(c, 0.0) for c in class_names])
    return dict(mAcc=best_macc, allAcc=best_allacc, per_class_recall=recall_arr)

def load_kfold_csv(path):
    """Parse kfold_summary.csv → fold rows, aggregated row, per-class rows."""
    folds = []
    aggregated = {}
    per_class = {}

    with open(path) as f:
        rows = list(csv.reader(f))

    header = rows[0]
    for row in rows[1:7]:
        folds.append({header[i]: row[i] for i in range(len(header))})

    for row in rows:
        if row and row[0] == "aggregated":
            aggregated = {header[i]: row[i] for i in range(len(header))}
            break

    for i, row in enumerate(rows):
        if row and row[0] == "class":
            cls_header = row
            for crow in rows[i + 1:]:
                if not crow or not crow[0]:
                    break
                per_class[crow[0]] = {cls_header[j]: crow[j] for j in range(len(cls_header))}
            break

    return folds, aggregated, per_class


def load_kfold_cms(prefix):
    """Load per-fold confusion matrices → aggregated CM."""
    cms = []
    for fold in range(6):
        district = DISTRICTS[fold]
        path = os.path.join(EXP_BASE, f"{prefix}-fold{fold}-{district}", "confusion_matrix.npy")
        if os.path.exists(path):
            cms.append(np.load(path))
    return sum(cms) if cms else None


def metrics_from_cm(cm):
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return dict(
        allAcc=tp.sum() / (support.sum() + 1e-10),
        mAcc=np.mean(recall),
        macro_f1=np.mean(f1),
        weighted_f1=np.average(f1, weights=support) if support.sum() > 0 else 0.0,
        precision=precision, recall=recall, f1=f1, support=support,
    )


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_overall_metrics(all_agg, out_dir):
    """3-way grouped bar chart of aggregate metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    exp_names = list(all_agg.keys())
    n_exps = len(exp_names)
    x = np.arange(len(METRIC_KEYS))
    w = 0.25

    for i, name in enumerate(exp_names):
        vals = [float(all_agg[name].get(k, 0)) for k in METRIC_KEYS]
        offset = (i - (n_exps - 1) / 2) * w
        ax.bar(x + offset, vals, w, label=name, color=EXP_COLORS[name])
        for j, v in enumerate(vals):
            ax.text(x[j] + offset, v + 0.008, f"{v:.1%}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS, fontsize=11)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, 0))
    ax.set_title("District K-Fold CV: Aggregated Metrics", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "overall_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_f1(all_cls, out_dir):
    """3-way grouped bar chart of per-class F1."""
    fig, ax = plt.subplots(figsize=(14, 6))
    exp_names = list(all_cls.keys())
    n_exps = len(exp_names)
    x = np.arange(len(CLASS_NAMES))
    w = 0.25

    for i, name in enumerate(exp_names):
        vals = [float(all_cls[name].get(c, {}).get("f1", 0)) for c in CLASS_NAMES]
        offset = (i - (n_exps - 1) / 2) * w
        ax.bar(x + offset, vals, w, label=name, color=EXP_COLORS[name])

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.08)
    ax.set_title("Per-Class F1: District K-Fold CV", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_class_f1.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_fold_accuracy(all_folds, out_dir):
    """3-way grouped bar chart of allAcc per district."""
    fig, ax = plt.subplots(figsize=(12, 6))
    exp_names = list(all_folds.keys())
    n_exps = len(exp_names)
    x = np.arange(len(DISTRICTS))
    w = 0.25

    for i, name in enumerate(exp_names):
        vals = [float(all_folds[name][j]["allAcc"]) for j in range(6)]
        offset = (i - (n_exps - 1) / 2) * w
        ax.bar(x + offset, vals, w, label=name, color=EXP_COLORS[name])

    ax.set_xticks(x)
    ax.set_xticklabels(DISTRICTS, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel("Overall Accuracy")
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, 0))
    ax.set_title("Per-District Overall Accuracy", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_fold_accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, classes, title, out_path):
    """Row-normalized confusion matrix heatmap."""
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm_norm[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val > 0.5 else "black", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_plot_level_recall(plot_logs, out_dir):
    """3-way grouped bar chart of per-class recall from plot-level split."""
    fig, ax = plt.subplots(figsize=(14, 6))
    exp_names = list(plot_logs.keys())
    n_exps = len(exp_names)
    x = np.arange(len(PLOT_CLASS_NAMES))
    w = 0.25

    for i, name in enumerate(exp_names):
        vals = plot_logs[name]["per_class_recall"]
        offset = (i - (n_exps - 1) / 2) * w
        ax.bar(x + offset, vals, w, label=name, color=EXP_COLORS[name])

    ax.set_xticks(x)
    ax.set_xticklabels(PLOT_CLASS_NAMES, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel("Recall (Per-Class Accuracy)")
    ax.set_ylim(0, 1.08)
    ax.set_title("Per-Class Recall: Plot-Level Split (11 Classes)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "plot_level_recall.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(support, out_path):
    """Bar chart of class sample counts."""
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(CLASS_NAMES)))
    bars = ax.bar(CLASS_NAMES, support, color=colors)
    ax.set_ylabel("Number of Samples")
    ax.set_title("Class Distribution (K-Fold, 10 Classes)", fontsize=13, fontweight="bold")
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    for bar, s in zip(bars, support):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                str(int(s)), ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Report ───────────────────────────────────────────────────────────────────

def bold_best(vals_str, vals_float):
    """Bold the highest value in a list of formatted strings."""
    best = int(np.argmax(vals_float))
    result = list(vals_str)
    result[best] = f"**{result[best]}**"
    return result


def generate_report(out_dir, all_agg, all_cls, all_folds, all_cms, plot_logs):
    exp_names = list(all_agg.keys())
    lines = []
    w = lines.append

    w("# PTv3 Tree Species Classification: Experiment Report\n")
    w("Comparison of three approaches for individual tree species classification "
      "from airborne LiDAR point clouds.\n")

    # Setup
    w("## Experimental Setup\n")
    w("| | Detail |")
    w("|---|---|")
    w("| **Task** | Tree genus classification from individual LiDAR point clouds |")
    w("| **Dataset** | TreeScanPL: 6,789 trees across 271 plots in 6 forest districts |")
    w("| **Backbone** | Point Transformer v3 (PTv3-v1m1), pretrained on FOR-species20K |")
    w("| **Optimizer** | AdamW, OneCycleLR |")
    w("")

    w("### Evaluation Protocols\n")
    w("| | Plot-Level Split | District K-Fold CV |")
    w("|---|---|---|")
    w("| **Split** | 80/20 stratified by plot | 6-fold leave-one-district-out |")
    w("| **Classes** | 11 genera (incl. Abies) | 10 genera (excl. Abies) |")
    w("| **Samples** | 5,411 train / 1,378 test | ~5,300 / ~1,060 per fold |")
    w("| **Epochs** | 100 | 60 per fold |")
    w("")

    w("### Methods\n")
    w("| Method | Description |")
    w("|--------|-------------|")
    w("| **Baseline** | PTv3 point cloud features (512d) → classification MLP |")
    w("| **Projected fusion** | PTv3 (512d→128d) + AlphaEarth (64d→128d) projected to shared space, concatenated (256d) → MLP |")
    w("| **Direct fusion** | PTv3 (512d) + AlphaEarth (64d) concatenated raw (576d) → MLP |")
    w("")
    w("AlphaEarth embeddings are 64-dimensional satellite-derived features "
      "representing the ecological context of each plot location.\n")

    # Class distribution
    w("### Class Distribution\n")
    w("![Class Distribution](class_distribution.png)\n")

    # ── Plot-level results ─────────────────────────────────────────────
    w("## Plot-Level Split Results (11 Classes)\n")

    w("### Overall Accuracy\n")
    header = "| Metric | " + " | ".join(exp_names) + " |"
    sep = "|--------|" + "|".join(["--------"] * len(exp_names)) + "|"
    w(header)
    w(sep)
    for key, label in [("allAcc", "Overall Acc"), ("mAcc", "Mean Acc")]:
        fvals = [plot_logs[name][key] for name in exp_names]
        svals = [f"{v:.1%}" for v in fvals]
        svals = bold_best(svals, fvals)
        w(f"| {label} | " + " | ".join(svals) + " |")
    w("")

    w("### Per-Class Recall\n")
    w("![Plot-Level Recall](plot_level_recall.png)\n")
    header = "| Genus | " + " | ".join(exp_names) + " |"
    sep = "|-------|" + "|".join(["--------"] * len(exp_names)) + "|"
    w(header)
    w(sep)
    for i, c in enumerate(PLOT_CLASS_NAMES):
        fvals = [plot_logs[name]["per_class_recall"][i] for name in exp_names]
        svals = [f"{v:.3f}" for v in fvals]
        svals = bold_best(svals, fvals)
        w(f"| {c} | " + " | ".join(svals) + " |")
    w("")

    # ── K-fold results ─────────────────────────────────────────────────
    w("## District K-Fold CV Results (10 Classes)\n")
    w("### Aggregated Metrics\n")
    w("![Overall Metrics](overall_metrics.png)\n")

    header = "| Metric | " + " | ".join(exp_names) + " |"
    sep = "|--------|" + "|".join(["--------"] * len(exp_names)) + "|"
    w(header)
    w(sep)
    for k, label in zip(METRIC_KEYS, METRIC_LABELS):
        fvals = [float(all_agg[name].get(k, 0)) for name in exp_names]
        svals = [f"{v:.1%}" for v in fvals]
        svals = bold_best(svals, fvals)
        w(f"| {label} | " + " | ".join(svals) + " |")
    w("")

    # Per-class F1
    w("### Per-Class F1\n")
    w("![Per-Class F1](per_class_f1.png)\n")
    header = "| Genus | Support | " + " | ".join(exp_names) + " |"
    sep = "|-------|---------|" + "|".join(["--------"] * len(exp_names)) + "|"
    w(header)
    w(sep)
    for c in CLASS_NAMES:
        support = all_cls[exp_names[0]].get(c, {}).get("support", "?")
        fvals = [float(all_cls[name].get(c, {}).get("f1", 0)) for name in exp_names]
        svals = [f"{v:.3f}" for v in fvals]
        svals = bold_best(svals, fvals)
        w(f"| {c} | {support} | " + " | ".join(svals) + " |")
    w("")

    # Per-district accuracy
    w("### Per-District Overall Accuracy\n")
    w("![Per-Fold Accuracy](per_fold_accuracy.png)\n")
    header = "| District | N | " + " | ".join(exp_names) + " |"
    sep = "|----------|---|" + "|".join(["--------"] * len(exp_names)) + "|"
    w(header)
    w(sep)
    for j, d in enumerate(DISTRICTS):
        n = all_folds[exp_names[0]][j].get("n_samples", "?")
        fvals = [float(all_folds[name][j]["allAcc"]) for name in exp_names]
        svals = [f"{v:.1%}" for v in fvals]
        svals = bold_best(svals, fvals)
        w(f"| {d} | {n} | " + " | ".join(svals) + " |")
    w("")

    # Confusion matrices (baseline vs projected — the winner)
    w("### Confusion Matrices (K-Fold Aggregated, Normalized)\n")
    w("| Baseline | Projected Fusion (best) |")
    w("|----------|------------------------|")
    w("| ![Baseline](cm_baseline.png) | ![Projected](cm_projected.png) |")
    w("")

    # Key findings
    w("## Key Findings\n")

    # Compute deltas (projected vs baseline)
    proj_agg = all_agg["Projected"]
    base_agg = all_agg["Baseline"]
    d_acc = float(proj_agg["allAcc"]) - float(base_agg["allAcc"])
    d_f1 = float(proj_agg["macro_f1"]) - float(base_agg["macro_f1"])

    w(f"1. **Projected fusion is the best approach**, improving over baseline by "
      f"+{d_acc:.1%} overall accuracy and +{d_f1:.1%} macro F1.")
    w("")

    w("2. **Direct fusion underperforms projected fusion.** Raw concatenation of "
      "512d point features with 64d context allows the larger point features to "
      "dominate. Projection to a shared 128d space balances the two sources.")
    w("")

    # Top per-class improvements (projected vs baseline)
    deltas = []
    for c in CLASS_NAMES:
        pf = float(all_cls["Projected"].get(c, {}).get("f1", 0))
        bf = float(all_cls["Baseline"].get(c, {}).get("f1", 0))
        deltas.append((c, pf - bf))
    deltas.sort(key=lambda x: x[1], reverse=True)
    top3 = ", ".join(f"{c} (+{d:.3f})" for c, d in deltas[:3])
    w(f"3. **Largest per-class F1 gains** (projected vs baseline): {top3}")
    w("")

    # Gorlice
    gorlice_vals = {name: float(all_folds[name][0]["allAcc"]) for name in exp_names}
    w(f"4. **Gorlice is the hardest district** (most different species composition). "
      f"Projected fusion rescues it from {gorlice_vals['Baseline']:.0%} to "
      f"{gorlice_vals['Projected']:.0%} overall accuracy; direct fusion does not "
      f"({gorlice_vals['Direct']:.0%}).")
    w("")

    # Weak classes
    proj_cls = all_cls["Projected"]
    worst = sorted(CLASS_NAMES, key=lambda c: float(proj_cls.get(c, {}).get("f1", 0)))[:3]
    worst_str = ", ".join(f"{c} (F1={float(proj_cls[c]['f1']):.3f}, n={proj_cls[c]['support']})"
                          for c in worst)
    w(f"5. **Weakest classes**: {worst_str}. "
      "These have the fewest training samples — class imbalance remains the main bottleneck.")
    w("")

    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {report_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/report")
    args = parser.parse_args()
    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    # Load all experiments
    all_folds = {}
    all_agg = {}
    all_cls = {}
    all_cms = {}

    for name, cfg in EXPERIMENTS.items():
        print(f"Loading {name}...")
        folds, agg, cls = load_kfold_csv(cfg["csv"])
        all_folds[name] = folds
        all_agg[name] = agg
        all_cls[name] = cls
        all_cms[name] = load_kfold_cms(cfg["cm_prefix"])

    # Load plot-level results from train logs
    plot_logs = {}
    print("\nLoading plot-level results...")
    for name, exp_dir in PLOT_EXPERIMENTS.items():
        log_path = os.path.join(EXP_BASE, exp_dir, "train.log")
        if os.path.exists(log_path):
            plot_logs[name] = parse_log_metrics(log_path, PLOT_CLASS_NAMES)
            print(f"  {name}: allAcc={plot_logs[name]['allAcc']:.4f}  mAcc={plot_logs[name]['mAcc']:.4f}")
        else:
            print(f"  {name}: train.log not found at {log_path}")

    # Get support from baseline CM for class distribution
    baseline_metrics = metrics_from_cm(all_cms["Baseline"])

    # Generate plots
    print("\nGenerating plots...")
    plot_overall_metrics(all_agg, out)
    plot_per_class_f1(all_cls, out)
    plot_per_fold_accuracy(all_folds, out)
    plot_class_distribution(baseline_metrics["support"], os.path.join(out, "class_distribution.png"))
    if plot_logs:
        plot_plot_level_recall(plot_logs, out)

    # Confusion matrices: baseline vs projected (the winner)
    plot_confusion_matrix(
        all_cms["Baseline"], CLASS_NAMES,
        "Baseline — K-Fold Aggregated",
        os.path.join(out, "cm_baseline.png"),
    )
    plot_confusion_matrix(
        all_cms["Projected"], CLASS_NAMES,
        "Projected Fusion — K-Fold Aggregated",
        os.path.join(out, "cm_projected.png"),
    )

    # Generate report
    print("\nWriting report...")
    generate_report(out, all_agg, all_cls, all_folds, all_cms, plot_logs)

    print(f"\nDone! All outputs in {out}/")


if __name__ == "__main__":
    main()
