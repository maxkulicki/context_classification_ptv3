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
from matplotlib.colors import LinearSegmentedColormap

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

PLOT_CLASS_NAMES = [
    "Abies", "Acer", "Alnus", "Betula", "Carpinus", "Fagus",
    "Larix", "Picea", "Pinus", "Quercus", "Tilia",
]

PLOT_EXPERIMENTS = {
    "Baseline": "cls-ptv3-v1m1-0-base-finetune",
    "Projected": "cls-ptv3-v1m1-0-base-context-projected",
    "Direct": "cls-ptv3-v1m1-0-base-context-direct",
}

# Color map: white (low) → green (high)
TABLE_CMAP = LinearSegmentedColormap.from_list("wg", ["#ffffff", "#c7e9c0", "#41ab5d", "#006d2c"])


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


# ── Heatmap table helper ────────────────────────────────────────────────────

def plot_heatmap_table(data, row_labels, col_labels, title, out_path,
                       fmt=".1%", vmin=0.0, vmax=1.0, extra_col=None,
                       highlight_best=True, figwidth=None):
    """Render a numeric table as a color-coded heatmap image.

    Args:
        data: 2D array (rows x cols), values to display and color.
        row_labels: labels for each row.
        col_labels: labels for each data column.
        title: figure title.
        out_path: save path.
        fmt: format string for cell values (e.g. ".1%" for percentages, ".3f" for floats).
        vmin, vmax: color scale range.
        extra_col: optional list of strings, one per row, shown as a leading column (e.g. support).
        highlight_best: bold the best value per row.
        figwidth: optional figure width override.
    """
    n_rows, n_cols = data.shape
    n_display_cols = n_cols + (1 if extra_col else 0)

    if figwidth is None:
        figwidth = max(5, 1.8 * n_display_cols + 1.5)
    figheight = max(2.5, 0.45 * n_rows + 1.5)

    fig, ax = plt.subplots(figsize=(figwidth, figheight))

    # Draw colored cells for data columns
    col_offset = 1 if extra_col else 0
    im = ax.imshow(data, aspect="auto", cmap=TABLE_CMAP, vmin=vmin, vmax=vmax,
                   extent=[col_offset - 0.5, n_display_cols - 0.5, n_rows - 0.5, -0.5])

    # Extra column (uncolored)
    if extra_col:
        for i in range(n_rows):
            ax.add_patch(plt.Rectangle((-0.5, i - 0.5), 1, 1,
                                       facecolor="#f0f0f0", edgecolor="#cccccc", linewidth=0.5))
            ax.text(0, i, str(extra_col[i]), ha="center", va="center", fontsize=10)

    # Find best per row
    best_per_row = np.argmax(data, axis=1) if highlight_best else np.full(n_rows, -1)

    # Cell text
    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            if fmt.endswith("%"):
                text = f"{val:{fmt}}"
            else:
                text = f"{val:{fmt}}"
            weight = "bold" if j == best_per_row[i] else "normal"
            color = "white" if val > (vmax - vmin) * 0.7 + vmin else "black"
            ax.text(j + col_offset, i, text, ha="center", va="center",
                    fontsize=10, fontweight=weight, color=color)

    # Grid lines
    for i in range(n_rows + 1):
        ax.axhline(i - 0.5, color="#cccccc", linewidth=0.5)
    for j in range(n_display_cols + 1):
        ax.axvline(j - 0.5, color="#cccccc", linewidth=0.5)

    # Labels
    all_col_labels = ([""] + col_labels) if extra_col else col_labels
    ax.set_xticks(range(n_display_cols))
    ax.set_xticklabels(all_col_labels, fontsize=11, fontweight="bold")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="both", which="both", length=0)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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


def plot_class_distribution(support, out_path):
    """Bar chart of class sample counts."""
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(CLASS_NAMES)))
    ax.bar(range(len(CLASS_NAMES)), support, color=colors)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Class Distribution (K-Fold, 10 Classes)", fontsize=13, fontweight="bold")
    for i, s in enumerate(support):
        ax.text(i, s + 20, str(int(s)), ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Table image generation ───────────────────────────────────────────────────

def generate_table_images(out_dir, plot_logs, all_agg, all_cls, all_folds):
    exp_names = list(all_agg.keys())

    # 1. Plot-level overall accuracy
    data = np.array([[plot_logs[name][k] for name in exp_names]
                     for k in ["allAcc", "mAcc"]])
    plot_heatmap_table(
        data, ["Overall Acc", "Mean Acc"], exp_names,
        "Plot-Level Split: Overall Metrics (11 Classes)",
        os.path.join(out_dir, "table_plot_overall.png"),
        fmt=".1%", vmin=0.5, vmax=1.0,
    )

    # 2. Plot-level per-class recall
    data = np.array([[plot_logs[name]["per_class_recall"][i] for name in exp_names]
                     for i in range(len(PLOT_CLASS_NAMES))])
    plot_heatmap_table(
        data, PLOT_CLASS_NAMES, exp_names,
        "Plot-Level Split: Per-Class Recall (11 Classes)",
        os.path.join(out_dir, "table_plot_recall.png"),
        fmt=".2f", vmin=0.0, vmax=1.0,
    )

    # 3. K-fold aggregated metrics
    data = np.array([[float(all_agg[name].get(k, 0)) for name in exp_names]
                     for k in METRIC_KEYS])
    plot_heatmap_table(
        data, METRIC_LABELS, exp_names,
        "District K-Fold CV: Aggregated Metrics (10 Classes)",
        os.path.join(out_dir, "table_kfold_overall.png"),
        fmt=".1%", vmin=0.4, vmax=0.9,
    )

    # 4. K-fold per-class F1
    data = np.array([[float(all_cls[name].get(c, {}).get("f1", 0)) for name in exp_names]
                     for c in CLASS_NAMES])
    support = [all_cls[exp_names[0]].get(c, {}).get("support", "") for c in CLASS_NAMES]
    plot_heatmap_table(
        data, CLASS_NAMES, exp_names,
        "District K-Fold CV: Per-Class F1",
        os.path.join(out_dir, "table_kfold_f1.png"),
        fmt=".3f", vmin=0.0, vmax=1.0,
        extra_col=support,
    )

    # 5. K-fold per-district accuracy
    data = np.array([[float(all_folds[name][j]["allAcc"]) for name in exp_names]
                     for j in range(6)])
    n_samples = [all_folds[exp_names[0]][j].get("n_samples", "") for j in range(6)]
    plot_heatmap_table(
        data, DISTRICTS, exp_names,
        "District K-Fold CV: Per-District Overall Accuracy",
        os.path.join(out_dir, "table_kfold_districts.png"),
        fmt=".1%", vmin=0.2, vmax=1.0,
        extra_col=n_samples,
    )


# ── Report ───────────────────────────────────────────────────────────────────

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
    w("![Plot-Level Overall](table_plot_overall.png)\n")
    w("![Plot-Level Recall](plot_level_recall.png)\n")
    w("![Plot-Level Recall Table](table_plot_recall.png)\n")

    # ── K-fold results ─────────────────────────────────────────────────
    w("## District K-Fold CV Results (10 Classes)\n")
    w("![K-Fold Metrics](overall_metrics.png)\n")
    w("![K-Fold Metrics Table](table_kfold_overall.png)\n")
    w("![Per-Class F1](per_class_f1.png)\n")
    w("![Per-Class F1 Table](table_kfold_f1.png)\n")
    w("![Per-District Accuracy](per_fold_accuracy.png)\n")
    w("![Per-District Table](table_kfold_districts.png)\n")

    # Confusion matrices
    w("### Confusion Matrices (K-Fold Aggregated, Normalized)\n")
    w("| Baseline | Projected Fusion (best) |")
    w("|----------|------------------------|")
    w("| ![Baseline](cm_baseline.png) | ![Projected](cm_projected.png) |")
    w("")

    # Key findings
    w("## Key Findings\n")

    proj_agg = all_agg["Projected"]
    base_agg = all_agg["Baseline"]
    d_acc = float(proj_agg["allAcc"]) - float(base_agg["allAcc"])
    d_f1 = float(proj_agg["macro_f1"]) - float(base_agg["macro_f1"])

    w(f"1. **Projected fusion generalizes best.** On the district k-fold "
      f"(leave-one-district-out), projected fusion improves over baseline by "
      f"+{d_acc:.1%} overall accuracy and +{d_f1:.1%} macro F1. "
      f"This is the most reliable evaluation since it tests on entirely unseen districts.")
    w("")

    w(f"2. **Direct fusion wins on plot-level split but not on k-fold.** "
      f"Direct achieves {plot_logs['Direct']['allAcc']:.1%} overall accuracy on the "
      f"plot-level split (vs {plot_logs['Projected']['allAcc']:.1%} projected), "
      f"but drops to {float(all_agg['Direct']['allAcc']):.1%} on the k-fold "
      f"(vs {float(all_agg['Projected']['allAcc']):.1%} projected). "
      f"This suggests direct fusion overfits to the training distribution — "
      f"raw concatenation of 512d point features with 64d context lets the model "
      f"memorize plot-specific patterns rather than learning generalizable ecological context.")
    w("")

    # Top per-class improvements (projected vs baseline, kfold)
    deltas = []
    for c in CLASS_NAMES:
        pf = float(all_cls["Projected"].get(c, {}).get("f1", 0))
        bf = float(all_cls["Baseline"].get(c, {}).get("f1", 0))
        deltas.append((c, pf - bf))
    deltas.sort(key=lambda x: x[1], reverse=True)
    top3 = ", ".join(f"{c} (+{d:.3f})" for c, d in deltas[:3])
    w(f"3. **Largest per-class F1 gains on k-fold** (projected vs baseline): {top3}. "
      f"Projected fusion particularly helps species that co-occur with distinctive "
      f"satellite-visible forest types.")
    w("")

    gorlice_vals = {name: float(all_folds[name][0]["allAcc"]) for name in exp_names}
    w(f"4. **Gorlice is the hardest district** (most distinct species composition). "
      f"Projected fusion rescues it from {gorlice_vals['Baseline']:.0%} to "
      f"{gorlice_vals['Projected']:.0%} overall accuracy; direct fusion does not "
      f"({gorlice_vals['Direct']:.0%}), further confirming its poor generalization.")
    w("")

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

    # Load k-fold experiments
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

    # Load plot-level results
    plot_logs = {}
    print("\nLoading plot-level results...")
    for name, exp_dir in PLOT_EXPERIMENTS.items():
        log_path = os.path.join(EXP_BASE, exp_dir, "train.log")
        if os.path.exists(log_path):
            plot_logs[name] = parse_log_metrics(log_path, PLOT_CLASS_NAMES)
            print(f"  {name}: allAcc={plot_logs[name]['allAcc']:.4f}  mAcc={plot_logs[name]['mAcc']:.4f}")

    baseline_metrics = metrics_from_cm(all_cms["Baseline"])

    print("\nGenerating plots...")
    plot_overall_metrics(all_agg, out)
    plot_per_class_f1(all_cls, out)
    plot_per_fold_accuracy(all_folds, out)
    plot_class_distribution(baseline_metrics["support"], os.path.join(out, "class_distribution.png"))

    # Plot-level recall bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    exp_names = list(plot_logs.keys())
    x = np.arange(len(PLOT_CLASS_NAMES))
    w = 0.25
    for i, name in enumerate(exp_names):
        offset = (i - (len(exp_names) - 1) / 2) * w
        ax.bar(x + offset, plot_logs[name]["per_class_recall"], w,
               label=name, color=EXP_COLORS[name])
    ax.set_xticks(x)
    ax.set_xticklabels(PLOT_CLASS_NAMES, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel("Recall (Per-Class Accuracy)")
    ax.set_ylim(0, 1.08)
    ax.set_title("Per-Class Recall: Plot-Level Split (11 Classes)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "plot_level_recall.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Confusion matrices
    plot_confusion_matrix(all_cms["Baseline"], CLASS_NAMES,
                          "Baseline — K-Fold Aggregated", os.path.join(out, "cm_baseline.png"))
    plot_confusion_matrix(all_cms["Projected"], CLASS_NAMES,
                          "Projected Fusion — K-Fold Aggregated", os.path.join(out, "cm_projected.png"))

    # Heatmap tables
    print("Generating table images...")
    generate_table_images(out, plot_logs, all_agg, all_cls, all_folds)

    print("\nWriting report...")
    generate_report(out, all_agg, all_cls, all_folds, all_cms, plot_logs)

    print(f"\nDone! All outputs in {out}/")


if __name__ == "__main__":
    main()
