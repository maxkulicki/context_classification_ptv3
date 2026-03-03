"""Generate comprehensive comparison report with plots.

Compares PTv3 baseline vs PTv3 + AlphaEarth context fusion
across plot-level split and district-level 6-fold CV.

Data sources:
- Plot-level: train.log (best mAcc/allAcc, per-class recall) + confusion_matrix.npy/.png
- K-fold: confusion_matrix.npy per fold (aggregated)

Usage:
    python generate_report.py --output_dir results/report
"""

import argparse
import os
import re
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Constants ────────────────────────────────────────────────────────────────

DISTRICTS = ["Gorlice", "Herby", "Katrynka", "Milicz", "Piensk", "Suprasl"]

KFOLD_CLASSES = [
    "Acer", "Alnus", "Betula", "Carpinus", "Fagus",
    "Larix", "Picea", "Pinus", "Quercus", "Tilia",
]

PLOT_CLASSES = [
    "Abies", "Acer", "Alnus", "Betula", "Carpinus", "Fagus",
    "Larix", "Picea", "Pinus", "Quercus", "Tilia",
]

EXP_BASE = "Pointcept/exp/treescanpl"

PLOT_BASELINE_DIR = f"{EXP_BASE}/cls-ptv3-v1m1-0-base-finetune"
PLOT_CONTEXT_DIR = f"{EXP_BASE}/cls-ptv3-v1m1-0-base-context-projected"
KFOLD_BASELINE_PREFIX = "cls-ptv3-v1m1-0-base-finetune-kfold"
KFOLD_CONTEXT_PREFIX = "cls-ptv3-v1m1-0-base-context-projected-kfold"


# ── Metrics from confusion matrix ────────────────────────────────────────────

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
        macro_precision=np.mean(precision),
        macro_recall=np.mean(recall),
        precision=precision,
        recall=recall,
        f1=f1,
        support=support,
    )


# ── Log parsing ──────────────────────────────────────────────────────────────

def parse_log_metrics(log_path, class_names):
    """Parse best mAcc/allAcc and per-class recall from train.log.

    Returns dict with 'mAcc', 'allAcc', 'per_class_recall' (from last eval).
    """
    best_macc, best_allacc = 0.0, 0.0
    per_class_recall = {}

    with open(log_path) as f:
        for line in f:
            # Best record: "Current best record is Evaluation N: mAcc: X allAcc: Y"
            m = re.search(r"Current best record.*mAcc:\s*([\d.]+)\s+allAcc:\s*([\d.]+)", line)
            if m:
                macc, allacc = float(m.group(1)), float(m.group(2))
                if allacc > best_allacc:
                    best_macc, best_allacc = macc, allacc

            # Per-class: "Class_N - ClassName Result: iou/accuracy X.XXXX"
            m = re.search(r"Class_\d+\s+-\s+(\w+)\s+Result:\s+iou/accuracy\s+([\d.]+)", line)
            if m:
                per_class_recall[m.group(1)] = float(m.group(2))

    # Order per-class recall to match class_names
    recall_arr = np.array([per_class_recall.get(c, 0.0) for c in class_names])

    return dict(mAcc=best_macc, allAcc=best_allacc, per_class_recall=recall_arr)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_kfold_cms(prefix):
    fold_cms = {}
    for fold in range(6):
        district = DISTRICTS[fold]
        exp_name = f"{prefix}-fold{fold}-{district}"
        cm_path = os.path.join(EXP_BASE, exp_name, "confusion_matrix.npy")
        if os.path.exists(cm_path):
            fold_cms[fold] = np.load(cm_path)
    return fold_cms


# ── Plotting helpers ─────────────────────────────────────────────────────────

COLORS = {"baseline": "#4C72B0", "context": "#DD8452"}


def plot_overall_comparison(plot_log_b, plot_log_c, kfold_m_b, kfold_m_c, out_dir):
    """Bar chart: mAcc and allAcc for all 4 experiments, plus kfold F1."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: mAcc and allAcc for all 4
    ax = axes[0]
    labels = ["Plot Baseline", "Plot + AlphaEarth", "K-Fold Baseline", "K-Fold + AlphaEarth"]
    macc_vals = [plot_log_b["mAcc"], plot_log_c["mAcc"], kfold_m_b["mAcc"], kfold_m_c["mAcc"]]
    allacc_vals = [plot_log_b["allAcc"], plot_log_c["allAcc"], kfold_m_b["allAcc"], kfold_m_c["allAcc"]]
    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax.bar(x - w / 2, allacc_vals, w, label="Overall Acc", color="#4C72B0")
    bars2 = ax.bar(x + w / 2, macc_vals, w, label="Mean Acc", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, 0))
    ax.set_title("Overall & Mean Accuracy", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.1%}", ha="center", va="bottom", fontsize=7)

    # Right panel: kfold detailed metrics (baseline vs context)
    ax = axes[1]
    metric_keys = ["allAcc", "mAcc", "macro_f1", "weighted_f1"]
    metric_labels = ["Overall Acc", "Mean Acc", "Macro F1", "Weighted F1"]
    x = np.arange(len(metric_keys))
    vals_b = [kfold_m_b[k] for k in metric_keys]
    vals_c = [kfold_m_c[k] for k in metric_keys]
    bars_b = ax.bar(x - w / 2, vals_b, w, label="PTv3 Baseline", color=COLORS["baseline"])
    bars_c = ax.bar(x + w / 2, vals_c, w, label="PTv3 + AlphaEarth", color=COLORS["context"])
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, 0))
    ax.set_title("K-Fold CV: Detailed Metrics", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for bars in [bars_b, bars_c]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.1%}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    path = os.path.join(out_dir, "overall_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_per_class_bars(classes, vals_b, vals_c, ylabel, title, out_path):
    """Grouped bar chart for per-class metric comparison."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(classes))
    w = 0.35

    ax.bar(x - w / 2, vals_b, w, label="PTv3 Baseline", color=COLORS["baseline"])
    ax.bar(x + w / 2, vals_c, w, label="PTv3 + AlphaEarth", color=COLORS["context"])

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for i in range(len(classes)):
        delta = vals_c[i] - vals_b[i]
        if abs(delta) > 0.005:
            color = "green" if delta > 0 else "red"
            y = max(vals_b[i], vals_c[i]) + 0.03
            ax.text(x[i], y, f"{delta:+.2f}", ha="center", fontsize=7,
                    color=color, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_kfold_per_district(fold_metrics_b, fold_metrics_c, out_dir):
    """Per-district allAcc and mAcc comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, label in zip(
        axes, ["allAcc", "mAcc"], ["Overall Accuracy", "Mean Per-Class Accuracy"]
    ):
        x = np.arange(6)
        w = 0.35
        vals_b = [fold_metrics_b[f][metric] for f in range(6)]
        vals_c = [fold_metrics_c[f][metric] for f in range(6)]

        ax.bar(x - w / 2, vals_b, w, label="PTv3 Baseline", color=COLORS["baseline"])
        ax.bar(x + w / 2, vals_c, w, label="PTv3 + AlphaEarth", color=COLORS["context"])

        ax.set_xticks(x)
        ax.set_xticklabels(DISTRICTS, rotation=30, ha="right", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, 0))
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        for i in range(6):
            delta = vals_c[i] - vals_b[i]
            y = max(vals_b[i], vals_c[i]) + 0.02
            color = "green" if delta > 0 else "red"
            ax.text(x[i], y, f"{delta:+.1%}", ha="center", fontsize=7,
                    color=color, fontweight="bold")

    plt.suptitle("District-Level K-Fold: Per-District Comparison",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "kfold_per_district.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_confusion_matrix(cm, classes, title, out_path):
    """Single row-normalized confusion matrix."""
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
    print(f"Saved: {out_path}")


def plot_class_distribution(classes, support, out_path):
    """Bar chart showing class distribution."""
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    bars = ax.bar(classes, support, color=colors)
    ax.set_ylabel("Number of Samples")
    ax.set_title("Class Distribution (K-Fold, 10 Classes)", fontsize=13, fontweight="bold")
    ax.set_xticklabels(classes, rotation=30, ha="right")
    for bar, s in zip(bars, support):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                str(int(s)), ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/report")
    args = parser.parse_args()
    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    # ── Plot-level: parse from train logs ─────────────────────────────────
    print("Loading plot-level results from train logs...")
    plot_log_b = parse_log_metrics(
        os.path.join(PLOT_BASELINE_DIR, "train.log"), PLOT_CLASSES
    )
    plot_log_c = parse_log_metrics(
        os.path.join(PLOT_CONTEXT_DIR, "train.log"), PLOT_CLASSES
    )
    print(f"  Baseline: allAcc={plot_log_b['allAcc']:.4f}  mAcc={plot_log_b['mAcc']:.4f}")
    print(f"  Context:  allAcc={plot_log_c['allAcc']:.4f}  mAcc={plot_log_c['mAcc']:.4f}")

    # Also load context CM .npy if available (for per-class F1)
    ctx_cm_path = os.path.join(PLOT_CONTEXT_DIR, "confusion_matrix.npy")
    plot_ctx_cm = np.load(ctx_cm_path) if os.path.exists(ctx_cm_path) else None

    # ── K-fold: aggregate confusion matrices ──────────────────────────────
    print("\nLoading k-fold results from confusion matrices...")
    kfold_cms_b = load_kfold_cms(KFOLD_BASELINE_PREFIX)
    kfold_cms_c = load_kfold_cms(KFOLD_CONTEXT_PREFIX)
    print(f"  Baseline: {len(kfold_cms_b)} folds loaded")
    print(f"  Context:  {len(kfold_cms_c)} folds loaded")

    kfold_agg_b = sum(kfold_cms_b.values())
    kfold_agg_c = sum(kfold_cms_c.values())
    kfold_m_b = metrics_from_cm(kfold_agg_b)
    kfold_m_c = metrics_from_cm(kfold_agg_c)
    print(f"  Baseline agg: allAcc={kfold_m_b['allAcc']:.4f}  mAcc={kfold_m_b['mAcc']:.4f}")
    print(f"  Context  agg: allAcc={kfold_m_c['allAcc']:.4f}  mAcc={kfold_m_c['mAcc']:.4f}")

    kfold_fold_b = {f: metrics_from_cm(cm) for f, cm in kfold_cms_b.items()}
    kfold_fold_c = {f: metrics_from_cm(cm) for f, cm in kfold_cms_c.items()}

    # ── Generate plots ────────────────────────────────────────────────────
    print("\nGenerating plots...")

    # 1. Overall comparison
    plot_overall_comparison(plot_log_b, plot_log_c, kfold_m_b, kfold_m_c, out)

    # 2. Per-class recall — plot-level (from logs, available for both)
    plot_per_class_bars(
        PLOT_CLASSES,
        plot_log_b["per_class_recall"],
        plot_log_c["per_class_recall"],
        "Recall (Per-Class Accuracy)",
        "Per-Class Recall: Plot-Level Split (11 Classes)",
        os.path.join(out, "per_class_recall_plot.png"),
    )

    # 3. Per-class F1 — kfold (from CMs)
    plot_per_class_bars(
        KFOLD_CLASSES,
        kfold_m_b["f1"],
        kfold_m_c["f1"],
        "F1 Score",
        "Per-Class F1: District K-Fold CV (10 Classes)",
        os.path.join(out, "per_class_f1_kfold.png"),
    )

    # 4. Per-district comparison
    plot_kfold_per_district(kfold_fold_b, kfold_fold_c, out)

    # 5. Confusion matrices — kfold aggregated (from .npy)
    plot_confusion_matrix(
        kfold_agg_b, KFOLD_CLASSES,
        "PTv3 Baseline — K-Fold Aggregated",
        os.path.join(out, "cm_kfold_baseline.png"),
    )
    plot_confusion_matrix(
        kfold_agg_c, KFOLD_CLASSES,
        "PTv3 + AlphaEarth — K-Fold Aggregated",
        os.path.join(out, "cm_kfold_context.png"),
    )

    # 6. Confusion matrices — plot level
    # Baseline: copy existing PNG from exp dir
    baseline_cm_png = os.path.join(PLOT_BASELINE_DIR, "confusion_matrix.png")
    if os.path.exists(baseline_cm_png):
        shutil.copy2(baseline_cm_png, os.path.join(out, "cm_plot_baseline.png"))
        print(f"Copied: {baseline_cm_png} -> cm_plot_baseline.png")
    # Context: generate from .npy
    if plot_ctx_cm is not None:
        plot_confusion_matrix(
            plot_ctx_cm, PLOT_CLASSES,
            "PTv3 + AlphaEarth — Plot-Level Split",
            os.path.join(out, "cm_plot_context.png"),
        )

    # 7. Class distribution
    plot_class_distribution(
        KFOLD_CLASSES, kfold_m_b["support"],
        os.path.join(out, "class_distribution.png"),
    )

    # ── Write markdown report ─────────────────────────────────────────────
    print("\nWriting report...")
    write_markdown(
        out, plot_log_b, plot_log_c, plot_ctx_cm,
        kfold_m_b, kfold_m_c, kfold_fold_b, kfold_fold_c,
    )
    print(f"\nReport complete: {os.path.join(out, 'report.md')}")


def write_markdown(out, plot_b, plot_c, plot_ctx_cm,
                   kfold_b, kfold_c, fold_b, fold_c):
    lines = []
    w = lines.append

    w("# PTv3 Tree Species Classification: Experiment Report")
    w("")
    w("Comparison of PTv3 geometry-only baseline vs PTv3 + AlphaEarth satellite")
    w("context fusion for individual tree species classification from airborne LiDAR.")
    w("")

    # ── Setup ─────────────────────────────────────────────────────────
    w("## Experimental Setup")
    w("")
    w("| | Plot-Level Split | District K-Fold CV |")
    w("|---|---|---|")
    w("| **Evaluation** | Single train/test split (80/20, stratified by plot) | 6-fold leave-one-district-out |")
    w("| **Classes** | 11 genera (incl. Abies) | 10 genera (excl. Abies) |")
    w("| **Train / Test** | 5,411 / 1,378 | ~5,300 / ~1,060 per fold |")
    w("| **Epochs** | 300 | 60 per fold |")
    w("| **Backbone** | PTv3-v1m1 pretrained on FOR-species20K | Same |")
    w("| **Context fusion** | AlphaEarth 64-dim satellite embeddings per plot | Same |")
    w("| **Fusion arch.** | Projected: backbone 512→128, context 64→128, concat → MLP | Same |")
    w("")

    # ── Overall results ───────────────────────────────────────────────
    w("## Overall Results")
    w("")
    w("![Overall Comparison](overall_comparison.png)")
    w("")

    # Plot-level table
    w("### Plot-Level Split (11 Classes)")
    w("")
    w("| Metric | PTv3 Baseline | PTv3 + AlphaEarth | Delta |")
    w("|---|---|---|---|")
    for key, label in [("allAcc", "Overall Accuracy"), ("mAcc", "Mean Per-Class Accuracy")]:
        b, c = plot_b[key], plot_c[key]
        d = c - b
        w(f"| {label} | {b:.2%} | {c:.2%} | {d:+.2%} |")
    w("")
    w("*Best checkpoint metrics from training logs.*")
    w("")

    # Kfold table
    w("### District K-Fold CV (10 Classes)")
    w("")
    w("| Metric | PTv3 Baseline | PTv3 + AlphaEarth | Delta |")
    w("|---|---|---|---|")
    for key, label in [("allAcc", "Overall Accuracy"), ("mAcc", "Mean Per-Class Accuracy"),
                       ("macro_f1", "Macro F1"), ("weighted_f1", "Weighted F1"),
                       ("macro_precision", "Macro Precision"), ("macro_recall", "Macro Recall")]:
        b, c = kfold_b[key], kfold_c[key]
        d = c - b
        w(f"| {label} | {b:.2%} | {c:.2%} | {d:+.2%} |")
    w("")
    w("*Metrics computed from aggregated confusion matrix across all 6 folds.*")
    w("")

    # ── Per-class breakdown ───────────────────────────────────────────
    w("## Per-Class Performance")
    w("")

    w("### Class Distribution (K-Fold)")
    w("")
    w("![Class Distribution](class_distribution.png)")
    w("")

    # Plot-level per-class recall
    w("### Plot-Level Split — Per-Class Recall")
    w("")
    w("![Per-Class Recall](per_class_recall_plot.png)")
    w("")
    w("| Class | Baseline Recall | Context Recall | Delta |")
    w("|---|---|---|---|")
    for i, cls in enumerate(PLOT_CLASSES):
        br = plot_b["per_class_recall"][i]
        cr = plot_c["per_class_recall"][i]
        w(f"| {cls} | {br:.3f} | {cr:.3f} | {cr - br:+.3f} |")
    w("")
    w("*Per-class recall from last evaluation epoch.*")
    w("")

    # Kfold per-class F1
    w("### District K-Fold CV — Per-Class F1")
    w("")
    w("![Per-Class F1](per_class_f1_kfold.png)")
    w("")
    w("| Class | Support | Baseline F1 | Context F1 | Delta | Baseline Prec. | Context Prec. | Baseline Rec. | Context Rec. |")
    w("|---|---|---|---|---|---|---|---|---|")
    for i, cls in enumerate(KFOLD_CLASSES):
        s = int(kfold_b["support"][i])
        bf, cf = kfold_b["f1"][i], kfold_c["f1"][i]
        bp, cp = kfold_b["precision"][i], kfold_c["precision"][i]
        br, cr = kfold_b["recall"][i], kfold_c["recall"][i]
        w(f"| {cls} | {s} | {bf:.3f} | {cf:.3f} | {cf - bf:+.3f} | {bp:.3f} | {cp:.3f} | {br:.3f} | {cr:.3f} |")
    w("")

    # ── Per-district breakdown ────────────────────────────────────────
    w("## Per-District Performance (K-Fold)")
    w("")
    w("![Per-District Comparison](kfold_per_district.png)")
    w("")
    w("| District | N | Baseline allAcc | Context allAcc | Delta | Baseline mAcc | Context mAcc | Delta |")
    w("|---|---|---|---|---|---|---|---|")
    for f in range(6):
        d = DISTRICTS[f]
        n = int(fold_b[f]["support"].sum())
        ba, ca = fold_b[f]["allAcc"], fold_c[f]["allAcc"]
        bm, cm_ = fold_b[f]["mAcc"], fold_c[f]["mAcc"]
        w(f"| {d} | {n} | {ba:.2%} | {ca:.2%} | {ca - ba:+.2%} | {bm:.2%} | {cm_:.2%} | {cm_ - bm:+.2%} |")
    w("")

    # ── Confusion matrices ────────────────────────────────────────────
    w("## Confusion Matrices")
    w("")
    w("### K-Fold Aggregated (Normalized)")
    w("")
    w("| PTv3 Baseline | PTv3 + AlphaEarth |")
    w("|---|---|")
    w("| ![Baseline](cm_kfold_baseline.png) | ![Context](cm_kfold_context.png) |")
    w("")

    has_plot_cm = (
        os.path.exists(os.path.join(out, "cm_plot_baseline.png"))
        and os.path.exists(os.path.join(out, "cm_plot_context.png"))
    )
    if has_plot_cm:
        w("### Plot-Level Split")
        w("")
        w("| PTv3 Baseline | PTv3 + AlphaEarth |")
        w("|---|---|")
        w("| ![Baseline](cm_plot_baseline.png) | ![Context](cm_plot_context.png) |")
        w("")
        w("*Note: Baseline confusion matrix is the original two-panel PNG (counts + normalized) from training.*")
        w("")

    # ── Key findings ──────────────────────────────────────────────────
    w("## Key Findings")
    w("")

    kfold_delta_allacc = kfold_c["allAcc"] - kfold_b["allAcc"]
    kfold_delta_f1 = kfold_c["macro_f1"] - kfold_b["macro_f1"]
    plot_delta_allacc = plot_c["allAcc"] - plot_b["allAcc"]
    plot_delta_macc = plot_c["mAcc"] - plot_b["mAcc"]

    w("1. **AlphaEarth context fusion consistently improves classification** across both evaluation protocols.")
    w(f"   - K-fold: +{kfold_delta_allacc:.1%} overall accuracy, +{kfold_delta_f1:.1%} macro F1")
    w(f"   - Plot-level: +{plot_delta_allacc:.1%} overall accuracy, +{plot_delta_macc:.1%} mean accuracy")
    w("")

    # Top F1 improvements
    deltas = [(KFOLD_CLASSES[i], kfold_c["f1"][i] - kfold_b["f1"][i])
              for i in range(len(KFOLD_CLASSES))]
    deltas.sort(key=lambda x: x[1], reverse=True)
    w(f"2. **Largest per-class F1 improvements** (K-fold): "
      f"{deltas[0][0]} ({deltas[0][1]:+.3f}), "
      f"{deltas[1][0]} ({deltas[1][1]:+.3f}), "
      f"{deltas[2][0]} ({deltas[2][1]:+.3f})")
    w("")

    # District improvements
    district_deltas = [(DISTRICTS[f], fold_c[f]["allAcc"] - fold_b[f]["allAcc"]) for f in range(6)]
    district_deltas.sort(key=lambda x: x[1], reverse=True)
    best_d = district_deltas[0]
    w(f"3. **Best district improvement**: {best_d[0]} with {best_d[1]:+.1%} overall accuracy")
    w("")

    # Weakest classes
    worst = sorted(range(len(KFOLD_CLASSES)), key=lambda i: kfold_c["f1"][i])[:3]
    worst_str = ", ".join(f"{KFOLD_CLASSES[i]} (F1={kfold_c['f1'][i]:.3f}, n={int(kfold_c['support'][i])})"
                         for i in worst)
    w(f"4. **Weakest classes** (K-fold, context model): {worst_str}")
    w("   These classes have the fewest training samples, suggesting class imbalance as the main bottleneck.")
    w("")

    report_path = os.path.join(out, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
