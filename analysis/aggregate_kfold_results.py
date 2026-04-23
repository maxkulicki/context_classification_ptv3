"""Aggregate confusion matrices across k-fold cross-validation runs.

Loads per-fold confusion_matrix.npy files, sums them into a single
cross-validated CM, computes per-fold and aggregated metrics, and
generates summary outputs (console table, CSV, confusion matrix PNG).

Dependencies: numpy, matplotlib (no sklearn needed).

Usage:
    python aggregate_kfold_results.py \
        --exp_base Pointcept/exp/treescanpl \
        --output_dir results/kfold
"""

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# District names corresponding to fold indices 0-5
DISTRICTS = ["Gorlice", "Herby", "Katrynka", "Milicz", "Piensk", "Suprasl"]

CLASS_NAMES = [
    "Acer", "Alnus", "Betula", "Carpinus", "Fagus",
    "Larix", "Picea", "Pinus", "Quercus", "Tilia",
]

NUM_FOLDS = 6


def metrics_from_cm(cm):
    """Compute classification metrics from a confusion matrix.

    Args:
        cm: (num_classes, num_classes) array, rows=true, cols=predicted.

    Returns:
        dict with allAcc, mAcc, macro_f1, weighted_f1, macro_precision,
        macro_recall, and per-class arrays.
    """
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)

    precision_cls = tp / (tp + fp + 1e-10)
    recall_cls = tp / (tp + fn + 1e-10)
    f1_cls = 2 * precision_cls * recall_cls / (precision_cls + recall_cls + 1e-10)

    all_acc = tp.sum() / (support.sum() + 1e-10)
    m_acc = np.mean(recall_cls)  # per-class accuracy = recall
    macro_f1 = np.mean(f1_cls)
    weighted_f1 = np.average(f1_cls, weights=support) if support.sum() > 0 else 0.0
    macro_precision = np.mean(precision_cls)
    macro_recall = np.mean(recall_cls)

    return dict(
        allAcc=all_acc,
        mAcc=m_acc,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        precision_cls=precision_cls,
        recall_cls=recall_cls,
        f1_cls=f1_cls,
        support=support,
    )


def plot_confusion_matrix(cm, class_names, out_path):
    """Generate two-panel confusion matrix PNG (counts + row-normalized)."""
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    # Panel 1: raw counts
    im1 = axes[0].imshow(cm, interpolation="nearest", cmap="Blues")
    axes[0].set_title("Confusion Matrix (counts)", fontsize=14)
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_xticks(range(len(class_names)))
    axes[0].set_yticks(range(len(class_names)))
    axes[0].set_xticklabels(class_names, rotation=45, ha="right")
    axes[0].set_yticklabels(class_names)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    # Annotate cells
    thresh1 = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            axes[0].text(
                j, i, f"{cm[i, j]:d}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh1 else "black",
                fontsize=9,
            )

    # Panel 2: row-normalized (recall per class)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
    im2 = axes[1].imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Confusion Matrix (normalized by true class)", fontsize=14)
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_xticks(range(len(class_names)))
    axes[1].set_yticks(range(len(class_names)))
    axes[1].set_xticklabels(class_names, rotation=45, ha="right")
    axes[1].set_yticklabels(class_names)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    thresh2 = 0.5
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            axes[1].text(
                j, i, f"{cm_norm[i, j]:.2f}",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh2 else "black",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate k-fold cross-validation results"
    )
    parser.add_argument(
        "--exp_base",
        type=str,
        default="Pointcept/exp/treescanpl",
        help="Base experiment directory containing fold subdirs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/kfold",
        help="Output directory for aggregated results",
    )
    parser.add_argument(
        "--exp_prefix",
        type=str,
        default="cls-ptv3-v1m1-0-base-finetune-kfold",
        help="Experiment name prefix (before -foldN-District)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load per-fold confusion matrices
    fold_cms = []
    fold_metrics = []
    missing_folds = []

    for fold_idx in range(NUM_FOLDS):
        district = DISTRICTS[fold_idx]
        exp_name = f"{args.exp_prefix}-fold{fold_idx}-{district}"
        cm_path = os.path.join(args.exp_base, exp_name, "confusion_matrix.npy")

        if not os.path.exists(cm_path):
            missing_folds.append((fold_idx, district, cm_path))
            continue

        cm = np.load(cm_path)
        fold_cms.append(cm)
        m = metrics_from_cm(cm)
        m["fold"] = fold_idx
        m["district"] = district
        m["n_samples"] = int(cm.sum())
        fold_metrics.append(m)
        print(f"Fold {fold_idx} ({district}): loaded CM with {m['n_samples']} samples")

    if missing_folds:
        print(f"\nWARNING: {len(missing_folds)} fold(s) missing:")
        for fold_idx, district, path in missing_folds:
            print(f"  Fold {fold_idx} ({district}): {path}")

    if not fold_cms:
        print("ERROR: No fold confusion matrices found. Exiting.")
        sys.exit(1)

    # Aggregate: sum all CMs
    cm_total = sum(fold_cms)
    total_samples = int(cm_total.sum())
    agg_metrics = metrics_from_cm(cm_total)
    print(f"\nTotal samples in aggregated CM: {total_samples}")

    # Compute mean +/- std across folds
    metric_keys = ["allAcc", "mAcc", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]
    fold_values = {k: [m[k] for m in fold_metrics] for k in metric_keys}
    means = {k: np.mean(v) for k, v in fold_values.items()}
    stds = {k: np.std(v) for k, v in fold_values.items()}

    # Console summary
    print("\n" + "=" * 70)
    print("K-FOLD CROSS-VALIDATION SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<20} {'Mean':>10} {'Std':>10} {'Aggregated':>12}")
    print("-" * 52)
    for k in metric_keys:
        print(f"{k:<20} {means[k]:>10.4f} {stds[k]:>10.4f} {agg_metrics[k]:>12.4f}")

    # Per-fold table
    print(f"\n{'Fold':<6} {'District':<12} {'N':>6} ", end="")
    for k in metric_keys:
        print(f"{k:>12}", end="")
    print()
    print("-" * (6 + 12 + 6 + 2 + len(metric_keys) * 12))
    for m in fold_metrics:
        print(f"{m['fold']:<6} {m['district']:<12} {m['n_samples']:>6} ", end="")
        for k in metric_keys:
            print(f"{m[k]:>12.4f}", end="")
        print()

    # Per-class breakdown from aggregated CM
    print(f"\n{'Class':<12} {'Support':>8} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 46)
    for i, name in enumerate(CLASS_NAMES):
        print(
            f"{name:<12} {int(agg_metrics['support'][i]):>8} "
            f"{agg_metrics['f1_cls'][i]:>8.4f} "
            f"{agg_metrics['precision_cls'][i]:>10.4f} "
            f"{agg_metrics['recall_cls'][i]:>8.4f}"
        )

    # Save CSV
    csv_path = os.path.join(args.output_dir, "kfold_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header section: per-fold metrics
        header = ["fold", "district", "n_samples"] + metric_keys
        writer.writerow(header)
        for m in fold_metrics:
            row = [m["fold"], m["district"], m["n_samples"]]
            row += [f"{m[k]:.6f}" for k in metric_keys]
            writer.writerow(row)

        # Mean/std rows
        writer.writerow([])
        writer.writerow(
            ["mean", "", total_samples] + [f"{means[k]:.6f}" for k in metric_keys]
        )
        writer.writerow(
            ["std", "", ""] + [f"{stds[k]:.6f}" for k in metric_keys]
        )
        writer.writerow(
            ["aggregated", "", total_samples]
            + [f"{agg_metrics[k]:.6f}" for k in metric_keys]
        )

        # Per-class breakdown
        writer.writerow([])
        writer.writerow(["class", "support", "f1", "precision", "recall"])
        for i, name in enumerate(CLASS_NAMES):
            writer.writerow([
                name,
                int(agg_metrics["support"][i]),
                f"{agg_metrics['f1_cls'][i]:.6f}",
                f"{agg_metrics['precision_cls'][i]:.6f}",
                f"{agg_metrics['recall_cls'][i]:.6f}",
            ])

    print(f"\nSaved CSV: {csv_path}")

    # Save aggregated CM as npy
    np.save(os.path.join(args.output_dir, "confusion_matrix_total.npy"), cm_total)

    # Generate confusion matrix plot
    plot_path = os.path.join(args.output_dir, "confusion_matrix_kfold.png")
    plot_confusion_matrix(cm_total, CLASS_NAMES, plot_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
