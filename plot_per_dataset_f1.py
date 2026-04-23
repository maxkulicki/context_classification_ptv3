"""
Regenerate 3_per_dataset_f1.png — per-dataset macro F1 across splits.

Macro F1 per dataset is computed from predictions.csv (for all three splits).
Classes with support=0 in a given dataset are skipped.

Usage:
    conda run -n context_baseline python plot_per_dataset_f1.py
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

ROOT_EVAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "Pointcept", "eval_results")
OUT = os.path.join(ROOT_EVAL, "comparison_splits")
os.makedirs(OUT, exist_ok=True)

MODELS = {
    "PTv3":         {"val_id": "val_id/ptv3_baseline_120ep",
                     "val_ood": "val_ood/ptv3_baseline_120ep",
                     "test":    "ptv3_baseline_120ep"},
    "PTv3+SINR":    {"val_id": "val_id/ptv3_sinr_gauss_120ep",
                     "val_ood": "val_ood/ptv3_sinr_gauss_120ep",
                     "test":    "ptv3_sinr_gauss_120ep"},
    "PTv3+AE":      {"val_id": "val_id/ptv3_ae_combined_aug_vmf_120ep",
                     "val_ood": "val_ood/ptv3_ae_combined_aug_vmf_120ep",
                     "test":    "ptv3_ae_combined_aug_vmf_120ep"},
    "PTv3+AE+SINR": {"val_id": "val_id/ptv3_ae_sinr_cat_aux_200ep",
                     "val_ood": "val_ood/ptv3_ae_sinr_cat_aux_200ep",
                     "test":    "ptv3_ae_sinr_cat_aux_200ep"},
}

MODEL_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
splits       = ["val_id", "val_ood", "test"]
split_labels = {"val_id": "Val ID", "val_ood": "Val OOD", "test": "Test"}


def macro_f1_from_predictions(csv_path):
    """Return {dataset: macro_f1} computed from predictions.csv."""
    # Collect per-dataset confusion: {ds: {true: {pred: count}}}
    per_ds = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    classes = set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds  = row["dataset"]
            tg  = row["true_genus"]
            pg  = row["pred_genus"]
            per_ds[ds][tg][pg] += 1
            classes.add(tg)
            classes.add(pg)

    classes = sorted(classes)
    result = {}
    for ds, genus_dict in per_ds.items():
        tp = fp = fn = 0
        f1s = []
        for g in classes:
            tp_g = genus_dict[g][g]
            fp_g = sum(genus_dict[other][g] for other in classes if other != g)
            fn_g = sum(genus_dict[g][other] for other in classes if other != g)
            support = tp_g + fn_g
            if support == 0:
                continue   # absent class — skip
            prec = tp_g / (tp_g + fp_g) if (tp_g + fp_g) > 0 else 0.0
            rec  = tp_g / (tp_g + fn_g) if (tp_g + fn_g) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)
        result[ds] = float(np.mean(f1s)) if f1s else 0.0
    return result


# ---------------------------------------------------------------------------
# Gather data
# ---------------------------------------------------------------------------
model_labels = list(MODELS.keys())
data = {}   # label → split → {ds: f1}

for label, split_dirs in MODELS.items():
    data[label] = {}
    for split, rel_dir in split_dirs.items():
        csv_path = os.path.join(ROOT_EVAL, rel_dir, "predictions.csv")
        data[label][split] = macro_f1_from_predictions(csv_path)
        print(f"{label} {split}: {data[label][split]}")

# Per-split dataset list (sorted, union across models)
ds_per_split = {
    split: sorted({ds for label in model_labels
                   for ds in data[label][split]})
    for split in splits
}

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
n_models = len(model_labels)
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig.suptitle("Per-Dataset Macro F1 by Split", fontsize=13, fontweight="bold")

handles = None
for ax, split in zip(axes, splits):
    ds_list = ds_per_split[split]
    n_ds    = len(ds_list)
    x_ds    = np.arange(n_ds)
    w       = 0.18
    bar_handles = []
    for mi, (label, color) in enumerate(zip(model_labels, MODEL_COLORS)):
        vals   = [data[label][split].get(ds, float("nan")) for ds in ds_list]
        offset = (mi - n_models / 2 + 0.5) * w
        bars   = ax.bar(x_ds + offset, vals, w,
                        label=label, color=color,
                        edgecolor="white", linewidth=0.4)
        bar_handles.append(bars[0])
        for bar, v in zip(bars, vals):
            if not np.isnan(v) and v > 0.04:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        min(v + 0.005, 0.97),
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=6, rotation=90, clip_on=False)
    if handles is None:
        handles = bar_handles
    ax.set_xticks(x_ds)
    ax.set_xticklabels(ds_list, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.05)
    ax.set_title(split_labels[split], fontsize=11, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

fig.legend(handles, model_labels, loc="lower center", ncol=n_models,
           fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.07, 1, 1])

out_path = os.path.join(OUT, "3_per_dataset_f1.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_path}")
