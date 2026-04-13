"""
Compare val_id / val_ood / test metrics side-by-side across models.

Val metrics are read from the epoch that produced model_best_ood.pth (i.e. the
epoch where allAcc_ood last improved during training).  Test metrics are read
from eval_results/{model}/metrics.json.

Usage (from context_classification_ptv3/):
    python visualize_results_splits.py

Writes:  Pointcept/eval_results/comparison_splits/
"""

import re, json, csv, os, copy
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT_EXP  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "Pointcept", "exp", "snapshot_10class_dual_val")
ROOT_EVAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "Pointcept", "eval_results")
OUT       = os.path.join(ROOT_EVAL, "comparison_splits")
os.makedirs(OUT, exist_ok=True)

# Display label → (exp_dir_name, eval_dir_name)
MODELS = {
    "PTv3":         ("ptv3_small_4gpu_120ep",                   "ptv3_baseline_120ep"),
    "PTv3+SINR":    ("ptv3_ctx_sinr_gauss_4gpu_120ep",          "ptv3_sinr_gauss_120ep"),
    "PTv3+AE":      ("ptv3_ctx_ae_combined_aug_vmf_4gpu_120ep", "ptv3_ae_combined_aug_vmf_120ep"),
    "PTv3+AE+SINR": ("ptv3_ctx_ae_sinr_cat_aux_4gpu_200ep",     "ptv3_ae_sinr_cat_aux_200ep"),
}

PALETTE_SPLIT = {
    "val_id":  "#4C72B0",
    "val_ood": "#DD8452",
    "test":    "#55A868",
}
MODEL_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

CLASS_NAMES = ["Abies", "Acer", "Alnus", "Betula", "Carpinus",
               "Fagus", "Larix", "Picea", "Pinus", "Quercus"]

DATA_ROOT = "/net/pr2/projects/plgrid/plggtreeseg/data/snapshot_10class_dual_val_npy"

def genus_counts_from_txt(split):
    """Count trees per genus from the split txt file."""
    txt = os.path.join(DATA_ROOT, f"standardized_dataset_{split}.txt")
    counts = {g: 0 for g in CLASS_NAMES}
    with open(txt) as f:
        for line in f:
            genus = line.strip().split("/")[0]
            if genus in counts:
                counts[genus] += 1
    return counts

genus_n = {split: genus_counts_from_txt(split) for split in ["val_id", "val_ood", "test"]}

# ---------------------------------------------------------------------------
# Log parser — returns val_id / val_ood dicts at the best_ood epoch
# ---------------------------------------------------------------------------
def parse_best_ood_epoch(log_path):
    """
    Scan train.log and return (val_id_dict, val_ood_dict) for the epoch whose
    val_ood allAcc was the all-time best (== the epoch saved as model_best_ood.pth).

    Each returned dict has:
        allAcc      float
        mAcc        float
        per_genus   {genus_name: acc}   (10 entries)
        per_dataset {dataset_name: acc}
    """
    cur_id  = None
    cur_ood = None
    best_ood_val = -1.0
    best_id  = None
    best_ood = None

    re_id_summary  = re.compile(r'val_id: mIoU/mAcc/allAcc ([\d.]+)/([\d.]+)/([\d.]+)')
    re_ood_summary = re.compile(r'val_ood: mIoU/mAcc/allAcc ([\d.]+)/([\d.]+)/([\d.]+)')
    re_class       = re.compile(r'(val_id|val_ood) Class_\d+-(\w+): iou/acc [\d.]+/([\d.]+)')
    re_dataset     = re.compile(r'(val_id|val_ood) dataset (\S+?) acc ([\d.]+)')
    re_best_ood    = re.compile(r'Currently Best allAcc_ood: ([\d.]+)')

    with open(log_path) as f:
        for line in f:
            m = re_id_summary.search(line)
            if m:
                cur_id = dict(allAcc=float(m.group(3)), mAcc=float(m.group(2)),
                              per_genus={}, per_dataset={})
                continue
            m = re_ood_summary.search(line)
            if m:
                cur_ood = dict(allAcc=float(m.group(3)), mAcc=float(m.group(2)),
                               per_genus={}, per_dataset={})
                continue
            m = re_class.search(line)
            if m:
                split, genus, acc = m.group(1), m.group(2), float(m.group(3))
                (cur_id if split == 'val_id' else cur_ood)['per_genus'][genus] = acc
                continue
            m = re_dataset.search(line)
            if m:
                split, ds, acc = m.group(1), m.group(2).rstrip(':'), float(m.group(3))
                (cur_id if split == 'val_id' else cur_ood)['per_dataset'][ds] = acc
                continue
            m = re_best_ood.search(line)
            if m:
                val = float(m.group(1))
                if val > best_ood_val and cur_id is not None and cur_ood is not None:
                    best_ood_val = val
                    best_id  = copy.deepcopy(cur_id)
                    best_ood = copy.deepcopy(cur_ood)

    return best_id, best_ood

# ---------------------------------------------------------------------------
# Load test metrics from metrics.json + predictions.csv
# ---------------------------------------------------------------------------
def load_test(eval_dir):
    path = os.path.join(ROOT_EVAL, eval_dir, "metrics.json")
    with open(path) as f:
        m = json.load(f)
    d = dict(
        allAcc = m["overall"]["all_acc"],
        mAcc   = m["overall"]["m_acc"],
        per_genus   = {g: v["acc"] for g, v in m["per_genus"].items()},
        per_dataset = {ds: v["acc"] for ds, v in m["per_dataset"].items()},
    )
    return d

# ---------------------------------------------------------------------------
# Gather all data
# ---------------------------------------------------------------------------
model_labels = list(MODELS.keys())
data = {}   # label → {"val_id": ..., "val_ood": ..., "test": ...}

for label, (exp_dir, eval_dir) in MODELS.items():
    log = os.path.join(ROOT_EXP, exp_dir, "train.log")
    val_id, val_ood = parse_best_ood_epoch(log)
    test = load_test(eval_dir)
    data[label] = {"val_id": val_id, "val_ood": val_ood, "test": test}
    print(f"{label}: val_id allAcc={val_id['allAcc']:.4f}  "
          f"val_ood allAcc={val_ood['allAcc']:.4f}  "
          f"test allAcc={test['allAcc']:.4f}")

splits       = ["val_id", "val_ood", "test"]
split_labels = {"val_id": "Val ID", "val_ood": "Val OOD", "test": "Test"}

# ---------------------------------------------------------------------------
# 1. Overall metrics — allAcc and mAcc side-by-side
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
fig.suptitle("Overall Metrics by Split", fontsize=13, fontweight="bold")

n_models = len(model_labels)
n_splits  = len(splits)
width     = 0.22
x         = np.arange(n_models)

for ax, metric, title in zip(axes, ["allAcc", "mAcc"], ["Overall Accuracy", "Mean Accuracy"]):
    for si, split in enumerate(splits):
        vals   = [data[m][split][metric] for m in model_labels]
        offset = (si - n_splits / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width,
                        label=split_labels[split],
                        color=PALETTE_SPLIT[split],
                        edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6.5,
                    rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

# Shared legend below both panels
handles = [plt.Rectangle((0,0),1,1, color=PALETTE_SPLIT[s]) for s in splits]
fig.legend(handles, [split_labels[s] for s in splits],
           loc="lower center", ncol=len(splits), fontsize=9,
           frameon=True, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.07, 1, 1])
out_path = os.path.join(OUT, "1_overall_metrics.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_path}")

# ---------------------------------------------------------------------------
# 2. Per-genus accuracy heatmap — one panel per split
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
fig.suptitle("Per-Genus Accuracy by Split  (at best val_ood epoch for val; model_best_ood for test)",
             fontsize=11, fontweight="bold")

for ax, split in zip(axes, splits):
    counts = genus_n[split]
    ylabels = [f"{g}  (n={counts[g]})" for g in CLASS_NAMES]

    mat = np.array([
        [data[m][split]["per_genus"].get(g, 0.0) for m in model_labels]
        for g in CLASS_NAMES
    ])
    # Mask cells where genus has n=0 (no trees in this split)
    zero_mask = np.array([[counts[g] == 0 for _ in model_labels] for g in CLASS_NAMES])

    im = ax.imshow(mat, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(model_labels, rotation=30, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_yticklabels(ylabels, fontsize=8.5)
    ax.set_title(split_labels[split], fontsize=11, fontweight="bold")

    for gi, g in enumerate(CLASS_NAMES):
        for mi in range(n_models):
            if zero_mask[gi, mi]:
                # Grey out and show dash
                ax.add_patch(plt.Rectangle((mi - 0.5, gi - 0.5), 1, 1,
                                           color="#cccccc", zorder=2))
                ax.text(mi, gi, "—", ha="center", va="center",
                        fontsize=9, color="#666666", zorder=3)
            else:
                v = mat[gi, mi]
                ax.text(mi, gi, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if v < 0.65 else "white")

fig.colorbar(im, ax=axes[-1], label="Accuracy", shrink=0.8)
out_path = os.path.join(OUT, "2_per_genus_acc_heatmap.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_path}")

# ---------------------------------------------------------------------------
# 3. Per-dataset accuracy — one panel per split
# ---------------------------------------------------------------------------
# Collect dataset lists per split
ds_per_split = {
    split: sorted({ds for m in model_labels
                   for ds in data[m][split]["per_dataset"]})
    for split in splits
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig.suptitle("Per-Dataset Accuracy by Split", fontsize=13, fontweight="bold")

handles = None
for ax, split in zip(axes, splits):
    ds_list = ds_per_split[split]
    n_ds    = len(ds_list)
    x_ds    = np.arange(n_ds)
    w       = 0.18
    bar_handles = []
    for mi, (m, color) in enumerate(zip(model_labels, MODEL_COLORS)):
        vals   = [data[m][split]["per_dataset"].get(ds, float("nan")) for ds in ds_list]
        offset = (mi - n_models / 2 + 0.5) * w
        bars   = ax.bar(x_ds + offset, vals, w,
                        label=m, color=color, edgecolor="white", linewidth=0.4)
        bar_handles.append(bars[0])
        # Value labels: only show on bars tall enough, rotated to avoid overlap
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
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(split_labels[split], fontsize=11, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

fig.legend(handles, model_labels, loc="lower center", ncol=n_models,
           fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.07, 1, 1])
out_path = os.path.join(OUT, "3_per_dataset_acc.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_path}")

print(f"\nAll plots saved to: {OUT}")
