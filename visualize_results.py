"""
Visualize and compare test-set evaluation results across multiple models.

Usage (from context_classification_ptv3/):
    python visualize_results.py

Reads:  Pointcept/eval_results/{model}/metrics.json
Writes: Pointcept/eval_results/comparison/  (PNG files)
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "Pointcept", "eval_results")
OUT  = os.path.join(BASE, "comparison")
os.makedirs(OUT, exist_ok=True)

# Ordered dict: display label → folder name
MODELS = {
    "PTv3":          "ptv3_baseline_120ep",
    "PTv3+SINR":     "ptv3_sinr_gauss_120ep",
    "PTv3+AE":       "ptv3_ae_combined_aug_vmf_120ep",
    "PTv3+AE+SINR":  "ptv3_ae_sinr_cat_aux_200ep",
}

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load(folder):
    path = os.path.join(BASE, folder, "metrics.json")
    with open(path) as f:
        return json.load(f)

data = {label: load(exp) for label, exp in MODELS.items()}
model_labels = list(data.keys())
colors = {label: PALETTE[i] for i, label in enumerate(model_labels)}

# Derived helpers
ref = next(iter(data.values()))
genus_list = [g for g, v in ref["per_genus"].items() if v["support"] > 0]
dataset_list = list(ref["per_dataset"].keys())

# ---------------------------------------------------------------------------
# 1. Overall metrics bar chart
# ---------------------------------------------------------------------------
METRICS = [
    ("all_acc",    "Overall Acc"),
    ("m_acc",      "Mean Acc"),
    ("macro_f1",   "Macro F1"),
    ("weighted_f1","Weighted F1"),
]

fig, axes = plt.subplots(1, 4, figsize=(13, 4.5), sharey=False)
fig.suptitle("Test Set — Overall Metrics", fontsize=13, fontweight="bold", y=1.02)

for ax, (key, label) in zip(axes, METRICS):
    vals = [data[m]["overall"][key] for m in model_labels]
    bars = ax.bar(range(len(model_labels)), vals,
                  color=[colors[m] for m in model_labels],
                  edgecolor="white", linewidth=0.5)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(model_labels)))
    ax.set_xticklabels(model_labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

plt.tight_layout()
out_path = os.path.join(OUT, "1_overall_metrics.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_path}")

# ---------------------------------------------------------------------------
# 2. Per-genus F1 grouped bar chart
# ---------------------------------------------------------------------------
n_genera = len(genus_list)
n_models  = len(model_labels)
width = 0.18
x = np.arange(n_genera)

ns_by_genus = {g: ref["per_genus"][g]["support"] for g in genus_list}

fig, ax = plt.subplots(figsize=(13, 5))
for mi, m in enumerate(model_labels):
    f1_vals = [data[m]["per_genus"][g]["f1"] for g in genus_list]
    offset  = (mi - n_models / 2 + 0.5) * width
    ax.bar(x + offset, f1_vals, width,
           label=m, color=colors[m], edgecolor="white", linewidth=0.4)

ax.set_xticks(x)
ax.set_xticklabels([f"{g}\n(n={ns_by_genus[g]})" for g in genus_list], fontsize=9)
ax.set_ylabel("F1 Score")
ax.set_ylim(0, 1.05)
ax.set_title("Test Set — Per-Genus F1", fontsize=12, fontweight="bold")
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.legend(fontsize=9, loc="upper right")
plt.tight_layout()
out_path = os.path.join(OUT, "2_per_genus_f1.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_path}")

# ---------------------------------------------------------------------------
# 3. Per-genus accuracy grouped bar chart
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 5))
for mi, m in enumerate(model_labels):
    acc_vals = [data[m]["per_genus"][g]["acc"] for g in genus_list]
    offset   = (mi - n_models / 2 + 0.5) * width
    ax.bar(x + offset, acc_vals, width,
           label=m, color=colors[m], edgecolor="white", linewidth=0.4)

ax.set_xticks(x)
ax.set_xticklabels([f"{g}\n(n={ns_by_genus[g]})" for g in genus_list], fontsize=9)
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.1)
ax.set_title("Test Set — Per-Genus Accuracy", fontsize=12, fontweight="bold")
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.legend(fontsize=9, loc="upper right")
plt.tight_layout()
out_path = os.path.join(OUT, "3_per_genus_acc.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_path}")

# ---------------------------------------------------------------------------
# 4. Per-dataset accuracy grouped bar chart
# ---------------------------------------------------------------------------
n_ds   = len(dataset_list)
width_ds = 0.18
x_ds = np.arange(n_ds)
ds_ns = {d: ref["per_dataset"][d]["total"] for d in dataset_list}

fig, ax = plt.subplots(figsize=(8, 5))
for mi, m in enumerate(model_labels):
    vals   = [data[m]["per_dataset"][d]["acc"] for d in dataset_list]
    offset = (mi - n_models / 2 + 0.5) * width_ds
    bars   = ax.bar(x_ds + offset, vals, width_ds,
                    label=m, color=colors[m], edgecolor="white", linewidth=0.4)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x_ds)
ax.set_xticklabels([f"{d}\n(n={ds_ns[d]})" for d in dataset_list], fontsize=10)
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.1)
ax.set_title("Test Set — Per-Dataset Accuracy", fontsize=12, fontweight="bold")
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.legend(fontsize=9)
plt.tight_layout()
out_path = os.path.join(OUT, "4_per_dataset_acc.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_path}")

# ---------------------------------------------------------------------------
# 5. Heatmap: genus F1 × model
# ---------------------------------------------------------------------------
f1_matrix = np.array([
    [data[m]["per_genus"][g]["f1"] for m in model_labels]
    for g in genus_list
])

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(f1_matrix, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(n_models))
ax.set_xticklabels(model_labels, rotation=30, ha="right", fontsize=9)
ax.set_yticks(range(n_genera))
ax.set_yticklabels([f"{g} (n={ns_by_genus[g]})" for g in genus_list], fontsize=9)
ax.set_title("Per-Genus F1 Heatmap", fontsize=12, fontweight="bold")
plt.colorbar(im, ax=ax, label="F1 Score")

for gi in range(n_genera):
    for mi in range(n_models):
        v = f1_matrix[gi, mi]
        ax.text(mi, gi, f"{v:.2f}", ha="center", va="center",
                fontsize=8.5, color="black" if v < 0.65 else "white")

plt.tight_layout()
out_path = os.path.join(OUT, "5_genus_f1_heatmap.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_path}")

# ---------------------------------------------------------------------------
# 6. Per-dataset genus F1 heatmaps (3 panels, one per dataset)
# ---------------------------------------------------------------------------
import csv
from collections import defaultdict

def compute_per_dataset_genus_f1(folder):
    """Read predictions.csv and return {dataset: {genus: f1}}."""
    csv_path = os.path.join(BASE, folder, "predictions.csv")
    # Accumulate tp, fp, fn per (dataset, genus)
    tp = defaultdict(lambda: defaultdict(int))
    fp = defaultdict(lambda: defaultdict(int))
    fn = defaultdict(lambda: defaultdict(int))
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds    = row["dataset"]
            true  = row["true_genus"]
            pred  = row["pred_genus"]
            if true == pred:
                tp[ds][true] += 1
            else:
                fn[ds][true] += 1
                fp[ds][pred]  += 1
    result = {}
    for ds in tp:
        result[ds] = {}
        genera_in_ds = set(tp[ds]) | set(fn[ds])
        for g in genera_in_ds:
            t = tp[ds].get(g, 0)
            p = fp[ds].get(g, 0)
            n = fn[ds].get(g, 0)
            prec = t / (t + p) if (t + p) > 0 else 0.0
            rec  = t / (t + n) if (t + n) > 0 else 0.0
            result[ds][g] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return result

# Build {model: {dataset: {genus: f1}}}
per_ds_genus = {
    label: compute_per_dataset_genus_f1(folder)
    for label, folder in MODELS.items()
}

# Genus counts per dataset (from any model's predictions)
def genus_counts_per_dataset(folder):
    csv_path = os.path.join(BASE, folder, "predictions.csv")
    counts = defaultdict(lambda: defaultdict(int))
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            counts[row["dataset"]][row["true_genus"]] += 1
    return counts

genus_counts = genus_counts_per_dataset(list(MODELS.values())[0])

fig, axes = plt.subplots(1, len(dataset_list),
                         figsize=(5 * len(dataset_list), 6),
                         constrained_layout=True)
fig.suptitle("Per-Dataset × Per-Genus F1", fontsize=13, fontweight="bold")

for ax, ds in zip(axes, dataset_list):
    # Genera present in this dataset (sorted)
    ds_genera = sorted(
        [g for g, n in genus_counts[ds].items() if n > 0]
    )
    n_g = len(ds_genera)

    mat = np.array([
        [per_ds_genus[m][ds].get(g, 0.0) for m in model_labels]
        for g in ds_genera
    ])

    im = ax.imshow(mat, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(model_labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(range(n_g))
    ax.set_yticklabels(
        [f"{g} (n={genus_counts[ds][g]})" for g in ds_genera],
        fontsize=9
    )
    ax.set_title(ds, fontsize=11, fontweight="bold")

    for gi in range(n_g):
        for mi in range(n_models):
            v = mat[gi, mi]
            ax.text(mi, gi, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if v < 0.65 else "white")

fig.colorbar(im, ax=axes[-1], label="F1 Score", shrink=0.8)

out_path = os.path.join(OUT, "6_per_dataset_genus_heatmaps.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_path}")

print(f"\nAll plots saved to: {OUT}")
