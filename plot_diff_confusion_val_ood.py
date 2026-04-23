"""
Differential confusion matrices for val_ood — 4 main models.

Left panel: PTv3 baseline (row-normalised).
Remaining 3 panels: difference (model − baseline) in row-normalised confusion.
  Green = model improves over baseline for that cell.
  Red   = model is worse.

Usage:
    conda run -n context_baseline python plot_diff_confusion_val_ood.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT_EVAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "Pointcept", "eval_results", "val_ood")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "Pointcept", "eval_results", "comparison_splits")
os.makedirs(OUT, exist_ok=True)

CLASS_NAMES = ["Abies", "Acer", "Alnus", "Betula", "Carpinus",
               "Fagus", "Larix", "Picea", "Pinus", "Quercus"]

MODELS = {
    "PTv3":         "ptv3_baseline_120ep",
    "PTv3+SINR":    "ptv3_sinr_gauss_120ep",
    "PTv3+AE":      "ptv3_ae_combined_aug_vmf_120ep",
    "PTv3+AE+SINR": "ptv3_ae_sinr_cat_aux_200ep",
}

def load_cm(name):
    path = os.path.join(ROOT_EVAL, name, "confusion_matrix.npy")
    return np.load(path).astype(float)

def row_normalise(cm):
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.where(row_sums > 0, cm / row_sums, 0.0)

cms = {label: load_cm(d) for label, d in MODELS.items()}
cms_norm = {label: row_normalise(cm) for label, cm in cms.items()}

baseline_label = "PTv3"
baseline_norm  = cms_norm[baseline_label]

# Row sums from baseline (for support annotation)
baseline_support = cms[baseline_label].sum(axis=1).astype(int)

model_labels = list(MODELS.keys())
n = len(CLASS_NAMES)

fig, axes = plt.subplots(1, 4, figsize=(22, 6), constrained_layout=True)
fig.suptitle("Val OOD confusion matrices — row-normalised  |  diff = model − PTv3 baseline",
             fontsize=12, fontweight="bold")

# ── Panel 0: baseline (absolute normalised) ──────────────────────────────────
ax = axes[0]
im0 = ax.imshow(baseline_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_title(f"{baseline_label}\n(baseline)", fontsize=11, fontweight="bold")
ax.set_xticks(range(n)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(n))
ylabels = [f"{g}  (n={baseline_support[i]})" for i, g in enumerate(CLASS_NAMES)]
ax.set_yticklabels(ylabels, fontsize=8)
ax.set_ylabel("True genus", fontsize=9)
ax.set_xlabel("Predicted genus", fontsize=9)
thresh = 0.5
for i in range(n):
    for j in range(n):
        v = baseline_norm[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                color="white" if v > thresh else "black")
fig.colorbar(im0, ax=ax, shrink=0.8, label="Recall")

# ── Panels 1–3: differential ─────────────────────────────────────────────────
diff_max = 0.6   # symmetric clim
cmap_div = plt.cm.RdYlGn   # red=worse, green=better

for ax, label in zip(axes[1:], [m for m in model_labels if m != baseline_label]):
    diff = cms_norm[label] - baseline_norm
    im = ax.imshow(diff, cmap=cmap_div, vmin=-diff_max, vmax=diff_max, aspect="auto")
    ax.set_title(f"{label}\n(diff vs PTv3)", fontsize=11, fontweight="bold")
    ax.set_xticks(range(n)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Predicted genus", fontsize=9)
    for i in range(n):
        for j in range(n):
            v = diff[i, j]
            # Show value; use dark text for near-zero cells
            color = "black" if abs(v) < 0.3 else "white"
            sign  = "+" if v > 0 else ""
            ax.text(j, i, f"{sign}{v:.2f}", ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Δ recall")

out_path = os.path.join(OUT, "4_diff_confusion_val_ood.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_path}")
