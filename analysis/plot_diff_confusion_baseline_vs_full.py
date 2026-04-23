"""
3-panel confusion matrix: PTv3 baseline | PTv3+AE+SINR | diff (full − baseline).
All on val_ood. Row-normalised (recall per true class).

Usage:
    conda run -n context_baseline python plot_diff_confusion_baseline_vs_full.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_EVAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "Pointcept", "eval_results", "val_ood")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "Pointcept", "eval_results", "comparison_splits")
os.makedirs(OUT, exist_ok=True)

CLASS_NAMES = ["Abies", "Acer", "Alnus", "Betula", "Carpinus",
               "Fagus", "Larix", "Picea", "Pinus", "Quercus"]

def load_cm(name):
    return np.load(os.path.join(ROOT_EVAL, name, "confusion_matrix.npy")).astype(float)

def row_normalise(cm):
    s = cm.sum(axis=1, keepdims=True)
    return np.where(s > 0, cm / s, 0.0)

cm_base = load_cm("ptv3_baseline_120ep")
cm_full = load_cm("ptv3_ae_sinr_cat_aux_200ep")
cn_base = row_normalise(cm_base)
cn_full = row_normalise(cm_full)
diff    = cn_full - cn_base

support   = cm_base.sum(axis=1).astype(int)
ylabels   = [f"{g}  (n={support[i]})" for i, g in enumerate(CLASS_NAMES)]
n         = len(CLASS_NAMES)

fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
fig.suptitle("Val OOD — row-normalised confusion  |  PTv3 baseline vs PTv3+AE+SINR",
             fontsize=12, fontweight="bold")

# ── Panel 0: baseline ────────────────────────────────────────────────────────
ax = axes[0]
im0 = ax.imshow(cn_base, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_title("PTv3 (baseline)", fontsize=11, fontweight="bold")
ax.set_xticks(range(n)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8.5)
ax.set_yticks(range(n)); ax.set_yticklabels(ylabels, fontsize=8.5)
ax.set_ylabel("True genus", fontsize=9)
ax.set_xlabel("Predicted genus", fontsize=9)
for i in range(n):
    for j in range(n):
        v = cn_base[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                color="white" if v > 0.5 else "black")
fig.colorbar(im0, ax=ax, shrink=0.8, label="Recall")

# ── Panel 1: full model ───────────────────────────────────────────────────────
ax = axes[1]
im1 = ax.imshow(cn_full, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_title("PTv3+AE+SINR", fontsize=11, fontweight="bold")
ax.set_xticks(range(n)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8.5)
ax.set_yticks(range(n)); ax.set_yticklabels(ylabels, fontsize=8.5)
ax.set_xlabel("Predicted genus", fontsize=9)
for i in range(n):
    for j in range(n):
        v = cn_full[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                color="white" if v > 0.5 else "black")
fig.colorbar(im1, ax=ax, shrink=0.8, label="Recall")

# ── Panel 2: differential ─────────────────────────────────────────────────────
ax = axes[2]
im2 = ax.imshow(diff, cmap="RdYlGn", vmin=-0.6, vmax=0.6, aspect="auto")
ax.set_title("Diff  (PTv3+AE+SINR − PTv3)", fontsize=11, fontweight="bold")
ax.set_xticks(range(n)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8.5)
ax.set_yticks(range(n)); ax.set_yticklabels(CLASS_NAMES, fontsize=8.5)
ax.set_xlabel("Predicted genus", fontsize=9)
for i in range(n):
    for j in range(n):
        v = diff[i, j]
        color = "black" if abs(v) < 0.3 else "white"
        sign  = "+" if v > 0 else ""
        ax.text(j, i, f"{sign}{v:.2f}", ha="center", va="center", fontsize=7.5, color=color)
fig.colorbar(im2, ax=ax, shrink=0.8, label="Δ recall")

out_path = os.path.join(OUT, "5_confusion_baseline_vs_full_val_ood.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_path}")
