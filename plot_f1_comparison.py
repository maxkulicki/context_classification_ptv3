"""
Regenerate 1_overall_metrics.png showing macroF1 only (instead of allAcc/mAcc).

Val macroF1 is taken at the best val_ood epoch (same as model_best_ood.pth),
computed as mean(2*IoU_i/(1+IoU_i)) over genera from per-class IoU in the log.
Test macroF1 is read from eval_results/{model}/metrics.json.

Usage:
    conda run -n context_baseline python plot_f1_comparison.py
"""
import re, json, os, copy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_EXP  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "Pointcept", "exp", "snapshot_10class_dual_val")
ROOT_EVAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "Pointcept", "eval_results")
OUT       = os.path.join(ROOT_EVAL, "comparison_splits")
os.makedirs(OUT, exist_ok=True)

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
split_labels = {"val_id": "Val ID", "val_ood": "Val OOD", "test": "Test"}


def iou_to_f1(iou_arr):
    return float(np.mean(2 * iou_arr / (1 + iou_arr)))


def parse_best_ood_epoch(log_path):
    """Return (val_id_f1, val_ood_f1) at the epoch where val_ood allAcc peaked."""
    re_id_summary  = re.compile(r'val_id: mIoU/mAcc/allAcc')
    re_ood_summary = re.compile(r'val_ood: mIoU/mAcc/allAcc')
    re_class       = re.compile(r'(val_id|val_ood) Class_\d+-\w+: iou/acc ([\d.]+)/')
    re_best_ood    = re.compile(r'Currently Best allAcc_ood: ([\d.]+)')

    cur_id_ious  = []
    cur_ood_ious = []
    collecting   = None
    best_ood_val = -1.0
    best_id_f1   = None
    best_ood_f1  = None

    with open(log_path) as f:
        for line in f:
            if re_id_summary.search(line):
                cur_id_ious  = []
                collecting   = 'val_id'
                continue
            if re_ood_summary.search(line):
                cur_ood_ious = []
                collecting   = 'val_ood'
                continue
            m = re_class.search(line)
            if m:
                split, iou = m.group(1), float(m.group(2))
                if split == 'val_id':  cur_id_ious.append(iou)
                else:                  cur_ood_ious.append(iou)
                continue
            m = re_best_ood.search(line)
            if m:
                val = float(m.group(1))
                if val > best_ood_val and len(cur_id_ious) == 10 and len(cur_ood_ious) == 10:
                    best_ood_val = val
                    best_id_f1  = iou_to_f1(np.array(cur_id_ious))
                    best_ood_f1 = iou_to_f1(np.array(cur_ood_ious))

    return best_id_f1, best_ood_f1


def load_test_f1(eval_dir):
    path = os.path.join(ROOT_EVAL, eval_dir, "metrics.json")
    with open(path) as f:
        m = json.load(f)
    return m["overall"]["macro_f1"]


# ---------------------------------------------------------------------------
# Gather data
# ---------------------------------------------------------------------------
model_labels = list(MODELS.keys())
f1 = {}   # label → {split: float}

for label, (exp_dir, eval_dir) in MODELS.items():
    log = os.path.join(ROOT_EXP, exp_dir, "train.log")
    id_f1, ood_f1 = parse_best_ood_epoch(log)
    test_f1 = load_test_f1(eval_dir)
    f1[label] = {"val_id": id_f1, "val_ood": ood_f1, "test": test_f1}
    print(f"{label}: val_id F1={id_f1:.4f}  val_ood F1={ood_f1:.4f}  test F1={test_f1:.4f}")

# ---------------------------------------------------------------------------
# Plot — grouped by split, models side by side
# ---------------------------------------------------------------------------
splits   = ["val_id", "val_ood", "test"]
n_models = len(model_labels)
n_splits = len(splits)
width    = 0.18
x        = np.arange(n_splits)

PALETTE_MODEL = {
    "PTv3":         "#4C72B0",
    "PTv3+SINR":    "#DD8452",
    "PTv3+AE":      "#55A868",
    "PTv3+AE+SINR": "#C44E52",
}

fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle("Macro F1 by Split", fontsize=13, fontweight="bold")

for mi, model in enumerate(model_labels):
    vals   = [f1[model][split] for split in splits]
    offset = (mi - n_models / 2 + 0.5) * width
    bars   = ax.bar(x + offset, vals, width,
                    label=model,
                    color=PALETTE_MODEL[model],
                    edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7,
                rotation=90)

ax.set_xticks(x)
ax.set_xticklabels([split_labels[s] for s in splits], fontsize=11)
ax.set_ylim(0, 1.10)
ax.set_ylabel("Macro F1", fontsize=10)
ax.set_title("Macro F1", fontsize=11, fontweight="bold")
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE_MODEL[m]) for m in model_labels]
fig.legend(handles, model_labels,
           loc="lower center", ncol=n_models, fontsize=9,
           frameon=True, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.07, 1, 1])

out_path = os.path.join(OUT, "1_overall_metrics.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_path}")
