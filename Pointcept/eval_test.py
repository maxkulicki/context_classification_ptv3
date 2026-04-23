"""
Standalone test-set evaluator for trained PTv3 classification models.

Runs inference on the test split (standardized_dataset_test.txt) and produces:
  metrics.json         — overall + per-genus + per-dataset metrics
  predictions.csv      — per-tree: stem, dataset, true_genus, pred_genus, correct
  confusion_matrix.npy — (num_classes, num_classes) raw counts
  confusion_matrix.png — row-normalised heatmap

Usage (from context_classification_ptv3/Pointcept/, inside the apptainer):
    python eval_test.py \\
        --config      configs/standardized_dataset/cls-ptv3-ctx-sinr-gauss-10class-dual-val-4gpu.py \\
        --checkpoint  exp/snapshot_10class_dual_val/ptv3_ctx_sinr_gauss_4gpu_120ep/model/model_best.pth \\
        --output_dir  eval_results/ptv3_ctx_sinr_gauss

The script reuses the val dataset config (transforms, context sources, context_pth)
but switches split to 'test'. Run create_test_split_txt.py first to generate
standardized_dataset_test.txt in the data root.
"""

import sys
import os
import argparse
import json

import numpy as np
import torch
import torch.utils.data

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure the local Pointcept checkout is on the path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from pointcept.engines.defaults import default_config_parser
from pointcept.models import build_model
from pointcept.datasets import build_dataset, collate_fn as pt_collate_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_checkpoint(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    # Strip DDP "module." prefix if present
    weight = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(weight, strict=True)
    if missing:
        print(f"WARNING: missing keys in checkpoint: {missing}")
    if unexpected:
        print(f"WARNING: unexpected keys in checkpoint: {unexpected}")
    print(f"Loaded checkpoint: {ckpt_path}")


def plot_confusion_matrix(cm, class_names, out_path):
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n), max(6, n - 1)))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True genus",
        xlabel="Predicted genus",
        title="Confusion matrix (row-normalised)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    thresh = 0.5
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{cm_norm[i, j]:.2f}",
                ha="center", va="center", fontsize=7,
                color="white" if cm_norm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix plot saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PTv3 model on the test split"
    )
    parser.add_argument("--config",      required=True,
                        help="Training config .py file")
    parser.add_argument("--checkpoint",  required=True,
                        help="Model checkpoint .pth (e.g. model_best.pth)")
    parser.add_argument("--output_dir",  required=True,
                        help="Directory to write evaluation results")
    parser.add_argument("--split",       default="test",
                        help="Dataset split name (default: test)")
    parser.add_argument("--batch_size",  type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Config ---
    cfg = default_config_parser(args.config, {})
    class_names = list(cfg.data.names)
    num_classes = cfg.data.num_classes
    ignore_idx  = cfg.data.ignore_index

    # --- Model ---
    print("Building model ...")
    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint)
    model.to(device)
    model.eval()

    # --- Dataset ---
    # Inherit val config (transforms + context_pth + context_sources) but switch split.
    # Ensure source_id is collected so per-dataset breakdown works.
    val_cfg = dict(cfg.data.val)
    val_cfg["split"] = args.split
    for t in val_cfg.get("transform", []):
        if isinstance(t, dict) and t.get("type") == "Collect":
            keys = list(t.get("keys", ()))
            if "source_id" not in keys:
                keys.append("source_id")
            t["keys"] = tuple(keys)

    print(f"Building dataset (split='{args.split}') ...")
    dataset = build_dataset(val_cfg)
    loader  = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=pt_collate_fn,
    )
    source_names = dataset.source_names   # list[str], indexed by source_id
    n_sources    = len(source_names)
    print(f"  {len(dataset)} trees  |  {len(loader)} batches  |  {n_sources} datasets")
    print(f"  Datasets: {source_names}")

    # --- Inference ---
    confusion   = np.zeros((num_classes, num_classes), dtype=np.int64)
    ds_correct  = np.zeros(n_sources, dtype=np.int64)
    ds_total    = np.zeros(n_sources, dtype=np.int64)
    all_stems   = []
    all_preds   = []
    all_labels  = []
    all_src_ids = []
    sample_cursor = 0

    with torch.no_grad():
        for batch_idx, input_dict in enumerate(loader):
            batch_n = input_dict["category"].shape[0]

            for key, val in input_dict.items():
                if isinstance(val, torch.Tensor):
                    input_dict[key] = val.to(device, non_blocking=True)

            output_dict = model(input_dict)
            pred   = output_dict["cls_logits"].max(1)[1].cpu().numpy()
            label  = input_dict["category"].cpu().numpy()
            src_id = input_dict["source_id"].cpu().numpy().ravel()

            # Derive stems from the dataset's file list (no-shuffle loader → order preserved)
            for i in range(batch_n):
                idx = (sample_cursor + i) % len(dataset.data_list)
                stem = os.path.splitext(os.path.basename(dataset.data_list[idx]))[0]
                all_stems.append(stem)

            all_preds.extend(pred.tolist())
            all_labels.extend(label.tolist())
            all_src_ids.extend(src_id.tolist())

            valid = label != ignore_idx
            for p, l in zip(pred[valid], label[valid]):
                confusion[l, p] += 1
            for s, p, l, v in zip(src_id, pred, label, valid):
                if v and 0 <= s < n_sources:
                    ds_total[s]   += 1
                    ds_correct[s] += int(p == l)

            sample_cursor += batch_n
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(loader):
                print(f"  [{batch_idx + 1}/{len(loader)}]")

    # --- Compute metrics ---
    tp       = np.diag(confusion)
    fp       = confusion.sum(0) - tp
    fn       = confusion.sum(1) - tp
    support  = confusion.sum(1)

    acc_cls  = tp / (support + 1e-10)
    prec_cls = tp / (tp + fp + 1e-10)
    rec_cls  = tp / (tp + fn + 1e-10)
    f1_cls   = 2 * prec_cls * rec_cls / (prec_cls + rec_cls + 1e-10)

    all_acc     = float(tp.sum() / (confusion.sum() + 1e-10))
    present     = support > 0
    m_acc       = float(acc_cls[present].mean())
    macro_f1    = float(f1_cls[present].mean())
    weighted_f1 = float(np.average(f1_cls[present], weights=support[present]))

    per_genus = {
        name: {
            "acc":       float(acc_cls[i]),
            "precision": float(prec_cls[i]),
            "recall":    float(rec_cls[i]),
            "f1":        float(f1_cls[i]),
            "support":   int(support[i]),
        }
        for i, name in enumerate(class_names)
    }

    per_dataset = {
        source_names[i]: {
            "acc":     float(ds_correct[i] / ds_total[i]) if ds_total[i] > 0 else None,
            "correct": int(ds_correct[i]),
            "total":   int(ds_total[i]),
        }
        for i in range(n_sources)
        if ds_total[i] > 0
    }

    metrics = {
        "checkpoint": args.checkpoint,
        "config":     args.config,
        "split":      args.split,
        "n_trees":    int(sum(support)),
        "overall": {
            "all_acc":     all_acc,
            "m_acc":       m_acc,
            "macro_f1":    macro_f1,
            "weighted_f1": weighted_f1,
        },
        "per_genus":   per_genus,
        "per_dataset": per_dataset,
    }

    # --- Print summary ---
    print(f"\n{'=' * 58}")
    print(f"  Split: {args.split}  |  {int(sum(support))} trees evaluated")
    print(f"  allAcc={all_acc:.4f}  mAcc={m_acc:.4f}  "
          f"macroF1={macro_f1:.4f}  wF1={weighted_f1:.4f}")
    print(f"{'=' * 58}")
    print(f"  {'Genus':<14} {'Acc':>6} {'F1':>6} {'Prec':>6} {'Rec':>6} {'N':>5}")
    print(f"  {'-' * 44}")
    for name in class_names:
        g = per_genus[name]
        print(f"  {name:<14} {g['acc']:>6.3f} {g['f1']:>6.3f} "
              f"{g['precision']:>6.3f} {g['recall']:>6.3f} {g['support']:>5}")
    print(f"\n  {'Dataset':<30} {'Acc':>6} {'N':>5}")
    print(f"  {'-' * 42}")
    for ds_name, ds_m in sorted(per_dataset.items()):
        acc_str = f"{ds_m['acc']:.3f}" if ds_m["acc"] is not None else "  N/A"
        print(f"  {ds_name:<30} {acc_str:>6} {ds_m['total']:>5}")

    # --- Save outputs ---
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    csv_path = os.path.join(args.output_dir, "predictions.csv")
    with open(csv_path, "w") as f:
        f.write("stem,dataset,true_genus,pred_genus,correct\n")
        for stem, pred, label, sid in zip(all_stems, all_preds, all_labels, all_src_ids):
            if label == ignore_idx:
                continue
            true_g  = class_names[label] if 0 <= label < num_classes else "unknown"
            pred_g  = class_names[pred]  if 0 <= pred  < num_classes else "unknown"
            ds_name = source_names[sid]  if 0 <= sid   < n_sources   else "unknown"
            f.write(f"{stem},{ds_name},{true_g},{pred_g},{int(pred == label)}\n")
    print(f"Per-tree predictions saved to {csv_path}")

    cm_npy = os.path.join(args.output_dir, "confusion_matrix.npy")
    np.save(cm_npy, confusion)
    print(f"Confusion matrix saved to {cm_npy}")

    cm_png = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(confusion, class_names, cm_png)


if __name__ == "__main__":
    main()
