"""
GradCAM Feature Stream Attribution for ContextClassifier (Projected Fusion)

For each test sample across all 6 k-fold checkpoints:
1. Forward pass (eval mode, gradients enabled)
2. Capture the 256d fused tensor (input to cls_head) via forward hook
3. Backward from predicted class logit
4. Attribution = ReLU(gradient * activation), summed per stream:
   - PTv3 stream: dims 0–127
   - AE stream: dims 128–255
   - relative_ae = ae_attr / (ptv3_attr + ae_attr + eps)

Outputs a CSV with per-sample attributions.
"""

import argparse
import csv
import os
import sys

import torch
import torch.nn.functional as F

# Ensure Pointcept is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointcept.utils.config import Config
from pointcept.models.builder import build_model
from pointcept.datasets.builder import build_dataset
from pointcept.datasets.utils import collate_fn

DISTRICTS = ["Gorlice", "Herby", "Katrynka", "Milicz", "Piensk", "Suprasl"]
EXP_PATTERN = (
    "exp/treescanpl/"
    "cls-ptv3-v1m1-0-base-context-projected-kfold-fold{fold}-{district}/"
    "model/model_best.pth"
)


def load_model(cfg, checkpoint_path, device):
    """Build model from config and load checkpoint weights."""
    model = build_model(cfg.model)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Handle DDP checkpoint keys (module. prefix)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("module.", "")] = v
    model.load_state_dict(cleaned)
    model = model.to(device)
    model.eval()
    return model


def run_gradcam(model, data_dict, device, projection_dim=128):
    """Run forward + backward for one sample, return attribution dict."""
    # Move tensors to device
    input_dict = {}
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            input_dict[k] = v.to(device)
        else:
            input_dict[k] = v

    # Hook to capture fused tensor (input to cls_head)
    captured = {}

    def hook_fn(module, inp, out):
        # inp is a tuple; inp[0] is the fused tensor [B, 256]
        captured["fused"] = inp[0]

    handle = model.cls_head.register_forward_hook(hook_fn)

    try:
        # Forward pass with gradients
        with torch.enable_grad():
            output = model(input_dict)
            logits = output["cls_logits"]  # [1, num_classes]

            # Retain grad on the captured fused tensor
            fused = captured["fused"]
            fused.retain_grad()

            # Backward from predicted class logit
            pred_label = logits.argmax(dim=-1).item()
            target_score = logits[0, pred_label]
            target_score.backward()

            grad = fused.grad[0]  # [256]
            act = fused[0]  # [256]

            # GradCAM attribution: ReLU(grad * activation)
            attr = F.relu(grad * act)

            ptv3_attr = attr[:projection_dim].sum().item()
            ae_attr = attr[projection_dim:].sum().item()
            eps = 1e-8
            relative_ae = ae_attr / (ptv3_attr + ae_attr + eps)

            # Confidence
            probs = F.softmax(logits[0], dim=-1)
            confidence = probs.max().item()

            true_label = input_dict["category"].item()

    finally:
        handle.remove()

    return dict(
        true_label=true_label,
        pred_label=pred_label,
        correct=int(pred_label == true_label),
        ptv3_attr=ptv3_attr,
        ae_attr=ae_attr,
        relative_ae=relative_ae,
        confidence=confidence,
    )


def main():
    parser = argparse.ArgumentParser(description="GradCAM attribution analysis")
    parser.add_argument(
        "--config",
        default="configs/treescanpl/cls-ptv3-v1m1-0-base-context-projected-kfold.py",
        help="Path to config file",
    )
    parser.add_argument(
        "--output",
        default="gradcam_attributions.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config.fromfile(args.config)

    results = []

    for fold in range(6):
        district = DISTRICTS[fold]
        ckpt_path = EXP_PATTERN.format(fold=fold, district=district)

        if not os.path.exists(ckpt_path):
            print(f"WARNING: Checkpoint not found: {ckpt_path}, skipping fold {fold}")
            continue

        print(f"\n{'='*60}")
        print(f"Fold {fold}: {district} (checkpoint: {ckpt_path})")
        print(f"{'='*60}")

        # Build val dataset for this fold
        val_cfg = cfg.data.val.copy()
        val_cfg["fold"] = fold
        dataset = build_dataset(val_cfg)

        # Load model
        model = load_model(cfg, ckpt_path, device)

        num_samples = len(dataset)
        print(f"Processing {num_samples} test samples...")

        for i in range(num_samples):
            data = dataset[i]
            data_name = dataset.get_data_name(i)
            genus = data_name.rsplit("_", 1)[0]

            # Collate single sample into batch format
            batch = collate_fn([data])

            # Zero gradients
            model.zero_grad()

            attr = run_gradcam(model, batch, device)

            results.append(dict(
                sample_name=data_name,
                fold=fold,
                district=district,
                genus=genus,
                **attr,
            ))

            if (i + 1) % 100 == 0 or (i + 1) == num_samples:
                print(f"  [{i+1}/{num_samples}] last: {data_name} "
                      f"rel_ae={attr['relative_ae']:.3f} "
                      f"correct={attr['correct']}")

        # Free GPU memory before next fold
        del model
        torch.cuda.empty_cache()

    # Write CSV
    fieldnames = [
        "sample_name", "fold", "district", "genus",
        "true_label", "pred_label", "correct",
        "ptv3_attr", "ae_attr", "relative_ae", "confidence",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone! Wrote {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
