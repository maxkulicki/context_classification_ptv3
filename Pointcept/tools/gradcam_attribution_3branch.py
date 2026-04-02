"""
GradCAM Branch Attribution for MultiCatCtxCls-v1m2 (PTv3 + AE + SINR)

Runs over train, val_id and val_ood in a single invocation.
For each sample:
  1. Forward pass (eval mode, gradients enabled)
  2. Capture the 1024-d fused tensor (input to cls_head) via forward hook
  3. Backward from predicted class logit
  4. Attribution = ReLU(gradient * activation), summed per branch slice:
       PTv3:  dims [0 : backbone_embed_dim]                     default 0–511
       AE:    dims [backbone_embed_dim : backbone_embed_dim + context_embed_dim]  512–767
       SINR:  dims [backbone_embed_dim + context_embed_dim : end]                 768–1023
  5. Normalise to proportions summing to 1

Outputs one CSV per split to --out-dir, named {config_stem}_{split}_gradcam_3branch.csv
"""

import argparse
import csv
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointcept.utils.config import Config
from pointcept.models.builder import build_model
from pointcept.datasets.builder import build_dataset
from pointcept.datasets.utils import collate_fn

DEFAULT_CONFIG = (
    "configs/standardized_dataset/"
    "cls-ptv3-ctx-ae-sinr-cat-aux-10class-dual-val-4gpu.py"
)
DEFAULT_CHECKPOINT = (
    "exp/snapshot_10class_dual_val/"
    "ptv3_ctx_ae_sinr_cat_aux_4gpu_200ep/model/model_best.pth"
)
DEFAULT_OUT_DIR = (
    "/net/pr2/projects/plgrid/plggtreeseg/"
    "context_classification_ptv3/gradcam_analysis"
)


def load_model(cfg, checkpoint_path, device):
    cfg.model.backbone.enable_flash = False
    model = build_model(cfg.model)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned)
    model = model.to(device)
    model.eval()
    return model


def run_gradcam(model, data_dict, device, backbone_embed_dim, context_embed_dim):
    input_dict = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in data_dict.items()
    }

    captured = {}

    def hook_fn(module, inp, out):
        captured["fused"] = inp[0]  # (B, 1024)

    handle = model.cls_head.register_forward_hook(hook_fn)

    try:
        with torch.enable_grad():
            output = model(input_dict)
            logits = output["cls_logits"]  # (1, num_classes)

            fused = captured["fused"]
            fused.retain_grad()

            pred_label = logits.argmax(dim=-1).item()
            logits[0, pred_label].backward()

            grad = fused.grad[0]  # (1024,)
            act = fused[0]        # (1024,)

            attr = F.relu(grad * act)

            i0 = backbone_embed_dim
            i1 = backbone_embed_dim + context_embed_dim

            ptv3_attr = attr[:i0].sum().item()
            ae_attr   = attr[i0:i1].sum().item()
            sinr_attr = attr[i1:].sum().item()

            total = ptv3_attr + ae_attr + sinr_attr + 1e-8
            ptv3_prop = ptv3_attr / total
            ae_prop   = ae_attr   / total
            sinr_prop = sinr_attr / total

            probs = F.softmax(logits[0], dim=-1)
            confidence = probs.max().item()
            true_label = input_dict["category"].item()

    finally:
        handle.remove()

    return dict(
        true_label=true_label,
        pred_label=pred_label,
        correct=int(pred_label == true_label),
        ptv3_prop=ptv3_prop,
        ae_prop=ae_prop,
        sinr_prop=sinr_prop,
        confidence=confidence,
    )


FIELDNAMES = [
    "sample_name", "split",
    "true_label", "pred_label", "correct",
    "ptv3_prop", "ae_prop", "sinr_prop",
    "confidence",
]


def run_split(split, split_cfg, model, device, backbone_embed_dim, context_embed_dim, out_path):
    dataset = build_dataset(split_cfg)
    print(f"\n--- {split}  ({len(dataset)} samples) ---")

    results = []
    for i in range(len(dataset)):
        data = dataset[i]
        data_name = dataset.get_data_name(i)

        batch = collate_fn([data])
        model.zero_grad()

        attr = run_gradcam(model, batch, device, backbone_embed_dim, context_embed_dim)
        results.append(dict(sample_name=data_name, split=split, **attr))

        if (i + 1) % 100 == 0 or (i + 1) == len(dataset):
            print(
                f"  [{i+1}/{len(dataset)}] {data_name}  "
                f"ptv3={attr['ptv3_prop']:.3f}  ae={attr['ae_prop']:.3f}  "
                f"sinr={attr['sinr_prop']:.3f}  correct={attr['correct']}"
            )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)

    print(f"  Wrote {len(results)} rows → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="3-branch GradCAM attribution (all splits)")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help="Directory to write CSV files")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config.fromfile(args.config)

    backbone_embed_dim = cfg.model.get("backbone_embed_dim", 512)
    context_embed_dim  = cfg.model.get("context_embed_dim", 256)

    config_stem = os.path.splitext(os.path.basename(args.config))[0]
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Config:     {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Out dir:    {args.out_dir}")
    print(f"Prefix:     {config_stem}")
    print(f"Branch dims — PTv3: {backbone_embed_dim}, AE: {context_embed_dim}, SINR: {context_embed_dim}")

    model = load_model(cfg, args.checkpoint, device)

    # train split uses cfg.data.train; val splits share cfg.data.val with split override
    splits = [
        ("train",   cfg.data.train.copy()),
        ("val_id",  {**cfg.data.val.copy(), "split": "val_id"}),
        ("val_ood", {**cfg.data.val.copy(), "split": "val_ood"}),
    ]

    for split, split_cfg in splits:
        out_path = os.path.join(args.out_dir, f"{config_stem}_{split}_gradcam_3branch.csv")
        run_split(split, split_cfg, model, device, backbone_embed_dim, context_embed_dim, out_path)

    print("\nAll splits done.")


if __name__ == "__main__":
    main()
