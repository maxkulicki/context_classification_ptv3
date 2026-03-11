"""
Evaluate existing checkpoints with different point densities.
Compares model performance at 8192 vs 16384 input points.

Usage (from inside container):
    cd /workspace/Pointcept
    PYTHONPATH=. python tools/eval_point_density.py
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from pointcept.datasets import build_dataset
from pointcept.models import build_model
from pointcept.datasets.utils import collate_fn


CLASS_NAMES = [
    "Abies", "Acer", "Alnus", "Betula", "Carpinus",
    "Fagus", "Larix", "Picea", "Pinus", "Quercus", "Tilia",
]
NUM_CLASSES = 11

# Checkpoints to evaluate
CHECKPOINTS = {
    "PTv3 baseline (8k trained)": "exp/treescanpl/cls-ptv3-v1m1-0-base-finetune/model/model_best.pth",
    "PTv3+AE projected (8k trained)": "exp/treescanpl/cls-ptv3-v1m1-0-base-context-projected/model/model_best.pth",
}

# Model configs matching each checkpoint
MODEL_CONFIGS = {
    "PTv3 baseline (8k trained)": dict(
        type="DefaultClassifier",
        num_classes=NUM_CLASSES,
        backbone_embed_dim=512,
        backbone=dict(
            type="PT-v3m1",
            in_channels=6,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(32, 64, 128, 256, 512),
            enc_num_head=(2, 4, 8, 16, 32),
            enc_patch_size=(1024, 1024, 1024, 1024, 1024),
            dec_depths=(2, 2, 2, 2),
            dec_channels=(64, 64, 128, 256),
            dec_num_head=(4, 4, 8, 16),
            dec_patch_size=(1024, 1024, 1024, 1024),
            mlp_ratio=4, qkv_bias=True, qk_scale=None,
            attn_drop=0.0, proj_drop=0.0, drop_path=0.3,
            shuffle_orders=True, pre_norm=True,
            enable_rpe=False, enable_flash=True,
            upcast_attention=False, upcast_softmax=False,
            enc_mode=True,
            pdnorm_bn=False, pdnorm_ln=False,
            pdnorm_decouple=True, pdnorm_adaptive=False,
            pdnorm_affine=True,
            pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
        ),
        criteria=[],
    ),
    "PTv3+AE projected (8k trained)": dict(
        type="ContextClassifier",
        num_classes=NUM_CLASSES,
        backbone_embed_dim=512,
        context_dim=64,
        projection_dim=128,
        backbone=dict(
            type="PT-v3m1",
            in_channels=6,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(32, 64, 128, 256, 512),
            enc_num_head=(2, 4, 8, 16, 32),
            enc_patch_size=(1024, 1024, 1024, 1024, 1024),
            dec_depths=(2, 2, 2, 2),
            dec_channels=(64, 64, 128, 256),
            dec_num_head=(4, 4, 8, 16),
            dec_patch_size=(1024, 1024, 1024, 1024),
            mlp_ratio=4, qkv_bias=True, qk_scale=None,
            attn_drop=0.0, proj_drop=0.0, drop_path=0.3,
            shuffle_orders=True, pre_norm=True,
            enable_rpe=False, enable_flash=True,
            upcast_attention=False, upcast_softmax=False,
            enc_mode=True,
            pdnorm_bn=False, pdnorm_ln=False,
            pdnorm_decouple=True, pdnorm_adaptive=False,
            pdnorm_affine=True,
            pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
        ),
        criteria=[],
    ),
}

# Test transform (no augmentation, single pass)
TEST_TRANSFORM = [
    dict(type="NormalizeCoord"),
    dict(
        type="GridSample",
        grid_size=0.02,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
    ),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "category"),
        feat_keys=["coord", "normal"],
    ),
]

# Context version needs context key
TEST_TRANSFORM_CONTEXT = [
    dict(type="NormalizeCoord"),
    dict(
        type="GridSample",
        grid_size=0.02,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
    ),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "category", "context"),
        feat_keys=["coord", "normal"],
    ),
]

POINT_COUNTS = [8192, 16384]


def load_model(model_cfg, checkpoint_path):
    model = build_model(model_cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    # Strip module. prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.cuda().eval()
    return model


def evaluate(model, dataset, is_context_model=False):
    correct = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for idx in range(len(dataset)):
        data_dict = dataset[idx]
        category = data_dict.pop("category").item()

        for key in data_dict:
            if isinstance(data_dict[key], torch.Tensor):
                data_dict[key] = data_dict[key].cuda(non_blocking=True)

        # Add batch offset for single sample
        if "offset" not in data_dict:
            data_dict["offset"] = torch.tensor(
                [data_dict["coord"].shape[0]], dtype=torch.long
            ).cuda()

        with torch.no_grad():
            output = model(data_dict)
            logits = output["cls_logits"]
            pred = logits.argmax(dim=-1).item()

        if pred == category:
            correct += 1
            class_correct[category] += 1
        total += 1
        class_total[category] += 1

        if (idx + 1) % 200 == 0:
            print(f"  [{idx+1}/{len(dataset)}] running OA: {correct/total:.4f}")

    oa = correct / total
    per_class_acc = []
    for c in range(NUM_CLASSES):
        if class_total[c] > 0:
            acc = class_correct[c] / class_total[c]
        else:
            acc = 0.0
        per_class_acc.append(acc)

    macc = np.mean(per_class_acc)
    return oa, macc, per_class_acc


def main():
    results = {}

    for model_name, ckpt_path in CHECKPOINTS.items():
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}, skipping {model_name}")
            continue

        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"{'='*60}")

        model_cfg = MODEL_CONFIGS[model_name]
        model = load_model(model_cfg, ckpt_path)
        is_context = "Context" in model_cfg["type"]

        for num_points in POINT_COUNTS:
            print(f"\n--- Evaluating with {num_points} points ---")

            if is_context:
                dataset_cfg = dict(
                    type="TreeScanPLContextDataset",
                    split="test",
                    data_root="data/treescanpl",
                    class_names=CLASS_NAMES,
                    context_embeddings_path="plots_alphaearth_2018.csv",
                    sample_plotid_path="sample_plotid_mapping.csv",
                    num_points=num_points,
                    transform=TEST_TRANSFORM_CONTEXT,
                    test_mode=False,
                )
            else:
                dataset_cfg = dict(
                    type="TreeScanPLDataset",
                    split="test",
                    data_root="data/treescanpl",
                    class_names=CLASS_NAMES,
                    num_points=num_points,
                    transform=TEST_TRANSFORM,
                    test_mode=False,
                )

            dataset = build_dataset(dataset_cfg)
            oa, macc, per_class = evaluate(model, dataset, is_context)

            results[(model_name, num_points)] = {
                "OA": oa, "mAcc": macc, "per_class": per_class
            }

            print(f"\n  OA: {oa:.4f}  mAcc: {macc:.4f}")
            for i, name in enumerate(CLASS_NAMES):
                print(f"    {name:10s}: {per_class[i]:.4f}")

        del model
        torch.cuda.empty_cache()

    # Summary table
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    header = f"{'Model':<35s}"
    for np_ in POINT_COUNTS:
        header += f" | {np_:>5d}pt OA  {np_:>5d}pt mAcc"
    print(header)
    print("-" * len(header))

    for model_name in CHECKPOINTS:
        row = f"{model_name:<35s}"
        for np_ in POINT_COUNTS:
            key = (model_name, np_)
            if key in results:
                row += f" | {results[key]['OA']:>10.4f}  {results[key]['mAcc']:>11.4f}"
            else:
                row += f" | {'N/A':>10s}  {'N/A':>11s}"
        print(row)


if __name__ == "__main__":
    main()
