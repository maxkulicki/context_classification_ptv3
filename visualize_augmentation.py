"""
Visualize point cloud samples before and after data augmentation.

Saves PLY files (CloudCompare-compatible) for N random samples from a
Pointcept classification dataset, using the transform pipeline defined
in a config file.

Usage:
    python visualize_augmentation.py \\
        --config Pointcept/configs/standardized_dataset/cls-ptv3-baseline-genus-small.py \\
        --output /tmp/aug_vis \\
        --n_samples 10 \\
        --split train \\
        --seed 42
"""

import argparse
import copy
import os
import random
import sys

import numpy as np

# Make Pointcept importable from the script's directory
_HERE = os.path.dirname(os.path.abspath(__file__))
_POINTCEPT = os.path.join(_HERE, "Pointcept")
if _POINTCEPT not in sys.path:
    sys.path.insert(0, _POINTCEPT)

from pointcept.utils.config import Config
from pointcept.datasets.standardized_dataset import StandardizedDataset
from pointcept.datasets.transform import Compose


# ---------------------------------------------------------------------------
# PLY export (no open3d required)
# ---------------------------------------------------------------------------

def save_ply(coord: np.ndarray, filepath: str, color: np.ndarray = None) -> None:
    """Write an ASCII PLY file with XYZ and optional uint8 RGB."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    coord = np.asarray(coord, dtype=np.float32)
    n = len(coord)
    has_color = color is not None

    with open(filepath, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")

        if has_color:
            color = np.asarray(color, dtype=np.uint8)
            for i in range(n):
                f.write(
                    f"{coord[i,0]:.6f} {coord[i,1]:.6f} {coord[i,2]:.6f}"
                    f" {color[i,0]} {color[i,1]} {color[i,2]}\n"
                )
        else:
            for i in range(n):
                f.write(f"{coord[i,0]:.6f} {coord[i,1]:.6f} {coord[i,2]:.6f}\n")


def height_colormap(z: np.ndarray) -> np.ndarray:
    """Map Z values to blue→green→red uint8 RGB (N, 3)."""
    z = np.asarray(z, dtype=float)
    z_min, z_max = z.min(), z.max()
    t = (z - z_min) / (z_max - z_min) if z_max > z_min else np.zeros_like(z)
    r = np.clip(2 * t - 1, 0, 1)
    g = np.clip(1 - 2 * abs(t - 0.5), 0, 1)
    b = np.clip(1 - 2 * t, 0, 1)
    return (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

# Types used only for batching — not augmentation, not needed for visualisation
_SKIP_TYPES = {"ToTensor", "Collect"}


def apply_before_transforms(data_dict: dict, transform_cfg: list) -> dict:
    """Apply only the centering step (CenterShiftMean) from the config."""
    pre = [t for t in transform_cfg if t["type"] == "CenterShiftMean"]
    return Compose(pre)(copy.deepcopy(data_dict))


def apply_after_transforms(data_dict: dict, transform_cfg: list) -> dict:
    """Apply all augmentation+voxelisation transforms, skip ToTensor/Collect."""
    aug = [t for t in transform_cfg if t["type"] not in _SKIP_TYPES]
    return Compose(aug)(copy.deepcopy(data_dict))


def to_numpy_coord(coord) -> np.ndarray:
    """Accept torch.Tensor or np.ndarray and return np.ndarray."""
    if hasattr(coord, "numpy"):
        coord = coord.detach().cpu().numpy()
    return np.asarray(coord, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Save before/after augmentation PLY files for N random samples."
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to a Pointcept Python config file (relative to CWD or absolute).",
    )
    parser.add_argument(
        "--output", default="aug_visualization",
        help="Directory where PLY files are saved.",
    )
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load config
    cfg = Config.fromfile(args.config)
    dataset_cfg = cfg.data[args.split]
    transform_cfg = dataset_cfg["transform"]
    class_names = dataset_cfg["class_names"]

    # Build dataset without any transforms — we apply them manually below
    dataset = StandardizedDataset(
        split=dataset_cfg["split"],
        data_root=dataset_cfg["data_root"],
        class_names=class_names,
        label_level=dataset_cfg.get("label_level", "genus"),
        transform=None,
        test_mode=False,
        loop=1,
    )

    n_total = len(dataset)
    indices = random.sample(range(n_total), min(args.n_samples, n_total))
    print(f"Dataset size: {n_total}  |  Sampling {len(indices)} items  |  Output: {args.output}")

    for i, idx in enumerate(indices):
        raw = dataset.get_data(idx)
        sample_name = dataset.get_data_name(idx).replace("/", "_").replace("\\", "_")
        cls_id = int(raw["category"][0])
        cls_name = class_names[cls_id]

        # --- Before: centering only ---
        before = apply_before_transforms(raw, transform_cfg)
        coord_before = to_numpy_coord(before["coord"])

        # --- After: full augmentation + voxelisation ---
        after = apply_after_transforms(raw, transform_cfg)
        coord_after = to_numpy_coord(after["coord"])

        out_before = os.path.join(args.output, f"sample_{i:02d}_{cls_name}_before.ply")
        out_after  = os.path.join(args.output, f"sample_{i:02d}_{cls_name}_after.ply")

        save_ply(coord_before, out_before, height_colormap(coord_before[:, 2]))
        save_ply(coord_after,  out_after,  height_colormap(coord_after[:, 2]))

        print(
            f"[{i+1:2d}/{len(indices)}] {cls_name:12s}  "
            f"before={len(coord_before):5d} pts  after={len(coord_after):5d} pts"
            f"\n         {out_before}"
            f"\n         {out_after}"
        )


if __name__ == "__main__":
    main()
