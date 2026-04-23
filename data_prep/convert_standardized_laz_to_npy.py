"""Convert standardized_dataset LAZ files to NPY (coords-only) and rewrite splits."""

import argparse
import os
from pathlib import Path

import numpy as np
import laspy
import torch
import pointops


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_root", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--voxel_size", type=float, default=0.02)
    p.add_argument("--fps_points", type=int, default=8192)
    return p.parse_args()


def convert_file(in_path: Path, out_path: Path, voxel_size: float, fps_points: int):
    las = laspy.read(in_path)
    coord = np.stack([las.x, las.y, las.z], axis=-1).astype(np.float32)
    coord = voxel_downsample(coord, voxel_size)
    coord = fps_downsample(coord, fps_points)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, coord)





def voxel_downsample(coords, voxel_size):
    if voxel_size is None or voxel_size <= 0:
        return coords
    # integer voxel grid
    grid = (coords / voxel_size).astype(np.int64)
    # unique voxels, keep first point
    _, idx = np.unique(grid, axis=0, return_index=True)
    return coords[idx]









def fps_downsample(coords, fps_points):
    if fps_points is None or fps_points <= 0:
        return coords
    n = coords.shape[0]
    if n <= fps_points:
        return coords
    with torch.no_grad():
        pts = torch.from_numpy(coords).float().cuda(non_blocking=True)
        mask = pointops.farthest_point_sampling(
            pts,
            torch.tensor([n], device=pts.device, dtype=torch.long),
            torch.tensor([fps_points], device=pts.device, dtype=torch.long),
        )
        idx = mask.cpu().numpy()
    return coords[idx]


def rewrite_split(split_path: Path, input_root: Path, output_root: Path):
    if not split_path.exists():
        return
    rels = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    new_rels = []
    for rel in rels:
        rel = rel.replace("\\", "/")
        if rel.lower().endswith(".laz"):
            rel = rel[:-4] + ".npy"
        new_rels.append(rel)
    out_split = output_root / split_path.name
    out_split.write_text("\n".join(new_rels) + "\n")


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    class_dirs = [p for p in input_root.iterdir() if p.is_dir()]
    for class_dir in sorted(class_dirs):
        laz_files = sorted([p for p in class_dir.iterdir() if p.suffix.lower() == ".laz"])
        for in_path in laz_files:
            out_path = output_root / class_dir.name / (in_path.stem + ".npy")
            if out_path.exists() and not args.overwrite:
                continue
            convert_file(in_path, out_path, args.voxel_size, args.fps_points)
        print(f"{class_dir.name}: {len(laz_files)} files")

    # rewrite split files
    rewrite_split(input_root / "standardized_dataset_train.txt", input_root, output_root)
    rewrite_split(input_root / "standardized_dataset_val.txt", input_root, output_root)

    print(f"\nDone. NPY dataset at: {output_root}")


if __name__ == "__main__":
    main()
