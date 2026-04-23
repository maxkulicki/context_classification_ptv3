"""
FOR-species20K Dataset Preparation for PTv3 Classification

Converts individual tree LAZ files + metadata CSV into Pointcept
ModelNet40-compatible directory structure with .npy files.

Each LAZ file is one tree; species labels come from tree_metadata_dev.csv.

Usage:
    conda run -n lidar python prepare_forspecies20k.py \
        --input_dir /home/makskulicki/data/FORspecies20K/dev \
        --metadata_csv /home/makskulicki/data/FORspecies20K/tree_metadata_dev.csv \
        --output_dir /home/makskulicki/ptv3_cls/Pointcept/data/forspecies20k \
        --knn 30
"""

import argparse
import csv
import os
import time
import numpy as np
import laspy
import open3d as o3d
from collections import defaultdict
from pathlib import Path


def load_metadata(csv_path):
    """Load treeID -> species mapping from tree_metadata_dev.csv."""
    mapping = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tree_id = int(row["treeID"])
            species = row["species"].strip().replace(" ", "_")
            mapping[tree_id] = species
    return mapping


def estimate_normals(points, knn=30):
    """Estimate normals for a point cloud using Open3D.

    Uses camera-location orientation (point high above tree) instead of
    consistent_tangent_plane, which is near-quadratic and too slow for
    large trees (500K+ points).
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn)
    )
    # Orient normals outward/upward — trees are centered at origin so a
    # point far above gives physically sensible outward orientation.
    pcd.orient_normals_towards_camera_location(camera_location=[0, 0, 100])
    normals = np.asarray(pcd.normals).astype(np.float32)
    return normals


def save_tree_npy(path, coord, normal):
    """Save tree as binary .npy: float32 array of shape (N, 6) [x,y,z,nx,ny,nz]."""
    data = np.hstack([coord, normal]).astype(np.float32)
    np.save(path, data)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare FOR-species20K dataset for PTv3 classification"
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="Directory with individual tree LAZ/LAS files (dev/)",
    )
    parser.add_argument(
        "--metadata_csv", required=True,
        help="Path to tree_metadata_dev.csv",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory (Pointcept data/forspecies20k)",
    )
    parser.add_argument(
        "--knn", type=int, default=30,
        help="KNN neighbors for normal estimation (default: 30)",
    )
    parser.add_argument(
        "--min_points", type=int, default=100,
        help="Skip trees with fewer points than this (default: 100)",
    )
    parser.add_argument(
        "--max_points", type=int, default=100000,
        help="Randomly downsample trees above this point count (default: 100000)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process only first N files (0 = all, for testing)",
    )
    args = parser.parse_args()

    # Load metadata
    metadata = load_metadata(args.metadata_csv)
    print(f"Loaded metadata for {len(metadata)} trees")

    # Find all LAZ/LAS files
    laz_files = sorted(
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith((".laz", ".las"))
    )
    print(f"Found {len(laz_files)} LAZ/LAS files")

    if args.limit > 0:
        laz_files = laz_files[:args.limit]
        print(f"Limiting to first {args.limit} files")

    os.makedirs(args.output_dir, exist_ok=True)

    # Track per-species counters
    species_counters = defaultdict(int)
    all_samples = []
    skipped_no_metadata = 0
    skipped_too_few = 0
    total_processed = 0
    t_start = time.time()

    for fi, fname in enumerate(laz_files):
        # Derive treeID from filename (e.g., "00070.laz" -> 70)
        stem = Path(fname).stem
        try:
            tree_id = int(stem)
        except ValueError:
            print(f"  Warning: cannot parse treeID from '{fname}', skipping")
            continue

        if tree_id not in metadata:
            skipped_no_metadata += 1
            continue

        species_name = metadata[tree_id]

        # Load point cloud
        laz_path = os.path.join(args.input_dir, fname)
        las = laspy.read(laz_path)

        scale = las.header.scale
        x = np.array(las.X) * scale[0]
        y = np.array(las.Y) * scale[1]
        z = np.array(las.Z) * scale[2]
        coords = np.stack([x, y, z], axis=-1).astype(np.float32)

        # Skip tiny point clouds
        if len(coords) < args.min_points:
            skipped_too_few += 1
            continue

        # Downsample large trees
        if args.max_points > 0 and len(coords) > args.max_points:
            idx = np.random.choice(len(coords), args.max_points, replace=False)
            coords = coords[idx]

        # Center at origin
        centroid = coords.mean(axis=0)
        coords = coords - centroid

        # Estimate normals
        normals = estimate_normals(coords, knn=args.knn)

        # Create species directory and save
        sp_dir = os.path.join(args.output_dir, species_name)
        os.makedirs(sp_dir, exist_ok=True)

        species_counters[species_name] += 1
        idx = species_counters[species_name]
        sample_name = f"{species_name}_{idx:04d}"
        npy_path = os.path.join(sp_dir, f"{sample_name}.npy")

        save_tree_npy(npy_path, coords, normals)
        all_samples.append(sample_name)
        total_processed += 1

        elapsed = time.time() - t_start
        rate = total_processed / elapsed
        if total_processed % 50 == 0 or total_processed <= 5:
            remaining = (len(laz_files) - fi - 1) / rate if rate > 0 else 0
            print(
                f"  [{fi+1}/{len(laz_files)}] {sample_name} "
                f"({len(coords)} pts) — {rate:.1f} trees/s, "
                f"ETA {remaining/60:.0f} min"
            )

    elapsed = time.time() - t_start

    # Summary
    print(f"\n{'='*60}")
    print(f"Total trees processed: {total_processed}")
    print(f"Skipped (no metadata): {skipped_no_metadata}")
    print(f"Skipped (< {args.min_points} points): {skipped_too_few}")
    print(f"Time: {elapsed/60:.1f} min ({total_processed/elapsed:.1f} trees/s)")
    print(f"\nSpecies distribution ({len(species_counters)} species):")
    for sp_name in sorted(species_counters.keys()):
        print(f"  {sp_name}: {species_counters[sp_name]}")

    # Write train split file (full dev set, no val split)
    train_path = os.path.join(args.output_dir, "forspecies20k_train.txt")
    with open(train_path, "w") as f:
        f.write("\n".join(sorted(all_samples)))

    print(f"\nTrain samples: {len(all_samples)}")
    print(f"Split file: {train_path}")

    # Write class names file for config
    sorted_species = sorted(species_counters.keys())
    class_names_path = os.path.join(args.output_dir, "forspecies20k_class_names.txt")
    with open(class_names_path, "w") as f:
        f.write("\n".join(sorted_species))

    print(f"Class names ({len(sorted_species)} species): {class_names_path}")

    # Print Python snippet for config
    print(f"\n# Python class_names for config (copy-paste):")
    print("class_names = [")
    for sp in sorted_species:
        print(f'    "{sp}",')
    print("]")

    print(f"\nDataset ready at: {args.output_dir}")


if __name__ == "__main__":
    main()
