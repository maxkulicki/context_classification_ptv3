"""
TreeScanPL Dataset Preparation for PTv3 Classification

Extracts individual trees from plot-level LAZ files, computes normals,
and saves in ModelNet40-compatible format for Pointcept.

Usage:
    conda run -n lidar python prepare_treescanpl.py \
        --input_dir /home/makskulicki/data/TreeScanPL_2cm_partial \
        --output_dir /home/makskulicki/ptv3_cls/Pointcept/data/treescanpl \
        --species_csv /home/makskulicki/data/TreeScanPL_2cm_partial/species_id_names.csv \
        --test_ratio 0.2
"""

import argparse
import csv
import os
import numpy as np
import laspy
import open3d as o3d
from collections import defaultdict
from pathlib import Path


VALID_GENERA = {
    "Abies", "Acer", "Alnus", "Betula", "Carpinus",
    "Fagus", "Larix", "Picea", "Pinus", "Quercus", "Tilia",
}


def load_species_mapping(csv_path):
    """Load species CODE -> LATIN_NAME mapping from CSV."""
    mapping = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = int(row["CODE"])
            genus = row["LATIN_NAME"].split()[0]
            mapping[code] = genus
    return mapping


def estimate_normals(points, knn=30):
    """Estimate normals for a point cloud using Open3D."""
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


def extract_trees_from_laz(laz_path, species_mapping):
    """Extract individual qualifying trees from a LAZ plot file.

    Returns list of dicts: {species_name, coord, plot_name, tree_id}
    """
    las = laspy.read(laz_path)
    tree_ids = np.array(las.treeID)
    tree_sp = np.array(las.treeSP)
    completely_inside = np.array(las.completelyInside)

    scale = las.header.scale
    x = np.array(las.X) * scale[0]
    y = np.array(las.Y) * scale[1]
    z = np.array(las.Z) * scale[2]
    coords = np.stack([x, y, z], axis=-1).astype(np.float32)

    unique_tree_ids = np.unique(tree_ids[tree_ids > 0])
    trees = []

    for tid in unique_tree_ids:
        mask = tree_ids == tid
        sp = tree_sp[mask][0]
        inside = completely_inside[mask][0]

        if sp <= 0 or inside != 1:
            continue

        if sp not in species_mapping:
            print(f"  Warning: treeSP={sp} not in species CSV, skipping tree {tid}")
            continue

        genus = species_mapping[sp]
        if genus not in VALID_GENERA:
            continue

        tree_coords = coords[mask]

        # Center at origin
        centroid = tree_coords.mean(axis=0)
        tree_coords = tree_coords - centroid

        trees.append(
            dict(
                species_name=species_mapping[sp],
                species_code=sp,
                coord=tree_coords,
                tree_id=int(tid),
            )
        )

    return trees


def save_tree_npy(path, coord, normal):
    """Save tree as binary .npy: float32 array of shape (N, 6) [x,y,z,nx,ny,nz]."""
    data = np.hstack([coord, normal]).astype(np.float32)
    np.save(path, data)


def plot_level_split(plot_trees, test_ratio=0.2, seed=42):
    """Split plots into train/test sets, stratified by species distribution.

    Args:
        plot_trees: dict {plot_name: [(species_name, sample_name), ...]}
        test_ratio: fraction of plots for test
        seed: random seed

    Returns:
        train_samples: list of sample names
        test_samples: list of sample names
    """
    rng = np.random.RandomState(seed)
    plot_names = sorted(plot_trees.keys())

    # Compute per-plot species distribution for stratification
    # Simple approach: shuffle plots and greedily assign to test until ratio met
    rng.shuffle(plot_names)

    total_trees = sum(len(v) for v in plot_trees.values())
    target_test = int(total_trees * test_ratio)

    test_plots = set()
    test_count = 0
    for pname in plot_names:
        if test_count >= target_test:
            break
        test_plots.add(pname)
        test_count += len(plot_trees[pname])

    train_samples = []
    test_samples = []
    for pname in sorted(plot_trees.keys()):
        samples = [s[1] for s in plot_trees[pname]]
        if pname in test_plots:
            test_samples.extend(samples)
        else:
            train_samples.extend(samples)

    return sorted(train_samples), sorted(test_samples)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare TreeScanPL dataset for PTv3 classification"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory with LAZ plot files and species_id_names.csv",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory (Pointcept data/treescanpl)",
    )
    parser.add_argument(
        "--species_csv",
        required=True,
        help="Path to species_id_names.csv",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Fraction of plots for test set (default: 0.2)",
    )
    parser.add_argument(
        "--knn",
        type=int,
        default=30,
        help="KNN neighbors for normal estimation (default: 30)",
    )
    args = parser.parse_args()

    species_mapping = load_species_mapping(args.species_csv)
    print(f"Loaded {len(species_mapping)} species from CSV")

    laz_files = sorted(
        [f for f in os.listdir(args.input_dir) if f.endswith(".laz")]
    )
    print(f"Found {len(laz_files)} LAZ files")

    os.makedirs(args.output_dir, exist_ok=True)

    # Track per-species counters and per-plot tree lists
    species_counters = defaultdict(int)
    plot_trees = defaultdict(list)  # plot_name -> [(species_name, sample_name), ...]
    total_extracted = 0

    for fi, fname in enumerate(laz_files):
        plot_name = Path(fname).stem
        laz_path = os.path.join(args.input_dir, fname)
        print(f"\n[{fi+1}/{len(laz_files)}] Processing {fname}...")

        trees = extract_trees_from_laz(laz_path, species_mapping)
        print(f"  Extracted {len(trees)} qualifying trees")

        for tree in trees:
            sp_name = tree["species_name"]

            # Estimate normals
            normal = estimate_normals(tree["coord"], knn=args.knn)

            # Create species directory
            sp_dir = os.path.join(args.output_dir, sp_name)
            os.makedirs(sp_dir, exist_ok=True)

            # Name: SpeciesName_XXXX.txt
            species_counters[sp_name] += 1
            idx = species_counters[sp_name]
            sample_name = f"{sp_name}_{idx:04d}"
            npy_path = os.path.join(sp_dir, f"{sample_name}.npy")

            save_tree_npy(npy_path, tree["coord"], normal)
            plot_trees[plot_name].append((sp_name, sample_name))
            total_extracted += 1

            print(
                f"  Tree {tree['tree_id']} -> {sp_name} ({len(tree['coord'])} pts) -> {sample_name}.npy"
            )

    print(f"\n{'='*60}")
    print(f"Total trees extracted: {total_extracted}")
    print(f"Species distribution:")
    for sp_name in sorted(species_counters.keys()):
        print(f"  {sp_name}: {species_counters[sp_name]}")

    # Create train/test split
    train_samples, test_samples = plot_level_split(
        plot_trees, test_ratio=args.test_ratio
    )

    train_path = os.path.join(args.output_dir, "treescanpl_train.txt")
    test_path = os.path.join(args.output_dir, "treescanpl_test.txt")

    with open(train_path, "w") as f:
        f.write("\n".join(train_samples))
    with open(test_path, "w") as f:
        f.write("\n".join(test_samples))

    print(f"\nTrain samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")
    print(f"\nSplit files written to:")
    print(f"  {train_path}")
    print(f"  {test_path}")
    print(f"\nDataset ready at: {args.output_dir}")


if __name__ == "__main__":
    main()
