#!/usr/bin/env python3
"""
Build a standardized dataset of individual tree point clouds.

- One LAZ file per tree, organized by species subfolders
- Filenames encode dataset and plot of origin
- All files voxelized to 2 cm resolution
- Absolute coordinates preserved

Usage (requires lidar conda env):
    conda run -n lidar python build_standardized_dataset.py
"""

import re
from pathlib import Path

import laspy
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VOXEL_SIZE = 0.02  # 2 cm

BASE = Path(__file__).parent
DATA = BASE / "data"
OTHER = DATA / "other datasets"
FORSPECIES = OTHER / "FORspecies_withGPS"
FORINSTANCE = OTHER / "FORinstance_dataset"

OUTPUT_DIR = DATA / "standardized_dataset"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def voxelize(x, y, z, voxel_size=VOXEL_SIZE):
    """Voxelize points: round to grid, keep one point per voxel."""
    xi = np.round(x / voxel_size).astype(np.int64)
    yi = np.round(y / voxel_size).astype(np.int64)
    zi = np.round(z / voxel_size).astype(np.int64)

    # Unique voxel indices
    coords = np.column_stack([xi, yi, zi])
    _, unique_idx = np.unique(coords, axis=0, return_index=True)

    return x[unique_idx], y[unique_idx], z[unique_idx]


def write_laz(filepath, x, y, z):
    """Write x, y, z arrays to a LAZ file preserving absolute coordinates."""
    if len(x) == 0:
        return

    header = laspy.LasHeader(point_format=0, version="1.2")
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = [np.floor(np.min(x)), np.floor(np.min(y)), np.floor(np.min(z))]

    las = laspy.LasData(header)
    las.x = x
    las.y = y
    las.z = z

    filepath.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(filepath))


def read_las_xyz(filepath):
    """Read a LAS/LAZ file and return x, y, z arrays."""
    f = laspy.read(str(filepath))
    return np.array(f.x), np.array(f.y), np.array(f.z)


def read_txt_xyz(filepath):
    """Read a space-separated XYZ text file."""
    pts = np.loadtxt(str(filepath))
    return pts[:, 0], pts[:, 1], pts[:, 2]


def safe_species_folder(species_name):
    """Convert species name to folder-safe format with consistent capitalization."""
    # Capitalize genus, lowercase epithet (e.g. "pinus sylvestris" -> "Pinus_sylvestris")
    parts = species_name.strip().split()
    if len(parts) >= 2:
        parts[0] = parts[0].capitalize()
        parts[1] = parts[1].lower()
    elif len(parts) == 1:
        parts[0] = parts[0].capitalize()
    return "_".join(parts)


def save_tree(species, filename, x, y, z):
    """Voxelize and write a single tree."""
    x, y, z = voxelize(x, y, z)
    folder = OUTPUT_DIR / safe_species_folder(species)
    write_laz(folder / filename, x, y, z)


# ---------------------------------------------------------------------------
# Dataset processors
# ---------------------------------------------------------------------------

def process_treescanpl():
    """TreeScanPL: extract individual trees from plot LAZ files."""
    print("  TreeScanPL...")
    plot_dir = DATA / "TreeScanPL_downsapled_2cm_corrected"
    species_map_df = pd.read_csv(BASE / "species_id_names.csv")
    sp_dict = dict(zip(species_map_df["CODE"], species_map_df["LATIN_NAME"]))

    # Load tree features to get completelyInside filter
    trees_df = pd.read_csv(DATA / "tree_features" / "all_tree_features.csv", sep=";")
    trees_df = trees_df[(trees_df["tree_id"] > 0) & (trees_df["completelyInside"] == 1) & (trees_df["treeSP"] > 0)]
    valid_trees = set(zip(trees_df["plot_id"].astype(int), trees_df["tree_id"].astype(int)))

    count = 0
    for laz_file in sorted(plot_dir.glob("*.laz")):
        # Plot ID from filename (e.g., "Rem_Gorlice_2015_0101703.laz")
        # Extract the last numeric segment after the last underscore
        plot_id_str = laz_file.stem.rsplit("_", 1)[-1]
        plot_id_num = int(plot_id_str)

        f = laspy.read(str(laz_file))
        tree_ids = np.array(f.treeID)
        tree_sps = np.array(f.treeSP)
        x, y, z = np.array(f.x), np.array(f.y), np.array(f.z)

        unique_trees = np.unique(tree_ids)
        for tid in unique_trees:
            if tid <= 0:
                continue
            if (plot_id_num, int(tid)) not in valid_trees:
                continue

            mask = tree_ids == tid
            sp_code = int(tree_sps[mask][0])
            species = sp_dict.get(sp_code)
            if species is None:
                continue

            fname = f"TreeScanPL_plot{plot_id_num}_tree{tid}.laz"
            save_tree(species, fname, x[mask], y[mask], z[mask])
            count += 1

    print(f"    {count} trees")


def process_weiser():
    """Weiser: per-tree LAZ files, species from CSV."""
    print("  Weiser...")
    summary = pd.read_csv(OTHER / "weiser" / "tls_data" / "weiser_summary.csv")
    tls_dir = OTHER / "weiser" / "tls_data"

    count = 0
    for _, row in summary.iterrows():
        tree_id = row["tree_id"]
        species = row["species_latin"]
        location = row["location"]

        # Find the LAZ file — in location subfolder
        matches = list((tls_dir / location).glob(f"*{tree_id}*TLS-on.laz"))
        if not matches:
            # Try broader search
            matches = list(tls_dir.rglob(f"*{tree_id}*TLS*.laz"))
        if not matches:
            continue

        x, y, z = read_las_xyz(matches[0])
        fname = f"Weiser_{tree_id}.laz"
        save_tree(species, fname, x, y, z)
        count += 1

    print(f"    {count} trees")


def process_forinstance_subset(name, folder, tree_csv_name):
    """NIBIO / CULS: extract trees from plot-level LAS files."""
    print(f"  {name}...")
    base = FORINSTANCE / folder
    tree_df = pd.read_csv(base / tree_csv_name)
    tree_df = tree_df.dropna(subset=["treeSP"])

    species_lookup = {}
    for _, row in tree_df.iterrows():
        species_lookup[(int(row["plotID"]), int(row["treeID"]))] = row["species_latin"]

    count = 0
    for las_path in sorted(base.glob("plot_*_annotated.las")):
        plot_num = int(las_path.stem.split("_")[1])
        f = laspy.read(str(las_path))
        tree_ids = np.array(f.treeID)
        x, y, z = np.array(f.x), np.array(f.y), np.array(f.z)

        for tid in np.unique(tree_ids):
            if tid == 0:
                continue
            species = species_lookup.get((plot_num, int(tid)))
            if species is None:
                continue

            mask = tree_ids == tid
            fname = f"{name}_plot{plot_num}_tree{tid}.laz"
            save_tree(species, fname, x[mask], y[mask], z[mask])
            count += 1

    print(f"    {count} trees")


def process_forspecies_folder(dataset_name, folder_name, dataset_prefix):
    """ForSpecies-GPS datasets: one LAS/LAZ per tree in species subfolders."""
    print(f"  {dataset_name}...")
    base = FORSPECIES / folder_name
    count = 0

    for species_dir in sorted(base.iterdir()):
        if not species_dir.is_dir():
            continue
        species = species_dir.name.replace("_", " ")

        for pc_file in sorted(species_dir.iterdir()):
            if pc_file.suffix.lower() not in (".las", ".laz"):
                continue

            x, y, z = read_las_xyz(pc_file)
            # Try to extract plot info from filename
            stem = pc_file.stem
            fname = f"{dataset_prefix}_{stem}.laz"
            save_tree(species, fname, x, y, z)
            count += 1

    print(f"    {count} trees")


def process_wytham_woods():
    """Wytham Woods: TXT files, local coords."""
    print("  Wytham Woods...")
    base = FORSPECIES / "wytham_woods"
    count = 0

    for species_dir in sorted(base.iterdir()):
        if not species_dir.is_dir():
            continue
        species = species_dir.name.replace("_", " ")

        for txt_file in sorted(species_dir.glob("*.txt")):
            x, y, z = read_txt_xyz(txt_file)
            fname = f"WythamWoods_{txt_file.stem}.laz"
            save_tree(species, fname, x, y, z)
            count += 1

    print(f"    {count} trees")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Building standardized dataset (voxel size = {VOXEL_SIZE} m)...")
    print(f"Output: {OUTPUT_DIR}\n")

    process_treescanpl()
    process_weiser()
    process_forinstance_subset("NIBIO", "NIBIO", "tree_data_NIBIO.csv")
    process_forinstance_subset("CULS", "CULS", "tree_data_CULS.csv")
    process_forspecies_folder("Frey 2022", "Frey_2022", "Frey2022")
    process_forspecies_folder("Junttila/Yrttimaa", "junttila_yrttimaa", "Junttila")
    process_forspecies_folder("Puliti MLS", "Puliti_MLS", "PulitiMLS")
    process_forspecies_folder("Puliti ULS 2", "puliti_ULS_2", "PulitiULS2")
    process_forspecies_folder("Saarinen 2021", "saarinen2021", "Saarinen2021")
    process_wytham_woods()

    # Summary
    print("\nDone. Species folder summary:")
    total = 0
    for species_dir in sorted(OUTPUT_DIR.iterdir()):
        if species_dir.is_dir():
            n = len(list(species_dir.glob("*.laz")))
            total += n
            print(f"  {species_dir.name}: {n} trees")
    print(f"\n  Total: {total} trees")


if __name__ == "__main__":
    main()
