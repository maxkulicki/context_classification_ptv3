"""
Generate sample_name -> plot_id mapping for TreeScanPL.

Replays the same LAZ scanning logic as prepare_treescanpl.py and
generate_district_folds.py (same sorted file order, same species counters,
same filtering) and additionally records plot_id (parsed from LAZ filename).

Output: Pointcept/data/treescanpl/sample_plotid_mapping.csv

Usage:
    conda run -n lidar python generate_sample_plotid_mapping.py \
        --input_dir /home/makskulicki/data/TreeScanPL_2cm \
        --species_csv /home/makskulicki/tree_species_context_classification/species_id_names.csv \
        --output_dir /home/makskulicki/ptv3_cls/Pointcept/data/treescanpl
"""

import argparse
import csv
import os
import numpy as np
import laspy
from collections import defaultdict
from pathlib import Path


VALID_GENERA = {
    "Abies", "Acer", "Alnus", "Betula", "Carpinus",
    "Fagus", "Larix", "Picea", "Pinus", "Quercus", "Tilia",
}


def load_species_mapping(csv_path):
    """Load species CODE -> genus mapping from CSV."""
    mapping = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = int(row["CODE"])
            genus = row["LATIN_NAME"].split()[0]
            mapping[code] = genus
    return mapping


def extract_qualifying_tree_ids(laz_path, species_mapping):
    """Extract qualifying tree IDs and species from a LAZ file.

    Same filtering logic as prepare_treescanpl.py / generate_district_folds.py.
    Returns list of (tree_id, genus) tuples in sorted tree_id order.
    """
    las = laspy.read(laz_path)
    tree_ids = np.array(las.treeID)
    tree_sp = np.array(las.treeSP)
    completely_inside = np.array(las.completelyInside)

    unique_tree_ids = np.unique(tree_ids[tree_ids > 0])
    qualifying = []

    for tid in unique_tree_ids:
        mask = tree_ids == tid
        sp = tree_sp[mask][0]
        inside = completely_inside[mask][0]

        if sp <= 0 or inside != 1:
            continue
        if sp not in species_mapping:
            continue
        genus = species_mapping[sp]
        if genus not in VALID_GENERA:
            continue

        qualifying.append((int(tid), genus))

    return qualifying


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample_name -> plot_id mapping for TreeScanPL"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory with LAZ plot files",
    )
    parser.add_argument(
        "--species_csv",
        required=True,
        help="Path to species_id_names.csv",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory (Pointcept data/treescanpl)",
    )
    args = parser.parse_args()

    # 1. Load species mapping
    species_mapping = load_species_mapping(args.species_csv)
    print(f"Loaded {len(species_mapping)} species from CSV")

    # 2. Sort LAZ files exactly as prepare_treescanpl.py does
    laz_files = sorted(
        [f for f in os.listdir(args.input_dir) if f.endswith(".laz")]
    )
    print(f"Found {len(laz_files)} LAZ files")

    # 3. Replay the extraction logic to reconstruct sample -> plot_id mapping
    species_counters = defaultdict(int)
    sample_plotid = {}  # sample_name -> plot_id

    for fi, fname in enumerate(laz_files):
        stem = Path(fname).stem
        # Parse plot_id: Rem_{District}_{year}_{plotid_txt}
        parts = stem.split("_")
        plot_id = int(parts[3])
        laz_path = os.path.join(args.input_dir, fname)

        if (fi + 1) % 50 == 0 or fi == 0:
            print(f"  Scanning [{fi+1}/{len(laz_files)}] {fname} ...")

        qualifying = extract_qualifying_tree_ids(laz_path, species_mapping)

        for tid, genus in qualifying:
            species_counters[genus] += 1
            idx = species_counters[genus]
            sample_name = f"{genus}_{idx:04d}"
            sample_plotid[sample_name] = plot_id

    print(f"\nTotal samples mapped: {len(sample_plotid)}")
    print(f"Species counts: {dict(sorted(species_counters.items()))}")

    # 4. Verify against existing .npy files
    existing_samples = set()
    for genus_dir in sorted(os.listdir(args.output_dir)):
        genus_path = os.path.join(args.output_dir, genus_dir)
        if not os.path.isdir(genus_path) or genus_dir.startswith("treescanpl"):
            continue
        for f in os.listdir(genus_path):
            if f.endswith(".npy"):
                existing_samples.add(f.replace(".npy", ""))

    reconstructed = set(sample_plotid.keys())
    if reconstructed == existing_samples:
        print("All reconstructed samples match existing .npy files.")
    else:
        missing = existing_samples - reconstructed
        extra = reconstructed - existing_samples
        if missing:
            print(f"\nWARNING: {len(missing)} .npy files not matched:")
            for s in sorted(missing)[:20]:
                print(f"  {s}")
        if extra:
            print(f"\nWARNING: {len(extra)} reconstructed samples have no .npy:")
            for s in sorted(extra)[:20]:
                print(f"  {s}")
        # Use intersection
        valid = reconstructed & existing_samples
        print(f"Using intersection: {len(valid)} samples")
        sample_plotid = {k: v for k, v in sample_plotid.items() if k in valid}

    # 5. Write output CSV
    output_path = os.path.join(args.output_dir, "sample_plotid_mapping.csv")
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_name", "plot_id"])
        for name in sorted(sample_plotid.keys()):
            writer.writerow([name, sample_plotid[name]])

    print(f"\nMapping written to: {output_path}")
    print(f"Total entries: {len(sample_plotid)}")

    # 6. Summary
    unique_plots = set(sample_plotid.values())
    print(f"Unique plots: {len(unique_plots)}")


if __name__ == "__main__":
    main()
