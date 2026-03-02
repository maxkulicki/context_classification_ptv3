"""
Generate district-level k-fold cross-validation splits for TreeScanPL.

Replays the same LAZ scanning logic as prepare_treescanpl.py (same file order,
same species counters, same filtering) to reconstruct the sample→district
mapping, then writes 6 fold split files — one fold per district held out.

Usage:
    conda run -n lidar python generate_district_folds.py \
        --input_dir /home/makskulicki/data/TreeScanPL_2cm \
        --species_csv /home/makskulicki/data/TreeScanPL_2cm/species_id_names.csv \
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
    """Load species CODE -> LATIN_NAME mapping from CSV."""
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

    Same filtering logic as prepare_treescanpl.py but without reading
    full coordinates or estimating normals — only reads treeID, treeSP,
    completelyInside to determine which trees qualify.

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
        description="Generate district-level k-fold splits for TreeScanPL"
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
    parser.add_argument(
        "--min_districts",
        type=int,
        default=2,
        help="Exclude species present in fewer than this many districts (default: 2)",
    )
    args = parser.parse_args()

    species_mapping = load_species_mapping(args.species_csv)
    print(f"Loaded {len(species_mapping)} species from CSV")

    # Sort LAZ files exactly as prepare_treescanpl.py does
    laz_files = sorted(
        [f for f in os.listdir(args.input_dir) if f.endswith(".laz")]
    )
    print(f"Found {len(laz_files)} LAZ files")

    # Replay the extraction logic to reconstruct sample→district mapping
    species_counters = defaultdict(int)
    sample_district = {}  # sample_name → district
    district_samples = defaultdict(list)  # district → [sample_name, ...]

    for fi, fname in enumerate(laz_files):
        plot_name = Path(fname).stem
        # Parse district: Rem_{District}_{year}_{plotid}.laz
        district = fname.split("_")[1]
        laz_path = os.path.join(args.input_dir, fname)

        if (fi + 1) % 50 == 0 or fi == 0:
            print(f"  Scanning [{fi+1}/{len(laz_files)}] {fname} ...")

        qualifying = extract_qualifying_tree_ids(laz_path, species_mapping)

        for tid, genus in qualifying:
            species_counters[genus] += 1
            idx = species_counters[genus]
            sample_name = f"{genus}_{idx:04d}"

            sample_district[sample_name] = district
            district_samples[district].append(sample_name)

    total_samples = len(sample_district)
    districts = sorted(district_samples.keys())
    print(f"\nTotal samples reconstructed: {total_samples}")
    print(f"Districts ({len(districts)}): {', '.join(districts)}")

    # Filter species by minimum district presence
    genus_districts = defaultdict(set)  # genus → set of districts it appears in
    for sample_name, district in sample_district.items():
        genus = "_".join(sample_name.split("_")[:-1])
        genus_districts[genus].add(district)

    excluded_genera = set()
    for genus, dists in sorted(genus_districts.items()):
        if len(dists) < args.min_districts:
            excluded_genera.add(genus)
            count = sum(
                1 for s in sample_district
                if "_".join(s.split("_")[:-1]) == genus
            )
            print(
                f"  EXCLUDING {genus}: only in {len(dists)} district(s) "
                f"({', '.join(sorted(dists))}), {count} samples removed"
            )

    if excluded_genera:
        # Remove excluded samples from all mappings
        excluded_samples = {
            s for s in sample_district
            if "_".join(s.split("_")[:-1]) in excluded_genera
        }
        for s in excluded_samples:
            d = sample_district.pop(s)
            district_samples[d].remove(s)
        print(
            f"  After filtering: {len(sample_district)} samples "
            f"({len(excluded_samples)} removed)"
        )
        kept_genera = sorted(
            g for g in genus_districts if g not in excluded_genera
        )
        print(f"  Kept genera ({len(kept_genera)}): {', '.join(kept_genera)}")
    else:
        print(f"  All genera present in >= {args.min_districts} districts, none excluded.")

    # Verify reconstructed samples match existing .npy files
    existing_samples = set()
    for genus_dir in sorted(os.listdir(args.output_dir)):
        genus_path = os.path.join(args.output_dir, genus_dir)
        if not os.path.isdir(genus_path) or genus_dir.startswith("treescanpl"):
            continue
        for f in os.listdir(genus_path):
            if f.endswith(".npy"):
                existing_samples.add(f.replace(".npy", ""))

    reconstructed = set(sample_district.keys())
    if reconstructed != existing_samples:
        missing = existing_samples - reconstructed
        extra = reconstructed - existing_samples
        if missing:
            print(f"\nWARNING: {len(missing)} .npy files not matched by reconstruction:")
            for s in sorted(missing)[:10]:
                print(f"  {s}")
        if extra:
            print(f"\nWARNING: {len(extra)} reconstructed samples have no .npy file:")
            for s in sorted(extra)[:10]:
                print(f"  {s}")
        print("Proceeding with intersection of reconstructed and existing samples.")
        valid_samples = reconstructed & existing_samples
    else:
        print("All reconstructed samples match existing .npy files.")
        valid_samples = reconstructed

    # Generate fold splits
    print(f"\n{'='*70}")
    print(f"{'Fold':<6} {'Held-out District':<15} {'#Train':<8} {'#Test':<8}")
    print(f"{'='*70}")

    info_lines = []

    for k, held_out in enumerate(districts):
        test_samples = sorted(
            s for s in district_samples[held_out] if s in valid_samples
        )
        train_samples = sorted(
            s for d in districts if d != held_out
            for s in district_samples[d] if s in valid_samples
        )

        train_path = os.path.join(args.output_dir, f"treescanpl_fold{k}_train.txt")
        test_path = os.path.join(args.output_dir, f"treescanpl_fold{k}_test.txt")

        with open(train_path, "w") as f:
            f.write("\n".join(train_samples))
        with open(test_path, "w") as f:
            f.write("\n".join(test_samples))

        print(f"{k:<6} {held_out:<15} {len(train_samples):<8} {len(test_samples):<8}")
        info_lines.append(
            f"Fold {k}: hold out {held_out} "
            f"(train={len(train_samples)}, test={len(test_samples)})"
        )

    # Write info file
    info_path = os.path.join(args.output_dir, "treescanpl_folds_info.txt")
    with open(info_path, "w") as f:
        f.write("District-level 6-fold cross-validation splits\n")
        f.write(f"Total samples: {len(valid_samples)}\n")
        f.write(f"Districts: {', '.join(districts)}\n\n")
        for line in info_lines:
            f.write(line + "\n")
        f.write("\nPer-district species distribution:\n")
        for d in districts:
            d_samples = [s for s in district_samples[d] if s in valid_samples]
            sp_counts = defaultdict(int)
            for s in d_samples:
                genus = "_".join(s.split("_")[:-1])
                sp_counts[genus] += 1
            f.write(f"  {d}: {dict(sorted(sp_counts.items()))}\n")

    print(f"\nFold info written to: {info_path}")
    print(f"Split files written to: {args.output_dir}/treescanpl_fold{{0-5}}_{{train,test}}.txt")

    # Per-district species breakdown
    print(f"\nPer-district species distribution:")
    all_genera = sorted(g for g in VALID_GENERA if g not in excluded_genera)
    header = f"{'District':<12}" + "".join(f"{g:<9}" for g in all_genera) + "Total"
    print(header)
    print("-" * len(header))
    for d in districts:
        d_samples = [s for s in district_samples[d] if s in valid_samples]
        sp_counts = defaultdict(int)
        for s in d_samples:
            genus = "_".join(s.split("_")[:-1])
            sp_counts[genus] += 1
        row = f"{d:<12}" + "".join(f"{sp_counts.get(g, 0):<9}" for g in all_genera)
        row += str(len(d_samples))
        print(row)


if __name__ == "__main__":
    main()
