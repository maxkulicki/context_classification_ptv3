"""
Reorganize TreeScanPL dataset from species-level to genus-level.

- Moves files from species dirs into genus dirs with new sequential naming
- Regenerates train/test split files, excluding genera with < min_samples
- Preserves the existing plot-level train/test assignment

Usage:
    conda run -n lidar python reorganize_to_genus.py \
        --data_dir /home/makskulicki/ptv3_cls/Pointcept/data/treescanpl \
        --min_samples 50
"""

import argparse
import os
import shutil
import numpy as np
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--min_samples", type=int, default=50)
    args = parser.parse_args()

    data_dir = args.data_dir

    # 1. Read existing split files to know which samples are train vs test
    train_path = os.path.join(data_dir, "treescanpl_train.txt")
    test_path = os.path.join(data_dir, "treescanpl_test.txt")

    train_samples = set(np.loadtxt(train_path, dtype="str").tolist())
    test_samples = set(np.loadtxt(test_path, dtype="str").tolist())
    print(f"Existing split: {len(train_samples)} train, {len(test_samples)} test")

    # 2. Scan all species directories and map samples to genus
    species_dirs = sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    )

    # sample_name -> (species_dir, genus, split)
    sample_info = {}
    genus_samples = defaultdict(list)

    for sp_dir in species_dirs:
        genus = sp_dir.split("_")[0]
        sp_path = os.path.join(data_dir, sp_dir)
        txt_files = sorted([f for f in os.listdir(sp_path) if f.endswith(".txt")])
        for fname in txt_files:
            sample_name = fname[:-4]  # remove .txt
            split = None
            if sample_name in train_samples:
                split = "train"
            elif sample_name in test_samples:
                split = "test"
            sample_info[sample_name] = dict(
                species_dir=sp_dir,
                genus=genus,
                split=split,
                old_path=os.path.join(sp_path, fname),
            )
            genus_samples[genus].append(sample_name)

    print(f"\nTotal samples found: {len(sample_info)}")
    print(f"Total genera: {len(genus_samples)}")

    # 3. Identify qualifying genera
    qualifying_genera = sorted(
        [g for g, samples in genus_samples.items() if len(samples) >= args.min_samples]
    )
    excluded_genera = sorted(
        [g for g, samples in genus_samples.items() if len(samples) < args.min_samples]
    )

    print(f"\nQualifying genera (>= {args.min_samples}): {len(qualifying_genera)}")
    for g in qualifying_genera:
        print(f"  {g}: {len(genus_samples[g])}")
    print(f"\nExcluded genera (< {args.min_samples}): {len(excluded_genera)}")
    for g in excluded_genera:
        print(f"  {g}: {len(genus_samples[g])}")

    # 4. Create genus directories, move and rename files
    old_to_new = {}  # old_sample_name -> new_sample_name

    for genus in sorted(genus_samples.keys()):
        genus_dir = os.path.join(data_dir, genus)
        os.makedirs(genus_dir, exist_ok=True)

        # Sort samples within genus for deterministic ordering
        samples = sorted(genus_samples[genus])
        for idx, old_name in enumerate(samples, start=1):
            info = sample_info[old_name]
            new_name = f"{genus}_{idx:04d}"
            new_path = os.path.join(genus_dir, f"{new_name}.txt")

            # Move file
            shutil.move(info["old_path"], new_path)
            old_to_new[old_name] = new_name
            info["new_name"] = new_name

    print(f"\nMoved {len(old_to_new)} files")

    # 5. Remove empty species directories
    removed_dirs = 0
    for sp_dir in species_dirs:
        sp_path = os.path.join(data_dir, sp_dir)
        if os.path.isdir(sp_path) and not os.listdir(sp_path):
            os.rmdir(sp_path)
            removed_dirs += 1
    print(f"Removed {removed_dirs} empty species directories")

    # 6. Regenerate split files (only qualifying genera)
    qualifying_set = set(qualifying_genera)
    new_train = []
    new_test = []

    for old_name, new_name in sorted(old_to_new.items(), key=lambda x: x[1]):
        info = sample_info[old_name]
        genus = info["genus"]

        if genus not in qualifying_set:
            continue

        if info["split"] == "train":
            new_train.append(new_name)
        elif info["split"] == "test":
            new_test.append(new_name)

    new_train.sort()
    new_test.sort()

    with open(train_path, "w") as f:
        f.write("\n".join(new_train))
    with open(test_path, "w") as f:
        f.write("\n".join(new_test))

    print(f"\nNew split files:")
    print(f"  Train: {len(new_train)} samples")
    print(f"  Test:  {len(new_test)} samples")

    # 7. Print genus distribution per split
    train_genus = defaultdict(int)
    test_genus = defaultdict(int)
    for name in new_train:
        train_genus[name.split("_")[0]] += 1
    for name in new_test:
        test_genus[name.split("_")[0]] += 1

    print(f"\n{'Genus':<15} {'Train':>6} {'Test':>6} {'Total':>6}")
    print("-" * 40)
    for g in qualifying_genera:
        total = train_genus[g] + test_genus[g]
        print(f"  {g:<13} {train_genus[g]:>6} {test_genus[g]:>6} {total:>6}")
    print(
        f"  {'TOTAL':<13} {sum(train_genus.values()):>6} {sum(test_genus.values()):>6} {sum(train_genus.values()) + sum(test_genus.values()):>6}"
    )

    # 8. Delete stale .pth cache files
    for f in os.listdir(data_dir):
        if f.endswith(".pth"):
            os.remove(os.path.join(data_dir, f))
            print(f"\nRemoved stale cache: {f}")

    print("\nDone! Update config class_names and num_classes to match.")


if __name__ == "__main__":
    main()
