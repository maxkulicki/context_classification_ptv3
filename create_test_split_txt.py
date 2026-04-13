"""
Generate standardized_dataset_test.txt for the snapshot_10class_dual_val dataset.

Reads split.csv, filters rows with split=='test' and genus in the 10-class set,
checks that the corresponding .npy file exists, and writes the txt file.

Trees whose .npy file does not exist (Weiser, Wytham Woods, Frey 2022 — LAZ-only)
are skipped with a warning. Convert those LAZ files to .npy first if needed.

Usage (from context_classification_ptv3/):
    conda run -n context_baseline python create_test_split_txt.py

Output:
    /net/pr2/projects/plgrid/plggtreeseg/data/snapshot_10class_dual_val_npy/
        standardized_dataset_test.txt
"""

import os
import csv

SPLIT_CSV  = "data/snapshot_v1/split.csv"
DATA_ROOT  = "/net/pr2/projects/plgrid/plggtreeseg/data/snapshot_10class_dual_val_npy"
OUT_TXT    = os.path.join(DATA_ROOT, "standardized_dataset_test.txt")

CLASS_NAMES = {
    "Abies", "Acer", "Alnus", "Betula", "Carpinus",
    "Fagus", "Larix", "Picea", "Pinus", "Quercus",
}

rows = []
with open(SPLIT_CSV, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

# snapshot_v1 marks NIBIO as "val" (old holdout convention), but in
# snapshot_10class_dual_val NIBIO is the designated test set — include all of it.
test_rows = [r for r in rows if r["split"] == "test" or r["dataset"] == "NIBIO"]
print(f"Total test rows in split.csv: {len(test_rows)}")

skipped_genus  = 0
skipped_no_npy = 0
written = []
missing_by_dataset = {}

for row in test_rows:
    genus = row["genus"]
    if genus not in CLASS_NAMES:
        skipped_genus += 1
        continue

    stem = os.path.splitext(os.path.basename(row["laz_path"]))[0]
    rel  = f"{genus}/{stem}.npy"
    full = os.path.join(DATA_ROOT, rel)

    if not os.path.isfile(full):
        skipped_no_npy += 1
        # Derive dataset name the same way the dataset does (first underscore-split token)
        ds_name = stem.split("_")[0]
        missing_by_dataset[ds_name] = missing_by_dataset.get(ds_name, 0) + 1
        continue

    written.append(rel)

with open(OUT_TXT, "w") as f:
    for line in written:
        f.write(line + "\n")

print(f"\nWrote {len(written)} entries to {OUT_TXT}")
if skipped_genus:
    print(f"Skipped {skipped_genus} trees with genus outside the 10-class set")
if skipped_no_npy:
    print(f"Skipped {skipped_no_npy} trees with no .npy file:")
    for ds, n in sorted(missing_by_dataset.items()):
        print(f"  {ds}: {n} trees (LAZ-only — convert to npy to include)")
