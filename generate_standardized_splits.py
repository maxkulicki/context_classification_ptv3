"""Generate train/val splits for standardized_dataset.

Writes:
  standardized_dataset_train.txt
  standardized_dataset_val.txt

Each line is a relative path like: "Abies_alba/Frey2022_B1T1.laz".
"""

import argparse
import os
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        default="/net/pr2/projects/plgrid/plggtreeseg/data/standardized_dataset",
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_samples", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir) if args.output_dir else data_root

    rng = random.Random(args.seed)

    class_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
    train_list = []
    val_list = []

    excluded = []
    for class_dir in class_dirs:
        laz_files = sorted(
            [p for p in class_dir.iterdir() if p.suffix.lower() == ".laz"]
        )
        n = len(laz_files)
        if n < args.min_samples:
            excluded.append((class_dir.name, n))
            continue

        rng.shuffle(laz_files)
        val_count = max(1, int(round(n * args.val_ratio)))
        if n - val_count < 1:
            val_count = n - 1

        val_files = laz_files[:val_count]
        train_files = laz_files[val_count:]

        for p in train_files:
            train_list.append(f"{class_dir.name}/{p.name}")
        for p in val_files:
            val_list.append(f"{class_dir.name}/{p.name}")

        print(f"{class_dir.name}: {n} total -> {len(train_files)} train, {len(val_files)} val")

    train_path = output_dir / "standardized_dataset_train.txt"
    val_path = output_dir / "standardized_dataset_val.txt"

    train_path.write_text("\n".join(train_list) + "\n")
    val_path.write_text("\n".join(val_list) + "\n")

    print("\nSplit files written:")
    print(f"  {train_path}")
    print(f"  {val_path}")

    if excluded:
        print("\nExcluded classes (count < min_samples):")
        for name, n in excluded:
            print(f"  {name}: {n}")


if __name__ == "__main__":
    main()
