"""
Convert TreeScanPL dataset from .txt (CSV) to .npy (binary numpy).

Walks all genus subdirectories under data/treescanpl/, converts each .txt
point cloud file to .npy (float32), deletes the original .txt, and removes
stale .pth cache files so they get regenerated from the new format.

Usage:
    python convert_txt_to_npy.py [--data_root data/treescanpl]
"""

import argparse
import glob
import os
import numpy as np


def get_dir_size(path):
    """Return total size of all files under path in bytes."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


def human_size(nbytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.2f} PB"


def main():
    parser = argparse.ArgumentParser(description="Convert TreeScanPL .txt to .npy")
    parser.add_argument(
        "--data_root",
        default="data/treescanpl",
        help="Root directory of the TreeScanPL dataset",
    )
    args = parser.parse_args()
    data_root = args.data_root

    if not os.path.isdir(data_root):
        print(f"ERROR: {data_root} does not exist")
        return

    size_before = get_dir_size(data_root)
    print(f"Disk usage before: {human_size(size_before)}")

    # Find all .txt files in subdirectories (skip the split list files at root level)
    txt_files = []
    for dirpath, _, filenames in os.walk(data_root):
        # Skip the root directory (contains treescanpl_train.txt etc.)
        if os.path.normpath(dirpath) == os.path.normpath(data_root):
            continue
        for f in filenames:
            if f.endswith(".txt"):
                txt_files.append(os.path.join(dirpath, f))

    print(f"Found {len(txt_files)} .txt files to convert")

    converted = 0
    errors = 0
    for i, txt_path in enumerate(sorted(txt_files)):
        npy_path = txt_path[:-4] + ".npy"
        try:
            data = np.loadtxt(txt_path, delimiter=",").astype(np.float32)
            np.save(npy_path, data)
            # Verify the saved file
            check = np.load(npy_path)
            assert check.shape == data.shape, f"Shape mismatch: {check.shape} vs {data.shape}"
            os.remove(txt_path)
            converted += 1
        except Exception as e:
            print(f"  ERROR converting {txt_path}: {e}")
            errors += 1
            # Clean up partial .npy if it exists
            if os.path.exists(npy_path):
                os.remove(npy_path)

        if (i + 1) % 500 == 0 or (i + 1) == len(txt_files):
            print(f"  Progress: {i + 1}/{len(txt_files)} ({converted} converted, {errors} errors)")

    # Delete stale .pth cache files
    pth_pattern = os.path.join(data_root, "treescanpl_*.pth")
    pth_files = glob.glob(pth_pattern)
    for pth_path in pth_files:
        print(f"Removing stale cache: {pth_path}")
        os.remove(pth_path)

    size_after = get_dir_size(data_root)
    print(f"\nDone! Converted {converted} files ({errors} errors)")
    print(f"Removed {len(pth_files)} stale .pth cache files")
    print(f"Disk usage before: {human_size(size_before)}")
    print(f"Disk usage after:  {human_size(size_after)}")
    if size_before > 0:
        print(f"Reduction: {(1 - size_after / size_before) * 100:.1f}%")

    # Sanity check: load one .npy and verify shape
    npy_files = glob.glob(os.path.join(data_root, "**", "*.npy"), recursive=True)
    if npy_files:
        sample = np.load(npy_files[0])
        print(f"\nSanity check: {npy_files[0]}")
        print(f"  Shape: {sample.shape}, dtype: {sample.dtype}")
        print(f"  Expected: (N, 6) float32")
        assert sample.ndim == 2 and sample.shape[1] == 6, "Unexpected shape!"
        assert sample.dtype == np.float32, "Unexpected dtype!"
        print("  OK")


if __name__ == "__main__":
    main()
