"""
Compute per-tree geometry features from raw .npy point clouds.

For every tree in train / val_id / val_ood / test:
  - height      : z_max - z_min
  - crown_area  : 2D convex hull area of XY projection (m²)

Output: context_classification_ptv3/data/tree_geometry.csv
Columns: stem, split, genus, dataset, height, crown_area, n_pts, hull_degenerate

Run once; result is cached.  Re-run with --force to overwrite.

Usage:
    conda run -n context_baseline python compute_tree_geometry.py [--force]
"""

import argparse
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError

ROOT = Path('/net/pr2/projects/plgrid/plggtreeseg')
NPY_DIR = ROOT / 'data/snapshot_10class_dual_val_npy'
OUT_CSV = ROOT / 'context_classification_ptv3/data/tree_geometry.csv'

SPLITS = ['train', 'val_id', 'val_ood', 'test']


def load_split(split):
    """Return list of (stem, genus, dataset) for a split."""
    txt = NPY_DIR / f'standardized_dataset_{split}.txt'
    entries = []
    with open(txt) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            genus, fname = line.split('/', 1)
            stem = fname.replace('.npy', '')
            dataset = stem.split('_')[0]
            entries.append((stem, genus, dataset))
    return entries


def compute_geometry(args):
    stem, genus, dataset, split = args
    npy_path = NPY_DIR / genus / f'{stem}.npy'
    try:
        pts = np.load(npy_path)
    except Exception as e:
        return dict(stem=stem, split=split, genus=genus, dataset=dataset,
                    height=np.nan, crown_area=np.nan, n_pts=0,
                    hull_degenerate=True, error=str(e))

    n_pts = len(pts)
    height = float(pts[:, 2].max() - pts[:, 2].min())

    # 2D convex hull on XY projection
    xy = pts[:, :2]
    unique_xy = np.unique(xy, axis=0)
    hull_degenerate = False
    crown_area = 0.0

    if len(unique_xy) < 3:
        hull_degenerate = True
    else:
        try:
            hull = ConvexHull(unique_xy)
            crown_area = float(hull.volume)  # volume = area in 2D
        except QhullError:
            hull_degenerate = True

    return dict(stem=stem, split=split, genus=genus, dataset=dataset,
                height=height, crown_area=crown_area, n_pts=n_pts,
                hull_degenerate=hull_degenerate, error='')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='Overwrite existing output')
    parser.add_argument('--workers', type=int, default=min(16, cpu_count()),
                        help='Parallel workers')
    args = parser.parse_args()

    if OUT_CSV.exists() and not args.force:
        print(f'Output already exists: {OUT_CSV}  (use --force to recompute)')
        return

    all_tasks = []
    for split in SPLITS:
        entries = load_split(split)
        print(f'  {split}: {len(entries)} trees')
        for stem, genus, dataset in entries:
            all_tasks.append((stem, genus, dataset, split))

    print(f'\nComputing geometry for {len(all_tasks)} trees using {args.workers} workers...')

    with Pool(args.workers) as pool:
        results = pool.map(compute_geometry, all_tasks)

    df = pd.DataFrame(results)
    n_deg = df['hull_degenerate'].sum()
    n_err = (df['error'] != '').sum()
    print(f'Done. Degenerate hulls: {n_deg}, Errors: {n_err}')
    print(f'Height range: {df.height.min():.2f} – {df.height.max():.2f} m')
    print(f'Crown area range: {df.crown_area.min():.2f} – {df.crown_area.max():.2f} m²')

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f'Saved: {OUT_CSV}')


if __name__ == '__main__':
    main()
