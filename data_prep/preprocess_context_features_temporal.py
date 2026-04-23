"""
Build context_features_temporal_aug.pth for multi-temporal AE augmentation.

For each tree stem, stores:
    ctx_ae      : (64,) float32  — canonical AE embedding (copied from context_features.pth)
    ctx_ae_pool : (T, 64) float32 — one row per available year in all-years CSV

At training time, CtxPoolSample randomly picks one year from ctx_ae_pool.
Val uses ctx_ae directly (canonical, no pool sampling).

If a stem has no year data (missing in all-years CSV), ctx_ae_pool falls back
to a single-row pool containing ctx_ae, so training still runs.

Inputs:
    data/snapshot_v1/features.csv             — stem (via laz_path), dataset, tree_id
    data/trees_alphaearth_all_years.csv       — dataset, tree_id, ae_{year}_A00..A63
    data/snapshot_v1/context_features.pth    — canonical ctx_ae per stem

Output:
    data/snapshot_v1/context_features_temporal_aug.pth

Usage:
    python preprocess_context_features_temporal.py
    python preprocess_context_features_temporal.py --all-years data/my_all_years.csv
"""

import argparse
import os
import numpy as np
import torch
import pandas as pd

SNAPSHOT_DIR = "data/snapshot_v1"
FEATURES_CSV = os.path.join(SNAPSHOT_DIR, "features.csv")
CANONICAL_PTH = os.path.join(SNAPSHOT_DIR, "context_features.pth")
OUT_PTH = os.path.join(SNAPSHOT_DIR, "context_features_temporal_aug.pth")
ALL_YEARS_CSV = "data/trees_alphaearth_all_years.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-years", default=ALL_YEARS_CSV,
                        help=f"Path to all-years CSV (default: {ALL_YEARS_CSV})")
    parser.add_argument("--features-csv", default=FEATURES_CSV)
    parser.add_argument("--canonical-pth", default=CANONICAL_PTH)
    parser.add_argument("--output", "-o", default=OUT_PTH)
    args = parser.parse_args()

    print("=" * 60)
    print("Building context_features_temporal_aug.pth")
    print("=" * 60)

    # --- Load canonical ctx_ae per stem ---
    print(f"\nLoading canonical features from {args.canonical_pth} ...")
    canonical = torch.load(args.canonical_pth, weights_only=False)
    print(f"  {len(canonical)} stems")

    # --- Load features.csv: stem → (dataset, tree_id) ---
    print(f"\nLoading {args.features_csv} ...")
    feat_df = pd.read_csv(args.features_csv, usecols=["dataset", "tree_id", "laz_path"])
    feat_df["stem"] = feat_df["laz_path"].apply(
        lambda p: os.path.splitext(os.path.basename(str(p)))[0]
    )
    # Normalise tree_id to string for safe joins
    feat_df["tree_id"] = feat_df["tree_id"].astype(str)
    print(f"  {len(feat_df)} trees, {feat_df['stem'].nunique()} unique stems")

    # --- Load all-years CSV ---
    print(f"\nLoading {args.all_years} ...")
    years_df = pd.read_csv(args.all_years)
    years_df["tree_id"] = years_df["tree_id"].astype(str)

    # Discover year columns: ae_{year}_A00 → year list
    import re
    year_set = sorted({
        int(m.group(1))
        for c in years_df.columns
        for m in [re.match(r"ae_(\d+)_A00", c)]
        if m
    })
    print(f"  {len(years_df)} rows, years found: {year_set}")

    AE_BANDS = [f"A{i:02d}" for i in range(64)]

    # Build per-(dataset, tree_id) pool: list of (64,) arrays, one per year
    print("\nBuilding per-tree year pools...")
    # merge on dataset + tree_id
    merged = feat_df.merge(years_df, on=["dataset", "tree_id"], how="left")
    print(f"  Merged: {len(merged)} rows")

    n_missing = 0
    n_fallback = 0
    lookup = {}

    for _, row in merged.iterrows():
        stem = row["stem"]
        if stem not in canonical:
            continue  # stem not in canonical pth (shouldn't happen)

        ctx_ae = canonical[stem]["ctx_ae"]  # (64,)

        # Collect year embeddings — skip years where first band is NaN
        year_vecs = []
        for year in year_set:
            cols = [f"ae_{year}_{b}" for b in AE_BANDS]
            if cols[0] not in row.index or pd.isna(row[cols[0]]):
                continue
            vec = row[cols].values.astype(np.float32)
            if np.isnan(vec).any():
                continue
            year_vecs.append(vec)

        if len(year_vecs) == 0:
            # No year data: pool = canonical repeated once
            pool = ctx_ae[np.newaxis]  # (1, 64)
            n_fallback += 1
        else:
            pool = np.stack(year_vecs, axis=0)  # (T, 64)

        lookup[stem] = {
            "ctx_ae": ctx_ae,
            "ctx_ae_pool": pool,
        }

    n_missing = len(canonical) - len(lookup)
    print(f"\nResults:")
    print(f"  Stems with pool      : {len(lookup)}")
    print(f"  Stems missing in CSV : {n_missing}  (not in canonical pth)")
    print(f"  Stems using fallback : {n_fallback}  (no year data → pool=canonical)")

    # Sanity: pool shape distribution
    pool_sizes = [v["ctx_ae_pool"].shape[0] for v in lookup.values()]
    from collections import Counter
    print(f"  Pool size distribution: {dict(Counter(pool_sizes))}")

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(lookup, args.output)
    print(f"\nSaved {len(lookup)} entries to {args.output}")

    # Sanity check
    stem, entry = next(iter(lookup.items()))
    print(f"\nSample stem: {stem}")
    for k, v in entry.items():
        print(f"  {k}: shape={v.shape}  dtype={v.dtype}")


if __name__ == "__main__":
    main()
