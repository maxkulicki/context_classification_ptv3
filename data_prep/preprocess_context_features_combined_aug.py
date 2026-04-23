"""
Build context_features_combined_aug.pth from the combined spatial+temporal CSV.

For each tree stem, stores:
    ctx_ae      : (64,) float32   — canonical AE (copied from context_features.pth)
    ctx_ae_pool : (S*T, 64) float32 — n_shifts × n_years augmented samples

If a stem has no data in the combined CSV, falls back to pool = canonical (size 1).

Inputs:
    data/trees_alphaearth_combined_aug.csv     — output of fetch_alphaearth_embeddings_combined.py
    data/snapshot_v1/features.csv              — stem (via laz_path), latitude, longitude
    data/snapshot_v1/context_features.pth     — canonical ctx_ae per stem

Output:
    data/snapshot_v1/context_features_combined_aug.pth

Usage:
    python preprocess_context_features_combined_aug.py
"""

import argparse
import os
import numpy as np
import torch
import pandas as pd
from collections import Counter

SNAPSHOT_DIR = "data/snapshot_v1"
FEATURES_CSV = os.path.join(SNAPSHOT_DIR, "features.csv")
CANONICAL_PTH = os.path.join(SNAPSHOT_DIR, "context_features.pth")
COMBINED_CSV = "data/trees_alphaearth_combined_aug.csv"
OUT_PTH = os.path.join(SNAPSHOT_DIR, "context_features_combined_aug.pth")

AE_BANDS = [f"A{i:02d}" for i in range(64)]


def build_loc_key(lat: float, lon: float) -> str:
    return f"{lat:.6f},{lon:.6f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined-csv", default=COMBINED_CSV)
    parser.add_argument("--features-csv", default=FEATURES_CSV)
    parser.add_argument("--canonical-pth", default=CANONICAL_PTH)
    parser.add_argument("--output", "-o", default=OUT_PTH)
    args = parser.parse_args()

    print("=" * 60)
    print("Building context_features_combined_aug.pth")
    print("=" * 60)

    print(f"\nLoading canonical features from {args.canonical_pth} ...")
    canonical = torch.load(args.canonical_pth, weights_only=False)
    print(f"  {len(canonical)} stems")

    print(f"\nLoading {args.features_csv} ...")
    feat_df = pd.read_csv(args.features_csv, usecols=["laz_path", "latitude", "longitude"])
    feat_df["stem"] = feat_df["laz_path"].apply(
        lambda p: os.path.splitext(os.path.basename(str(p)))[0]
    )
    feat_df = feat_df.dropna(subset=["latitude", "longitude"])
    feat_df["loc_key"] = feat_df.apply(
        lambda r: build_loc_key(r["latitude"], r["longitude"]), axis=1
    )
    print(f"  {len(feat_df)} trees, {feat_df['loc_key'].nunique()} unique locations")

    print(f"\nLoading {args.combined_csv} ...")
    aug_df = pd.read_csv(args.combined_csv)
    aug_df = aug_df.dropna(subset=AE_BANDS)
    print(f"  {len(aug_df)} rows with valid embeddings")
    print(f"  Unique original locations: {aug_df['original_loc_key'].nunique()}")

    # Group by original_loc_key
    print("\nGrouping by location...")
    loc_to_pool: dict[str, np.ndarray] = {}
    for loc_key, grp in aug_df.groupby("original_loc_key"):
        vecs = grp[AE_BANDS].values.astype(np.float32)
        loc_to_pool[loc_key] = vecs  # (n_shifts*n_years, 64)

    print(f"  {len(loc_to_pool)} locations with data")

    # Assemble lookup
    print("\nAssembling per-stem lookup...")
    n_fallback = 0
    lookup = {}

    for _, row in feat_df.iterrows():
        stem = row["stem"]
        if stem not in canonical:
            continue

        ctx_ae = canonical[stem]["ctx_ae"]
        loc_key = row["loc_key"]

        if loc_key in loc_to_pool:
            pool = loc_to_pool[loc_key]
        else:
            pool = ctx_ae[np.newaxis]  # (1, 64) fallback
            n_fallback += 1

        lookup[stem] = {
            "ctx_ae": ctx_ae,
            "ctx_ae_pool": pool,
        }

    print(f"\nResults:")
    print(f"  Stems with pool      : {len(lookup)}")
    print(f"  Stems using fallback : {n_fallback}")
    pool_sizes = Counter(v["ctx_ae_pool"].shape[0] for v in lookup.values())
    print(f"  Pool size distribution: {dict(pool_sizes)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(lookup, args.output)
    print(f"\nSaved {len(lookup)} entries to {args.output}")

    stem, entry = next(iter(lookup.items()))
    print(f"\nSample stem: {stem}")
    for k, v in entry.items():
        print(f"  {k}: shape={v.shape}  dtype={v.dtype}")


if __name__ == "__main__":
    main()
