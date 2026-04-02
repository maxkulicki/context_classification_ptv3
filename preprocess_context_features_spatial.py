"""
Build context_features_spatial_aug.pth for spatial-shift AE augmentation.

For each tree stem, stores:
    ctx_ae      : (64,) float32  — canonical AE embedding (copied from context_features.pth)
    ctx_ae_pool : (K, 64) float32 — K spatially-shifted embeddings from augmented CSV

At training time, CtxPoolSample randomly picks one shifted sample from ctx_ae_pool.
Val uses ctx_ae directly (original location, no pool sampling).

If a stem has no augmented samples (missing location in augmented CSV), ctx_ae_pool
falls back to a single-row pool containing ctx_ae, so training still runs.

Inputs:
    data/snapshot_v1/features.csv             — stem (via laz_path), latitude, longitude
    data/trees_alphaearth_augmented.csv       — original_loc_key, A00..A63
    data/snapshot_v1/context_features.pth    — canonical ctx_ae per stem

Output:
    data/snapshot_v1/context_features_spatial_aug.pth

Usage:
    python preprocess_context_features_spatial.py
    python preprocess_context_features_spatial.py --augmented data/my_augmented.csv
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
OUT_PTH = os.path.join(SNAPSHOT_DIR, "context_features_spatial_aug.pth")
AUGMENTED_CSV = "data/trees_alphaearth_augmented.csv"

# Must match fetch_alphaearth_embeddings_augmented.py
METRES_PER_DEG_LAT = 111_320.0


def build_loc_key(lat: float, lon: float) -> str:
    """Round to 6 decimal places (~0.1 m) — matches fetch script."""
    return f"{lat:.6f},{lon:.6f}"


AE_BANDS = [f"A{i:02d}" for i in range(64)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--augmented", default=AUGMENTED_CSV,
                        help=f"Path to augmented CSV (default: {AUGMENTED_CSV})")
    parser.add_argument("--features-csv", default=FEATURES_CSV)
    parser.add_argument("--canonical-pth", default=CANONICAL_PTH)
    parser.add_argument("--output", "-o", default=OUT_PTH)
    args = parser.parse_args()

    print("=" * 60)
    print("Building context_features_spatial_aug.pth")
    print("=" * 60)

    # --- Load canonical ctx_ae per stem ---
    print(f"\nLoading canonical features from {args.canonical_pth} ...")
    canonical = torch.load(args.canonical_pth, weights_only=False)
    print(f"  {len(canonical)} stems")

    # --- Load features.csv: stem → loc_key ---
    print(f"\nLoading {args.features_csv} ...")
    feat_df = pd.read_csv(args.features_csv, usecols=["laz_path", "latitude", "longitude"])
    feat_df["stem"] = feat_df["laz_path"].apply(
        lambda p: os.path.splitext(os.path.basename(str(p)))[0]
    )
    feat_df = feat_df.dropna(subset=["latitude", "longitude"])
    feat_df["loc_key"] = feat_df.apply(
        lambda r: build_loc_key(r["latitude"], r["longitude"]), axis=1
    )
    print(f"  {len(feat_df)} trees with coordinates, {feat_df['loc_key'].nunique()} unique locations")

    # --- Load augmented CSV ---
    print(f"\nLoading {args.augmented} ...")
    aug_df = pd.read_csv(args.augmented)
    print(f"  {len(aug_df)} augmented samples")
    print(f"  Unique original locations: {aug_df['original_loc_key'].nunique()}")
    if "distance_m" in aug_df.columns:
        print(f"  Distance breakdown: {dict(Counter(aug_df['distance_m'].tolist()))}")

    # Drop rows with NaN in any AE band
    aug_df = aug_df.dropna(subset=AE_BANDS)
    print(f"  {len(aug_df)} samples after dropping NaN embeddings")

    # Group augmented samples by original_loc_key
    print("\nGrouping augmented samples by location...")
    loc_to_pool: dict[str, np.ndarray] = {}
    for loc_key, grp in aug_df.groupby("original_loc_key"):
        vecs = grp[AE_BANDS].values.astype(np.float32)
        loc_to_pool[loc_key] = vecs  # (K, 64)

    print(f"  {len(loc_to_pool)} locations with augmented data")

    # --- Build lookup ---
    print("\nAssembling per-stem lookup...")
    n_fallback = 0
    lookup = {}

    for _, row in feat_df.iterrows():
        stem = row["stem"]
        if stem not in canonical:
            continue

        ctx_ae = canonical[stem]["ctx_ae"]  # (64,)
        loc_key = row["loc_key"]

        if loc_key in loc_to_pool:
            pool = loc_to_pool[loc_key]  # (K, 64)
        else:
            pool = ctx_ae[np.newaxis]  # (1, 64) — fallback
            n_fallback += 1

        lookup[stem] = {
            "ctx_ae": ctx_ae,
            "ctx_ae_pool": pool,
        }

    n_missing = len(canonical) - len(lookup)
    print(f"\nResults:")
    print(f"  Stems with pool      : {len(lookup)}")
    print(f"  Stems missing in CSV : {n_missing}  (not in features.csv)")
    print(f"  Stems using fallback : {n_fallback}  (location not in augmented CSV)")

    pool_sizes = [v["ctx_ae_pool"].shape[0] for v in lookup.values()]
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
