#!/usr/bin/env python3
"""
Fetch AlphaEarth embeddings for spatially shifted locations across multiple years.

For every unique location in TREES_CSV, generates n_shifts spatially shifted
samples, each queried across n_years randomly chosen years.  This produces a
combined spatial+temporal augmentation pool.

Default parameters:
    n_shifts : 10  (random bearing, distance drawn uniformly from distances_m)
    n_years  : 5   (random subset of available years, without replacement)
    distances: [500, 1000, 2000] metres

Output (long format, one row per shift×year):
    original_loc_key | original_lat | original_lon
    | aug_lat | aug_lon | distance_m | shift_idx | year
    | A00 ... A63

Total rows ≈ n_unique_locations × n_shifts × n_years

Usage:
    python fetch_alphaearth_embeddings_combined.py
    python fetch_alphaearth_embeddings_combined.py --n-shifts 10 --n-years 5
    python fetch_alphaearth_embeddings_combined.py --project my-gcp-project
    python fetch_alphaearth_embeddings_combined.py --output data/trees_ae_combined_aug.csv
"""

import argparse
import math
import random
from pathlib import Path

import ee
import numpy as np
import pandas as pd

# =============================================================================
# PATHS & CONSTANTS
# =============================================================================

DATA_DIR = Path(__file__).parent / "data"
TREES_CSV = DATA_DIR / "all_trees_unified.csv"

COLLECTION_ID = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
EMBEDDING_BANDS = [f"A{i:02d}" for i in range(64)]

EE_BATCH_SIZE = 1000
METRES_PER_DEG_LAT = 111_320.0


# =============================================================================
# HELPERS
# =============================================================================

def build_loc_key(lat: float, lon: float) -> str:
    return f"{lat:.6f},{lon:.6f}"


def offset_point(lat: float, lon: float, distance_m: float, bearing_rad: float):
    d_lat = (distance_m * math.cos(bearing_rad)) / METRES_PER_DEG_LAT
    d_lon = (distance_m * math.sin(bearing_rad)) / (
        METRES_PER_DEG_LAT * math.cos(math.radians(lat))
    )
    return lat + d_lat, lon + d_lon


def discover_years() -> list[int]:
    collection = ee.ImageCollection(COLLECTION_ID)
    dates = collection.aggregate_array("system:time_start").getInfo()
    return sorted({pd.Timestamp(ms, unit="ms").year for ms in dates})


def make_ee_features(locs_df: pd.DataFrame) -> ee.FeatureCollection:
    features = []
    for _, row in locs_df.iterrows():
        geom = ee.Geometry.Point([float(row["lon"]), float(row["lat"])])
        features.append(ee.Feature(geom, {"loc_id": int(row["loc_id"])}))
    return ee.FeatureCollection(features)


def sample_embeddings_batch(locs_df: pd.DataFrame, image: ee.Image) -> pd.DataFrame:
    all_rows = []
    n_batches = math.ceil(len(locs_df) / EE_BATCH_SIZE)
    for batch_idx in range(n_batches):
        start = batch_idx * EE_BATCH_SIZE
        batch_df = locs_df.iloc[start: start + EE_BATCH_SIZE].copy()
        if n_batches > 1:
            print(f"      Batch {batch_idx + 1}/{n_batches} ({len(batch_df)} pts)...")
        fc = make_ee_features(batch_df)
        sampled = image.sampleRegions(collection=fc, scale=10, geometries=False)
        results = sampled.getInfo()
        for feat in results["features"]:
            all_rows.append(feat["properties"])
    if not all_rows:
        return pd.DataFrame(columns=["loc_id"] + EMBEDDING_BANDS)
    return pd.DataFrame(all_rows)


# =============================================================================
# AUGMENTATION GENERATION
# =============================================================================

def generate_shift_table(
    unique_locs: pd.DataFrame,
    years: list[int],
    distances_m: list[int],
    n_shifts: int,
    n_years: int,
    rng: random.Random,
) -> pd.DataFrame:
    """
    For each unique location, generate n_shifts spatial offsets × n_years years.
    Returns a DataFrame with one row per (location, shift, year):
        row_id | original_loc_key | original_lat | original_lon
        | aug_lat | aug_lon | distance_m | shift_idx | year
    row_id is used as loc_id in EE queries.
    """
    rows = []
    row_id = 0

    for _, loc in unique_locs.iterrows():
        # Sample n_shifts (bearing, distance) pairs
        shifts = []
        for _ in range(n_shifts):
            bearing = rng.uniform(0, 2 * math.pi)
            dist = rng.choice(distances_m)
            a_lat, a_lon = offset_point(loc["lat"], loc["lon"], dist, bearing)
            shifts.append((a_lat, a_lon, dist))

        # Sample n_years per shift (without replacement if possible)
        k = min(n_years, len(years))
        for shift_idx, (a_lat, a_lon, dist) in enumerate(shifts):
            chosen_years = rng.sample(years, k)
            for year in chosen_years:
                rows.append({
                    "row_id": row_id,
                    "original_loc_key": loc["_loc_key"],
                    "original_lat": loc["lat"],
                    "original_lon": loc["lon"],
                    "aug_lat": a_lat,
                    "aug_lon": a_lon,
                    "distance_m": dist,
                    "shift_idx": shift_idx,
                    "year": year,
                })
                row_id += 1

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fetch AlphaEarth combined spatial+temporal augmentations."
    )
    parser.add_argument("--n-shifts", type=int, default=10,
                        help="Spatial shifts per location (default: 10)")
    parser.add_argument("--n-years", type=int, default=5,
                        help="Years per shifted location (default: 5, without replacement)")
    parser.add_argument("--distances", type=int, nargs="+", default=[500, 1000, 2000],
                        help="Distances to sample from in metres (default: 500 1000 2000)")
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="Override available years (default: discover from collection)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--project", "-p", default=None)
    args = parser.parse_args()

    output_path = (
        Path(args.output) if args.output
        else DATA_DIR / "trees_alphaearth_combined_aug.csv"
    )
    rng = random.Random(args.seed)

    print("=" * 60)
    print("AlphaEarth — Combined Spatial+Temporal Augmentations")
    print("=" * 60)
    print(f"  Shifts per location : {args.n_shifts}")
    print(f"  Years per shift     : {args.n_years}")
    print(f"  Distances (m)       : {args.distances}")
    print(f"  Samples / location  : {args.n_shifts * args.n_years}")
    print(f"  Seed                : {args.seed}")

    # --- EE init ---
    print("\nInitializing Earth Engine...")
    ee.Authenticate()
    if args.project:
        ee.Initialize(project=args.project)
    else:
        ee.Initialize()
    print("  Done.")

    # --- Discover years ---
    if args.years:
        years = sorted(args.years)
        print(f"\nUsing specified years: {years}")
    else:
        print("\nDiscovering available years...")
        years = discover_years()
        print(f"  Found: {years}")

    if args.n_years > len(years):
        print(f"  WARNING: n_years={args.n_years} > available years={len(years)}, "
              f"will sample with replacement")

    # --- Load trees & deduplicate locations ---
    print(f"\nLoading {TREES_CSV} ...")
    trees = pd.read_csv(TREES_CSV)
    has_coords = trees["latitude"].notna() & trees["longitude"].notna()
    print(f"  {len(trees):,} trees, {has_coords.sum():,} with coordinates")

    trees["_loc_key"] = trees.apply(
        lambda r: build_loc_key(r["latitude"], r["longitude"])
        if pd.notna(r["latitude"]) and pd.notna(r["longitude"]) else None,
        axis=1,
    )
    valid = trees[has_coords]
    unique_locs = (
        valid[["_loc_key", "latitude", "longitude"]]
        .drop_duplicates("_loc_key")
        .reset_index(drop=True)
        .rename(columns={"latitude": "lat", "longitude": "lon"})
    )
    print(f"  {len(unique_locs):,} unique locations")

    # --- Generate shift×year table ---
    print("\nGenerating shift×year table...")
    shift_df = generate_shift_table(
        unique_locs, years, args.distances, args.n_shifts, args.n_years, rng
    )
    total = len(shift_df)
    print(f"  {total:,} rows  "
          f"({len(unique_locs):,} locs × {args.n_shifts} shifts × {args.n_years} years)")

    # --- Query EE year by year ---
    print(f"\nFetching embeddings (grouped by year)...")
    emb_records: dict[int, dict] = {}   # row_id → {A00: v, ...}

    for year in sorted(shift_df["year"].unique()):
        mask = shift_df["year"] == year
        year_locs = shift_df[mask][["row_id", "aug_lat", "aug_lon"]].copy()
        year_locs = year_locs.rename(columns={"aug_lat": "lat", "aug_lon": "lon",
                                               "row_id": "loc_id"})
        print(f"\n  Year {year}: {len(year_locs):,} samples...")
        collection = ee.ImageCollection(COLLECTION_ID)
        image = (
            collection
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .select(EMBEDDING_BANDS)
            .mosaic()
        )
        emb_df = sample_embeddings_batch(year_locs, image)
        n_missing = len(year_locs) - len(emb_df)
        if n_missing > 0:
            print(f"    WARNING: {n_missing} samples had no EE data (masked/ocean pixels)")
        for _, row in emb_df.iterrows():
            emb_records[int(row["loc_id"])] = {
                b: row.get(b, float("nan")) for b in EMBEDDING_BANDS
            }

    print(f"\n  Retrieved embeddings for {len(emb_records):,} / {total:,} rows")

    # --- Attach embeddings ---
    print("Assembling output...")
    for band in EMBEDDING_BANDS:
        shift_df[band] = shift_df["row_id"].map(
            lambda rid, b=band: emb_records.get(rid, {}).get(b, float("nan"))
        )

    meta_cols = [
        "row_id", "original_loc_key", "original_lat", "original_lon",
        "aug_lat", "aug_lon", "distance_m", "shift_idx", "year",
    ]
    shift_df = shift_df[meta_cols + EMBEDDING_BANDS]

    # --- Save ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shift_df.to_csv(output_path, index=False)

    n_with_emb = shift_df["A00"].notna().sum()
    print(f"\nOutput saved to {output_path}")
    print(f"  Rows total      : {len(shift_df):,}")
    print(f"  With embeddings : {n_with_emb:,}")
    print(f"  Missing         : {len(shift_df) - n_with_emb:,}")
    print(f"  Columns         : {len(shift_df.columns)} (9 meta + 64 embedding dims)")


if __name__ == "__main__":
    main()
