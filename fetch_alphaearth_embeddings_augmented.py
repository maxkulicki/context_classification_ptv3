#!/usr/bin/env python3
"""
Fetch AlphaEarth Satellite Embeddings with Distance-Based Spatial Augmentations.

For every unique location in TREES_CSV, generates 15 augmented samples:
    - 5 samples at  500 m in random directions, each using a random year
    - 5 samples at 1000 m in random directions, each using a random year
    - 5 samples at 2000 m in random directions, each using a random year

Output is one row per augmented sample (long format):
    original_loc_key | original_lat | original_lon
    | aug_lat | aug_lon | distance_m | year | aug_idx
    | A00 ... A63

Augmented locations are grouped by year before querying EE to minimise
the number of image loads. Within each year, locations are batched to stay
within EE memory limits.

Prerequisites:
    - earthengine-api installed
    - Authenticated via ee.Authenticate() (run once)
    - Google account registered for Earth Engine

Usage:
    python fetch_alphaearth_embeddings_augmented.py
    python fetch_alphaearth_embeddings_augmented.py --seed 42
    python fetch_alphaearth_embeddings_augmented.py --project my-gcp-project
    python fetch_alphaearth_embeddings_augmented.py --output data/trees_ae_augmented.csv
    python fetch_alphaearth_embeddings_augmented.py --years 2018 2019 2020
    python fetch_alphaearth_embeddings_augmented.py --n-augs 5 --distances 500 1000 2000
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

# Approximate metres per degree latitude (constant); longitude varies by lat.
METRES_PER_DEG_LAT = 111_320.0


# =============================================================================
# COORDINATE HELPERS
# =============================================================================

def offset_point(lat: float, lon: float, distance_m: float, bearing_rad: float):
    """
    Return (new_lat, new_lon) displaced from (lat, lon) by `distance_m` metres
    in the direction `bearing_rad` (radians, 0 = north, clockwise).

    Uses a flat-earth approximation — accurate to < 0.1% at 500 m.
    """
    d_lat = (distance_m * math.cos(bearing_rad)) / METRES_PER_DEG_LAT
    d_lon = (distance_m * math.sin(bearing_rad)) / (
        METRES_PER_DEG_LAT * math.cos(math.radians(lat))
    )
    return lat + d_lat, lon + d_lon


def build_location_key(lat: float, lon: float) -> str:
    return f"{lat:.6f},{lon:.6f}"


# =============================================================================
# EE HELPERS
# =============================================================================

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
    """
    Sample `image` at all locations in locs_df (must have loc_id, lat, lon).
    Returns DataFrame with loc_id + A00..A63.
    """
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
        return pd.DataFrame(columns=["aug_id"] + EMBEDDING_BANDS)
    return pd.DataFrame(all_rows)


# =============================================================================
# AUGMENTATION GENERATION
# =============================================================================

def generate_augmented_samples(
    unique_locs: pd.DataFrame,
    years: list[int],
    distances_m: list[int],
    n_augs: int,
    rng: random.Random,
) -> pd.DataFrame:
    """
    For each unique location, generate n_augs offset samples per distance.
    Returns a DataFrame with one row per augmented sample:
        aug_id | original_loc_key | original_lat | original_lon
        | aug_lat | aug_lon | distance_m | year | aug_idx
    aug_id is a global integer index used to match EE results back.
    """
    rows = []
    aug_id = 0

    for _, loc in unique_locs.iterrows():
        for dist in distances_m:
            for aug_idx in range(n_augs):
                bearing = rng.uniform(0, 2 * math.pi)
                year = rng.choice(years)
                a_lat, a_lon = offset_point(loc["lat"], loc["lon"], dist, bearing)
                rows.append({
                    "aug_id": aug_id,
                    "original_loc_key": loc["_loc_key"],
                    "original_lat": loc["lat"],
                    "original_lon": loc["lon"],
                    "aug_lat": a_lat,
                    "aug_lon": a_lon,
                    "distance_m": dist,
                    "year": year,
                    "aug_idx": aug_idx,
                })
                aug_id += 1

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fetch AlphaEarth embeddings with spatial augmentations."
    )
    parser.add_argument(
        "--years", type=int, nargs="+", default=None,
        help="Years to sample from (default: all years in collection)",
    )
    parser.add_argument(
        "--distances", type=int, nargs="+", default=[500, 1000, 2000],
        help="Offset distances in metres (default: 500 1000 2000)",
    )
    parser.add_argument(
        "--n-augs", type=int, default=5,
        help="Augmented samples per location per distance (default: 5)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output CSV path (default: data/trees_alphaearth_augmented.csv)",
    )
    parser.add_argument(
        "--project", "-p", default=None,
        help="Google Cloud project ID for Earth Engine initialization",
    )
    args = parser.parse_args()

    output_path = (
        Path(args.output) if args.output
        else DATA_DIR / "trees_alphaearth_augmented.csv"
    )
    rng = random.Random(args.seed)

    print("=" * 60)
    print("AlphaEarth — Distance-Based Spatial Augmentations")
    print("=" * 60)
    print(f"  Distances (m)   : {args.distances}")
    print(f"  Augs / distance : {args.n_augs}")
    print(f"  Total / location: {len(args.distances) * args.n_augs}")
    print(f"  Seed            : {args.seed}")

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

    # --- Load trees & deduplicate locations ---
    print(f"\nLoading {TREES_CSV} ...")
    trees = pd.read_csv(TREES_CSV)
    has_coords = trees["latitude"].notna() & trees["longitude"].notna()
    print(f"  {len(trees):,} trees, {has_coords.sum():,} with coordinates")

    trees["_loc_key"] = trees.apply(
        lambda r: build_location_key(r["latitude"], r["longitude"])
        if pd.notna(r["latitude"]) and pd.notna(r["longitude"])
        else None,
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

    # --- Generate augmented sample table ---
    print("\nGenerating augmented sample coordinates...")
    aug_df = generate_augmented_samples(
        unique_locs, years, args.distances, args.n_augs, rng
    )
    total_augs = len(aug_df)
    print(f"  {total_augs:,} augmented samples "
          f"({len(unique_locs):,} locs × {len(args.distances)} distances "
          f"× {args.n_augs} augs)")

    # --- Query EE year by year ---
    print(f"\nFetching embeddings (grouped by year)...")
    aug_df["lat"] = aug_df["aug_lat"]
    aug_df["lon"] = aug_df["aug_lon"]

    emb_records: dict[int, dict] = {}  # aug_id → {A00: v, ...}

    for year in years:
        year_mask = aug_df["year"] == year
        year_locs = aug_df[year_mask][["aug_id", "lat", "lon"]].copy()
        year_locs = year_locs.rename(columns={"aug_id": "aug_id"})  # keep name

        if len(year_locs) == 0:
            continue

        print(f"\n  Year {year}: {len(year_locs):,} samples to fetch...")
        collection = ee.ImageCollection(COLLECTION_ID)
        image = (
            collection
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .select(EMBEDDING_BANDS)
            .mosaic()
        )

        # aug_id used as loc_id in the batch sampler
        year_locs_ee = year_locs.rename(columns={"aug_id": "aug_id"}).copy()
        year_locs_ee["aug_id"] = year_locs_ee["aug_id"].astype(int)

        # reuse sample_embeddings_batch — it looks for 'aug_id' via make_ee_features
        # but make_ee_features expects 'loc_id'; pass aug_id as loc_id temporarily
        year_locs_ee = year_locs_ee.rename(columns={"aug_id": "loc_id"})
        emb_df = sample_embeddings_batch(year_locs_ee, image)
        # emb_df has 'loc_id' which is our aug_id
        emb_df = emb_df.rename(columns={"loc_id": "aug_id"})

        n_returned = len(emb_df)
        n_missing = len(year_locs) - n_returned
        if n_missing > 0:
            print(f"    WARNING: {n_missing} samples had no EE data (masked/ocean pixels)")

        for _, row in emb_df.iterrows():
            emb_records[int(row["aug_id"])] = {b: row.get(b, float("nan")) for b in EMBEDDING_BANDS}

    print(f"\n  Retrieved embeddings for {len(emb_records):,} / {total_augs:,} augmented samples")

    # --- Attach embeddings to aug_df ---
    print("Assembling output...")
    aug_df = aug_df.drop(columns=["lat", "lon"])  # drop temp cols

    for band in EMBEDDING_BANDS:
        aug_df[band] = aug_df["aug_id"].map(
            lambda aid, b=band: emb_records.get(aid, {}).get(b, float("nan"))
        )

    # Reorder columns for clarity
    meta_cols = [
        "aug_id", "original_loc_key", "original_lat", "original_lon",
        "aug_lat", "aug_lon", "distance_m", "year", "aug_idx",
    ]
    aug_df = aug_df[meta_cols + EMBEDDING_BANDS]

    # --- Save ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aug_df.to_csv(output_path, index=False)

    n_with_emb = aug_df["A00"].notna().sum()
    print(f"\nOutput saved to {output_path}")
    print(f"  Rows (augmented samples): {len(aug_df):,}")
    print(f"  With embeddings         : {n_with_emb:,}")
    print(f"  Without (masked/ocean)  : {len(aug_df) - n_with_emb:,}")
    print(f"  Columns                 : {len(aug_df.columns)} (9 meta + 64 embedding dims)")
    print(f"\nColumn layout:")
    print(f"  aug_id, original_loc_key, original_lat, original_lon,")
    print(f"  aug_lat, aug_lon, distance_m, year, aug_idx,")
    print(f"  A00 ... A63")


if __name__ == "__main__":
    main()
