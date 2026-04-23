#!/usr/bin/env python3
"""
Fetch AlphaEarth Satellite Embeddings for all trees with coordinates.

Queries the Google Earth Engine Satellite Embedding V1 Annual dataset
(GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL) for the 2018 annual image,
extracts the 64-dimensional embedding vector at each unique location,
then joins back to all trees in data/all_trees_unified.csv.

Trees without coordinates (NaN lat/lon) receive NaN embeddings.
Many trees share the same plot location — deduplication means EE is
queried once per unique coordinate pair, not once per tree.

Prerequisites:
    - earthengine-api installed
    - Authenticated via ee.Authenticate() (run once)
    - Google account registered for Earth Engine

Usage:
    python fetch_alphaearth_embeddings_expanded.py
    python fetch_alphaearth_embeddings_expanded.py --year 2019
    python fetch_alphaearth_embeddings_expanded.py --project my-gcp-project
    python fetch_alphaearth_embeddings_expanded.py --output data/trees_alphaearth.csv
"""

import argparse
import math
from pathlib import Path

import ee
import pandas as pd


# =============================================================================
# PATHS & CONSTANTS
# =============================================================================

DATA_DIR = Path(__file__).parent / "data"
TREES_CSV = DATA_DIR / "all_trees_unified.csv"

COLLECTION_ID = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
EMBEDDING_BANDS = [f"A{i:02d}" for i in range(64)]

# EE sampleRegions can handle thousands of points, but we batch conservatively
# to avoid hitting memory/timeout limits.
EE_BATCH_SIZE = 1000


# =============================================================================
# HELPERS
# =============================================================================

def build_location_key(lat: float, lon: float) -> str:
    """Round to 6 decimal places (~0.1 m) to build a stable join key."""
    return f"{lat:.6f},{lon:.6f}"


def make_ee_features(locs_df: pd.DataFrame) -> ee.FeatureCollection:
    """
    Create an EE FeatureCollection from a DataFrame with 'lat', 'lon', 'loc_id' columns.
    loc_id is an integer row index used to match results back.
    """
    features = []
    for _, row in locs_df.iterrows():
        geom = ee.Geometry.Point([float(row["lon"]), float(row["lat"])])
        features.append(ee.Feature(geom, {"loc_id": int(row["loc_id"])}))
    return ee.FeatureCollection(features)


def sample_embeddings_batch(
    locs_df: pd.DataFrame,
    image: ee.Image,
) -> pd.DataFrame:
    """
    Sample the embedding image at all locations in locs_df.
    Returns a DataFrame with loc_id + A00..A63 columns.
    Processes in batches of EE_BATCH_SIZE to avoid EE request limits.
    """
    all_rows = []
    n = len(locs_df)
    n_batches = math.ceil(n / EE_BATCH_SIZE)

    for batch_idx in range(n_batches):
        start = batch_idx * EE_BATCH_SIZE
        end = min(start + EE_BATCH_SIZE, n)
        batch_df = locs_df.iloc[start:end].copy()

        if n_batches > 1:
            print(f"    Batch {batch_idx + 1}/{n_batches} ({len(batch_df)} locations)...")

        fc = make_ee_features(batch_df)
        sampled = image.sampleRegions(
            collection=fc,
            scale=10,
            geometries=False,
        )
        results = sampled.getInfo()

        for feat in results["features"]:
            props = feat["properties"]
            all_rows.append(props)

    if not all_rows:
        return pd.DataFrame(columns=["loc_id"] + EMBEDDING_BANDS)

    return pd.DataFrame(all_rows)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fetch AlphaEarth embeddings for all trees with coordinates."
    )
    parser.add_argument(
        "--year", type=int, default=2018,
        help="Embedding year to fetch (default: 2018)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output CSV path (default: data/trees_alphaearth_{year}.csv)",
    )
    parser.add_argument(
        "--project", "-p", default=None,
        help="Google Cloud project ID for Earth Engine initialization",
    )
    args = parser.parse_args()

    output_path = (
        Path(args.output) if args.output
        else DATA_DIR / f"trees_alphaearth_{args.year}.csv"
    )

    print("=" * 60)
    print(f"AlphaEarth Embedding Extraction — Full Dataset (year={args.year})")
    print("=" * 60)

    # --- Authenticate & initialize ---
    print("\nInitializing Earth Engine...")
    ee.Authenticate()
    if args.project:
        ee.Initialize(project=args.project)
    else:
        ee.Initialize()
    print("  Authenticated and initialized.")

    # --- Load trees ---
    print(f"\nLoading trees from {TREES_CSV} ...")
    trees = pd.read_csv(TREES_CSV)
    print(f"  {len(trees):,} trees total")

    # Identify trees with valid coordinates
    has_coords = trees["latitude"].notna() & trees["longitude"].notna()
    print(f"  {has_coords.sum():,} trees with coordinates")
    print(f"  {(~has_coords).sum():,} trees without coordinates (will get NaN embeddings)")

    # --- Deduplicate locations ---
    trees["_loc_key"] = trees.apply(
        lambda r: build_location_key(r["latitude"], r["longitude"])
        if pd.notna(r["latitude"]) and pd.notna(r["longitude"])
        else None,
        axis=1,
    )

    valid_trees = trees[has_coords].copy()
    unique_locs = (
        valid_trees[["_loc_key", "latitude", "longitude"]]
        .drop_duplicates("_loc_key")
        .reset_index(drop=True)
    )
    unique_locs["loc_id"] = unique_locs.index
    unique_locs = unique_locs.rename(columns={"latitude": "lat", "longitude": "lon"})

    print(f"\n  {len(unique_locs):,} unique locations (deduplication reduces EE queries)")
    print(f"  Lat range : {unique_locs['lat'].min():.4f} → {unique_locs['lat'].max():.4f}")
    print(f"  Lon range : {unique_locs['lon'].min():.4f} → {unique_locs['lon'].max():.4f}")

    # --- Load EE image ---
    print(f"\nQuerying {COLLECTION_ID} for {args.year}...")
    collection = ee.ImageCollection(COLLECTION_ID)
    image = (
        collection
        .filterDate(f"{args.year}-01-01", f"{args.year + 1}-01-01")
        .select(EMBEDDING_BANDS)
        .mosaic()
    )

    # --- Sample ---
    print(f"  Sampling {len(unique_locs):,} unique locations (batch size={EE_BATCH_SIZE})...")
    emb_df = sample_embeddings_batch(unique_locs, image)
    print(f"  Retrieved embeddings for {len(emb_df):,} locations")

    # --- Build loc_id → embedding map ---
    loc_key_to_id = unique_locs.set_index("_loc_key")["loc_id"].to_dict()
    emb_by_id = emb_df.set_index("loc_id")

    # --- Join back to all trees ---
    print("\nJoining embeddings back to all trees...")
    emb_cols = {band: [] for band in EMBEDDING_BANDS}

    n_missing = 0
    for _, row in trees.iterrows():
        loc_key = row["_loc_key"]
        if loc_key is None:
            # No coordinates
            for band in EMBEDDING_BANDS:
                emb_cols[band].append(float("nan"))
            continue

        loc_id = loc_key_to_id.get(loc_key)
        if loc_id is None or loc_id not in emb_by_id.index:
            # Coordinate exists but EE returned no data (masked pixel, ocean, etc.)
            for band in EMBEDDING_BANDS:
                emb_cols[band].append(float("nan"))
            n_missing += 1
            continue

        emb_row = emb_by_id.loc[loc_id]
        for band in EMBEDDING_BANDS:
            emb_cols[band].append(emb_row.get(band, float("nan")))

    if n_missing > 0:
        print(f"  WARNING: {n_missing} trees have coordinates but no EE data (masked/ocean pixels)")

    # Attach embedding columns
    result = trees.drop(columns=["_loc_key"]).copy()
    for band in EMBEDDING_BANDS:
        result[band] = emb_cols[band]

    # --- Save ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    n_with_emb = result["A00"].notna().sum()
    print(f"\nOutput saved to {output_path}")
    print(f"  Rows            : {len(result):,}")
    print(f"  With embeddings : {n_with_emb:,}")
    print(f"  Without         : {len(result) - n_with_emb:,}")
    print(f"  Columns         : {len(result.columns)} (tree attrs + 64 embedding dims)")


if __name__ == "__main__":
    main()
