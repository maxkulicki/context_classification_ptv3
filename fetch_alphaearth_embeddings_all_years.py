#!/usr/bin/env python3
"""
Fetch AlphaEarth Satellite Embeddings for all trees across ALL available years.

Queries the Google Earth Engine Satellite Embedding V1 Annual dataset
(GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL), discovers every year present in
the collection, and extracts the 64-dimensional embedding vector at each
unique location for each year.

Output is a single wide-format CSV:
    <tree attrs> | ae_<year>_A00 ... ae_<year>_A63 | ae_<year+1>_A00 ...

Trees without coordinates (NaN lat/lon) receive NaN embeddings for all years.
Many trees share the same plot location — deduplication means EE is queried
once per unique coordinate pair per year.

Prerequisites:
    - earthengine-api installed
    - Authenticated via ee.Authenticate() (run once)
    - Google account registered for Earth Engine

Usage:
    python fetch_alphaearth_embeddings_all_years.py
    python fetch_alphaearth_embeddings_all_years.py --project my-gcp-project
    python fetch_alphaearth_embeddings_all_years.py --output data/trees_ae_all_years.csv
    python fetch_alphaearth_embeddings_all_years.py --years 2017 2018 2019
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


def discover_years() -> list[int]:
    """
    List all distinct years available in the ANNUAL collection.
    Reads system:time_start from each image and converts to year.
    """
    collection = ee.ImageCollection(COLLECTION_ID)
    dates = collection.aggregate_array("system:time_start").getInfo()
    years = sorted({
        pd.Timestamp(ms, unit="ms").year
        for ms in dates
    })
    return years


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
            print(f"      Batch {batch_idx + 1}/{n_batches} ({len(batch_df)} locations)...")

        fc = make_ee_features(batch_df)
        sampled = image.sampleRegions(
            collection=fc,
            scale=10,
            geometries=False,
        )
        results = sampled.getInfo()

        for feat in results["features"]:
            all_rows.append(feat["properties"])

    if not all_rows:
        return pd.DataFrame(columns=["loc_id"] + EMBEDDING_BANDS)

    return pd.DataFrame(all_rows)


def fetch_year_embeddings(
    year: int,
    unique_locs: pd.DataFrame,
    loc_key_to_id: dict,
    trees: pd.DataFrame,
) -> dict[str, list]:
    """
    Fetch embeddings for a single year and return a dict of
    { 'ae_{year}_A00': [...], ..., 'ae_{year}_A63': [...] }
    aligned to the rows of `trees`.
    """
    year_bands = [f"ae_{year}_{b}" for b in EMBEDDING_BANDS]

    print(f"\n  Year {year}: querying {COLLECTION_ID}...")
    collection = ee.ImageCollection(COLLECTION_ID)
    image = (
        collection
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .select(EMBEDDING_BANDS)
        .mosaic()
    )

    print(f"    Sampling {len(unique_locs):,} unique locations...")
    emb_df = sample_embeddings_batch(unique_locs, image)
    print(f"    Retrieved embeddings for {len(emb_df):,} locations")

    emb_by_id = emb_df.set_index("loc_id")

    emb_cols = {col: [] for col in year_bands}
    n_missing = 0

    for _, row in trees.iterrows():
        loc_key = row["_loc_key"]
        if loc_key is None:
            for col in year_bands:
                emb_cols[col].append(float("nan"))
            continue

        loc_id = loc_key_to_id.get(loc_key)
        if loc_id is None or loc_id not in emb_by_id.index:
            for col in year_bands:
                emb_cols[col].append(float("nan"))
            n_missing += 1
            continue

        emb_row = emb_by_id.loc[loc_id]
        for band, col in zip(EMBEDDING_BANDS, year_bands):
            emb_cols[col].append(emb_row.get(band, float("nan")))

    if n_missing > 0:
        print(f"    WARNING: {n_missing} trees have coordinates but no EE data for {year}")

    return emb_cols


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fetch AlphaEarth embeddings for all trees across all available years."
    )
    parser.add_argument(
        "--years", type=int, nargs="+", default=None,
        help="Specific years to fetch (default: all years in the collection)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output CSV path (default: data/trees_alphaearth_all_years.csv)",
    )
    parser.add_argument(
        "--project", "-p", default=None,
        help="Google Cloud project ID for Earth Engine initialization",
    )
    args = parser.parse_args()

    output_path = (
        Path(args.output) if args.output
        else DATA_DIR / "trees_alphaearth_all_years.csv"
    )

    print("=" * 60)
    print("AlphaEarth Embedding Extraction — All Years")
    print("=" * 60)

    # --- Authenticate & initialize ---
    print("\nInitializing Earth Engine...")
    ee.Authenticate()
    if args.project:
        ee.Initialize(project=args.project)
    else:
        ee.Initialize()
    print("  Authenticated and initialized.")

    # --- Discover years ---
    if args.years:
        years = sorted(args.years)
        print(f"\nUsing specified years: {years}")
    else:
        print("\nDiscovering available years in collection...")
        years = discover_years()
        print(f"  Found {len(years)} year(s): {years}")

    # --- Load trees ---
    print(f"\nLoading trees from {TREES_CSV} ...")
    trees = pd.read_csv(TREES_CSV)
    print(f"  {len(trees):,} trees total")

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
    loc_key_to_id = unique_locs.set_index("_loc_key")["loc_id"].to_dict()

    print(f"\n  {len(unique_locs):,} unique locations")
    print(f"  Lat range : {unique_locs['lat'].min():.4f} → {unique_locs['lat'].max():.4f}")
    print(f"  Lon range : {unique_locs['lon'].min():.4f} → {unique_locs['lon'].max():.4f}")

    # --- Fetch embeddings for each year ---
    print(f"\nFetching embeddings for {len(years)} year(s)...")
    all_emb_cols: dict[str, list] = {}

    for year in years:
        year_cols = fetch_year_embeddings(year, unique_locs, loc_key_to_id, trees)
        all_emb_cols.update(year_cols)

    # --- Assemble result ---
    print("\nAssembling output...")
    result = trees.drop(columns=["_loc_key"]).copy()
    for col, values in all_emb_cols.items():
        result[col] = values

    # --- Save ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    # Summary
    first_year_col = f"ae_{years[0]}_A00"
    n_with_emb = result[first_year_col].notna().sum()
    n_emb_cols = len(years) * 64

    print(f"\nOutput saved to {output_path}")
    print(f"  Rows              : {len(result):,}")
    print(f"  Years extracted   : {years}")
    print(f"  Embedding columns : {n_emb_cols} ({len(years)} years × 64 dims)")
    print(f"  Trees with emb.   : {n_with_emb:,} (checked against {first_year_col})")
    print(f"  Total columns     : {len(result.columns)}")


if __name__ == "__main__":
    main()
