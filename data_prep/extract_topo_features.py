#!/usr/bin/env python3
"""
Extract topographic features from FABDEM V1-2 for all trees in unified CSV.

Features extracted per tree:
  - elevation (m) — direct DTM value
  - slope (degrees) — Horn's method (Horn 1981)
  - northness (cos of aspect, -1 to 1) — cos(aspect), avoids circularity
  - eastness (sin of aspect, -1 to 1) — sin(aspect)
  - TRI (Terrain Ruggedness Index) — Riley et al. 1999, 3x3 window
  - TPI (Topographic Position Index) — elevation minus 5x5 neighborhood mean

Data source: FABDEM V1-2 (University of Bristol)
  - 30m resolution global DTM (forests & buildings removed from Copernicus GLO-30)
  - License: CC BY-NC-SA 4.0
  - Hawker et al. (2022), Environmental Research Letters

Usage:
  1. Run with --download-tiles to fetch required FABDEM tiles
  2. Run with --extract to compute features and write output CSV

  python extract_topo_features.py --download-tiles --tiles-dir ./fabdem_tiles
  python extract_topo_features.py --extract \
      --input all_trees_unified.csv \
      --tiles-dir ./fabdem_tiles \
      --output all_trees_unified_topo.csv
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from scipy.ndimage import uniform_filter


# ---------------------------------------------------------------------------
# FABDEM tile URL construction
# ---------------------------------------------------------------------------

def fabdem_tile_name(lat_int: int, lon_int: int) -> str:
    """Return the FABDEM tile filename for a given integer lat/lon."""
    lat_str = f"{'N' if lat_int >= 0 else 'S'}{abs(lat_int):02d}"
    lon_str = f"{'E' if lon_int >= 0 else 'W'}{abs(lon_int):03d}"
    return f"{lat_str}{lon_str}_FABDEM_V1-2.tif"


def fabdem_block_name(lat_int: int, lon_int: int) -> str:
    """Return the 10x10 degree block folder name."""
    # Python floor division already handles negatives correctly:
    # e.g. -2 // 10 = -1, * 10 = -10 (correct block start)
    lat_block_s = (lat_int // 10) * 10
    lon_block_w = (lon_int // 10) * 10
    lat_block_n = lat_block_s + 10
    lon_block_e = lon_block_w + 10

    def fmt_lat(v):
        return f"{'N' if v >= 0 else 'S'}{abs(v):02d}"

    def fmt_lon(v):
        return f"{'E' if v >= 0 else 'W'}{abs(v):03d}"

    return (
        f"{fmt_lat(lat_block_s)}{fmt_lon(lon_block_w)}-"
        f"{fmt_lat(lat_block_n)}{fmt_lon(lon_block_e)}_FABDEM_V1-2"
    )


def fabdem_download_url_hf(lat_int: int, lon_int: int) -> str:
    """Download URL for individual FABDEM tile from Hugging Face mirror.

    No authentication required. Individual GeoTIFF tiles, no ZIP handling.
    Source: https://huggingface.co/datasets/links-ads/fabdem-v12
    """
    block = fabdem_block_name(lat_int, lon_int)
    tile = fabdem_tile_name(lat_int, lon_int)
    return (
        f"https://huggingface.co/datasets/links-ads/fabdem-v12/"
        f"resolve/main/tiles/{block}/{tile}"
    )


def fabdem_zip_url_bristol(lat_int: int, lon_int: int) -> str:
    """Download URL for 10x10 degree ZIP block from University of Bristol.

    No authentication required, but files are large ZIP archives containing
    all 1x1 degree tiles in the block. Manual download + unzip recommended.
    """
    block = fabdem_block_name(lat_int, lon_int)
    return (
        f"https://data.bris.ac.uk/datasets/"
        f"s5hqmjcdj8yo2ibzi9b4ew3sn/{block}.zip"
    )


# ---------------------------------------------------------------------------
# Tile discovery
# ---------------------------------------------------------------------------

def find_required_tiles(df: pd.DataFrame) -> list[tuple[int, int]]:
    """Find all 1x1 degree tiles needed based on tree coordinates."""
    tiles = set()
    valid = df.dropna(subset=["latitude", "longitude"])
    for _, row in valid.iterrows():
        lat_int = int(math.floor(row["latitude"]))
        lon_int = int(math.floor(row["longitude"]))
        tiles.add((lat_int, lon_int))
    return sorted(tiles)


# ---------------------------------------------------------------------------
# Download tiles
# ---------------------------------------------------------------------------

def download_tiles(tiles: list[tuple[int, int]], tiles_dir: Path):
    """Download required FABDEM tiles from Hugging Face (no auth needed).

    Falls back to printing Bristol ZIP URLs for manual download if HF fails.
    """
    import urllib.request

    tiles_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for lat_int, lon_int in tiles:
        tile_file = tiles_dir / fabdem_tile_name(lat_int, lon_int)
        if tile_file.exists():
            print(f"  [skip] {tile_file.name} already exists")
            continue

        url = fabdem_download_url_hf(lat_int, lon_int)
        print(f"  [download] {tile_file.name}")
        print(f"    URL: {url}")
        try:
            urllib.request.urlretrieve(url, tile_file)
            # Verify it's a valid GeoTIFF
            with rasterio.open(tile_file) as src:
                _ = src.bounds
            print(f"    OK ({tile_file.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:
            print(f"    FAILED: {e}")
            if tile_file.exists():
                tile_file.unlink()
            failed.append((lat_int, lon_int))

    if failed:
        print("\n  Some tiles failed to download from Hugging Face.")
        print("  You can manually download the ZIP blocks from Bristol:")
        blocks_needed = set()
        for lat_int, lon_int in failed:
            blocks_needed.add(fabdem_zip_url_bristol(lat_int, lon_int))
        for url in sorted(blocks_needed):
            print(f"    {url}")
        print(f"\n  Unzip and place the .tif files in: {tiles_dir}/")



# ---------------------------------------------------------------------------
# Topographic feature computation
# ---------------------------------------------------------------------------

def compute_topo_from_dem(dem: np.ndarray, res_x: float, res_y: float) -> dict:
    """
    Compute topographic variables from a DEM window.

    Parameters
    ----------
    dem : 2D array, the elevation window
    res_x, res_y : pixel resolution in meters (approx)

    Returns dict of 2D arrays (same shape as dem) for each variable.
    """
    # Pad to handle edges
    dem_pad = np.pad(dem, 1, mode="edge")

    # Partial derivatives using Horn's method (3x3)
    # dz/dx
    dzdx = (
        (dem_pad[:-2, 2:] + 2 * dem_pad[1:-1, 2:] + dem_pad[2:, 2:])
        - (dem_pad[:-2, :-2] + 2 * dem_pad[1:-1, :-2] + dem_pad[2:, :-2])
    ) / (8.0 * res_x)

    # dz/dy  (note: row index increases downward = south for most grids)
    dzdy = (
        (dem_pad[:-2, :-2] + 2 * dem_pad[:-2, 1:-1] + dem_pad[:-2, 2:])
        - (dem_pad[2:, :-2] + 2 * dem_pad[2:, 1:-1] + dem_pad[2:, 2:])
    ) / (8.0 * res_y)

    # Slope (degrees)
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    slope_deg = np.degrees(slope_rad)

    # Aspect (degrees, 0=N clockwise)
    aspect = np.degrees(np.arctan2(-dzdx, dzdy)) % 360

    # Northness / Eastness
    northness = np.cos(np.radians(aspect))
    eastness = np.sin(np.radians(aspect))

    # TRI (Riley et al. 1999) — RMS of elevation differences in 3x3
    center = dem
    tri_sum = np.zeros_like(dem, dtype=float)
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            shifted = dem_pad[1 + di : 1 + di + dem.shape[0],
                              1 + dj : 1 + dj + dem.shape[1]]
            tri_sum += (shifted - center) ** 2
    tri = np.sqrt(tri_sum / 8.0)

    # TPI — elevation minus mean of 5x5 neighborhood
    mean_5x5 = uniform_filter(dem.astype(float), size=5, mode="nearest")
    tpi = dem - mean_5x5

    return {
        "slope": slope_deg,
        "northness": northness,
        "eastness": eastness,
        "tri": tri,
        "tpi": tpi,
    }


# ---------------------------------------------------------------------------
# Per-tree extraction
# ---------------------------------------------------------------------------

TOPO_COLUMNS = [
    "elevation", "slope", "northness", "eastness", "tri", "tpi",
]


def extract_features_for_trees(df: pd.DataFrame, tiles_dir: Path) -> pd.DataFrame:
    """
    Extract topographic features for each tree from local FABDEM tiles.

    Strategy: group trees by tile, open each tile once, read a generous window
    around the cluster, compute topo grids, then sample per-tree coords.
    """
    result_cols = {col: np.full(len(df), np.nan) for col in TOPO_COLUMNS}

    valid_mask = df["latitude"].notna() & df["longitude"].notna()
    valid_df = df[valid_mask].copy()

    if len(valid_df) == 0:
        print("No valid coordinates found.")
        for col in TOPO_COLUMNS:
            df[col] = np.nan
        return df

    # Assign tile key
    valid_df["_tile_lat"] = valid_df["latitude"].apply(lambda x: int(math.floor(x)))
    valid_df["_tile_lon"] = valid_df["longitude"].apply(lambda x: int(math.floor(x)))

    grouped = valid_df.groupby(["_tile_lat", "_tile_lon"])
    n_tiles = len(grouped)
    print(f"Processing {len(valid_df)} trees across {n_tiles} tiles...")

    for tile_idx, ((lat_int, lon_int), group) in enumerate(grouped):
        tile_file = tiles_dir / fabdem_tile_name(lat_int, lon_int)
        print(
            f"  [{tile_idx + 1}/{n_tiles}] Tile ({lat_int}, {lon_int}): "
            f"{len(group)} trees ... ",
            end="",
        )

        if not tile_file.exists():
            print("MISSING — skipping")
            continue

        try:
            with rasterio.open(tile_file) as src:
                # Read the entire tile (1 degree ≈ 3601x3601 at 1 arcsec)
                dem = src.read(1).astype(np.float64)
                nodata = src.nodata
                transform = src.transform

                # Approximate pixel resolution in meters at tile center
                mid_lat = lat_int + 0.5
                res_y_m = abs(transform.e) * 111320  # degrees to meters (latitude)
                res_x_m = (
                    abs(transform.a) * 111320 * math.cos(math.radians(mid_lat))
                )

            # Replace nodata with NaN
            if nodata is not None:
                dem[dem == nodata] = np.nan

            # Compute topo grids for entire tile
            topo = compute_topo_from_dem(dem, res_x_m, res_y_m)

            # Sample per tree
            n_extracted = 0
            for idx, row in group.iterrows():
                try:
                    r, c = rowcol(transform, row["longitude"], row["latitude"])
                    if 0 <= r < dem.shape[0] and 0 <= c < dem.shape[1]:
                        orig_idx = df.index.get_loc(idx) if idx in df.index else None
                        if orig_idx is None:
                            continue
                        result_cols["elevation"][orig_idx] = dem[r, c]
                        for key in topo:
                            result_cols[key][orig_idx] = topo[key][r, c]
                        n_extracted += 1
                except Exception:
                    continue

            print(f"{n_extracted} extracted")

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    # Attach to dataframe
    for col in TOPO_COLUMNS:
        df[col] = result_cols[col]

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract FABDEM topographic features for tree species classification."
    )
    parser.add_argument(
        "--download-tiles",
        action="store_true",
        help="Download required FABDEM tiles.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract topographic features.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="all_trees_unified.csv",
        help="Input CSV with columns: dataset, tree_id, species, latitude, longitude",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="all_trees_unified_topo.csv",
        help="Output CSV with appended topographic columns.",
    )
    parser.add_argument(
        "--tiles-dir",
        type=str,
        default="./fabdem_tiles",
        help="Directory for FABDEM tile storage.",
    )

    args = parser.parse_args()

    if not args.download_tiles and not args.extract:
        parser.print_help()
        sys.exit(1)

    tiles_dir = Path(args.tiles_dir)

    # Load data
    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input)
    print(f"  {len(df)} trees, {df['latitude'].notna().sum()} with coordinates")

    # Find required tiles
    tiles = find_required_tiles(df)
    print(f"  {len(tiles)} tiles needed: {tiles}")

    if args.download_tiles:
        print("\n--- Downloading FABDEM tiles ---")
        download_tiles(tiles, tiles_dir)

    if args.extract:
        print("\n--- Extracting topographic features ---")
        df = extract_features_for_trees(df, tiles_dir)

        # Summary
        for col in TOPO_COLUMNS:
            valid = df[col].notna().sum()
            print(f"  {col}: {valid}/{len(df)} values")

        # Save
        df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")

        # Show sample
        print("\nSample (first 5 with topo data):")
        sample = df[df["elevation"].notna()].head()
        print(sample[["dataset", "tree_id", "species"] + TOPO_COLUMNS].to_string())


if __name__ == "__main__":
    main()