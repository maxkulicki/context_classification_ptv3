#!/usr/bin/env python3
"""
Extract GeoPlantNet species distribution logits at TLS plot locations.

Reads Cloud Optimized GeoTIFFs remotely via HTTP range requests — no full
raster downloads needed (~4s per species for 272 plots). Only the pixels
at plot centroids are fetched.

Values:
    float32 logits : DeepSDM model output (sigmoid gives probability, but
                     logits are more informative as features since
                     probabilities are saturated near 1.0)
    -1.0 / NaN     : nodata (outside raster extent or species range)

Usage:
    python extract_gpn_logits.py
    python extract_gpn_logits.py --country germany
    python extract_gpn_logits.py --output data/plots_gpn_logits.csv
    python extract_gpn_logits.py --local-dir data/geoplantnet  # prefer local files

Adding new species:
    1. Browse the STAC catalog to find the species ID:
       curl -s https://geo.plantnet.org/stac-catalogs/geoplantnet/v1/species/{country}/collection.json
    2. Look for the item href, e.g. ./poland_8987_pinus_sylvestris_l.json
    3. Add an entry to the SPECIES dict below:
       "Column_name": (tree_code, gpn_id, "filename_suffix")

Adapting to other countries:
    Use --country {name} where {name} matches the STAC catalog directory
    (e.g. germany, norway, france). Available countries are listed at:
    https://geo.plantnet.org/stac-catalogs/geoplantnet/v1/species/catalog.json
    Note: species IDs are the same across countries, but not all species
    have maps for every country. The script reports errors for missing files.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer


# =============================================================================
# PATHS
# =============================================================================

DATA_DIR = Path(__file__).parent / "data"
PLOTS_CSV = DATA_DIR / "TreeScanPL_plot_locations.csv"

CRS_PUWG92 = "EPSG:2180"
CRS_GPN = "EPSG:3857"

BASE_URL = "https://geo.plantnet.org/stacs-download/geoplantnet/species"

# Mapping: column_name -> (tree_code, gpn_species_id, filename_suffix)
# tree_code matches species_id_names.csv CODE column
SPECIES = {
    "Pinus_sylvestris":      (1,   8987, "pinus_sylvestris_l"),
    "Larix_decidua":         (8,    385, "larix_decidua_mill"),
    "Picea_abies":           (11,  1283, "picea_abies_l_h_karst"),
    "Abies_alba":            (21,  6574, "abies_alba_mill"),
    "Fagus_sylvatica":       (30,  5314, "fagus_sylvatica_l"),
    "Quercus_robur":         (41,  9324, "quercus_robur_l"),
    "Quercus_rubra":         (43,  3659, "quercus_rubra_l"),
    "Acer_platanoides":      (44,   876, "acer_platanoides_l"),
    "Acer_pseudoplatanus":   (45,  4064, "acer_pseudoplatanus_l"),
    "Ulmus_minor":           (46,  3157, "ulmus_minor_mill"),
    "Fraxinus_excelsior":    (48,  9329, "fraxinus_excelsior_l"),
    "Carpinus_betulus":       (50,  9347, "carpinus_betulus_l"),
    "Betula_pendula":        (60,  8079, "betula_pendula_roth"),
    "Crataegus_monogyna":    (66,  5959, "crataegus_monogyna_jacq"),
    "Alnus_glutinosa":       (71,  6705, "alnus_glutinosa_l_gaertn"),
    "Alnus_incana":          (72,  4840, "alnus_incana_l_moench"),
    "Prunus_avium":          (82,  5263, "prunus_avium_l_l"),
    "Prunus_padus":          (86,  3204, "prunus_padus_l"),
    "Sorbus_aucuparia":      (87,  8419, "sorbus_aucuparia_l"),
    "Populus_tremula":       (91,   793, "populus_tremula_l"),
    "Tilia_cordata":         (95,  3704, "tilia_cordata_mill"),
    "Salix_caprea":          (97,  9687, "salix_caprea_l"),
    "Prunus_serotina":       (98,  8804, "prunus_serotina_ehrh"),
    "Acer_campestre":        (107,  795, "acer_campestre_l"),
}


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract GeoPlantNet logits at TLS plot locations."
    )
    parser.add_argument(
        "--output", "-o",
        default=str(DATA_DIR / "plots_gpn_logits.csv"),
        help="Output CSV path (default: data/plots_gpn_logits.csv)",
    )
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Optional directory with local GeoTIFFs (prefer over remote when available)",
    )
    parser.add_argument(
        "--country",
        default="poland",
        help="Country name for URL construction (default: poland)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("GeoPlantNet Logit Extraction")
    print("=" * 60)

    # Load plot locations (avoid pandas for env compatibility)
    print("\nLoading plot locations...")
    with open(PLOTS_CSV) as f:
        reader = csv.DictReader(f, delimiter=";")
        plots = list(reader)
    print(f"  {len(plots)} plots loaded")

    plot_x = [float(p["X"]) for p in plots]
    plot_y = [float(p["Y"]) for p in plots]

    # Transform coordinates to GPN CRS
    transformer = Transformer.from_crs(CRS_PUWG92, CRS_GPN, always_xy=True)
    coords_3857 = list(zip(*transformer.transform(plot_x, plot_y)))

    # Sample each species
    local_dir = Path(args.local_dir) if args.local_dir else None
    mode = "remote (COG range requests)" if local_dir is None else f"local-first ({local_dir})"
    print(f"\nSampling {len(SPECIES)} species at {len(plots)} plots [{mode}]...")
    species_values = {}

    for species_name, (tree_code, gpn_id, fname_suffix) in SPECIES.items():
        filename = f"gpn_v1_50m_{args.country}_{gpn_id}_{fname_suffix}.tif"

        # Default: remote via HTTP range requests; use local if --local-dir given and file exists
        if local_dir and (local_dir / filename).exists():
            source_path = str(local_dir / filename)
            source_label = "local"
        else:
            source_path = f"{BASE_URL}/{filename}"
            source_label = "remote"

        t0 = time.time()
        try:
            with rasterio.open(source_path) as src:
                values = np.array([v[0] for v in src.sample(coords_3857)])
                # Replace nodata with NaN
                if src.nodata is not None:
                    values = np.where(values == src.nodata, np.nan, values)
        except Exception as e:
            print(f"  {species_name:25s}  ERROR ({source_label}): {e}")
            values = np.full(len(plots), np.nan)

        elapsed = time.time() - t0
        species_values[species_name] = values

        n_valid = np.sum(np.isfinite(values))
        n_nodata = np.sum(np.isnan(values))
        val_min = np.nanmin(values) if n_valid > 0 else float("nan")
        val_max = np.nanmax(values) if n_valid > 0 else float("nan")
        print(
            f"  {species_name:25s}  valid={n_valid:3d}  nodata={n_nodata:3d}  "
            f"range=[{val_min:6.2f}, {val_max:6.2f}]  {elapsed:.1f}s ({source_label})"
        )

    # Build output CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["source", "file", "year", "num", "num_txt", "X", "Y"]
    header += list(species_values.keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(header)
        for i, plot in enumerate(plots):
            row = [plot[col] for col in ["source", "file", "year", "num", "num_txt", "X", "Y"]]
            for species_name in species_values:
                v = species_values[species_name][i]
                row.append("" if np.isnan(v) else f"{v:.6f}")
            writer.writerow(row)

    species_cols = list(species_values.keys())
    print(f"\nOutput saved to {output_path}")
    print(f"  Rows: {len(plots)}")
    print(f"  Columns: {len(header)} (7 plot attrs + {len(species_cols)} species)")


if __name__ == "__main__":
    main()
