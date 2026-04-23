"""
Precompute a pool of spatially-shifted SINR embeddings for location augmentation.

For each tree, 8 random bearings × 3 distances (10, 20, 50 km) = 24 shifted
embeddings are extracted. The pool is saved as:
    {stem: np.float32 array (24, 256)}

The flat-earth approximation is used for coordinate shifts, which is accurate
to < 0.1% at 50 km and fully sufficient for species-distribution-scale signals.

Usage (from context_classification_ptv3/):
    conda run -n context_baseline python extract_sinr_features_spatial_aug.py \
        --input  data/snapshot_v1/split.csv \
        --output data/snapshot_v1/sinr_spatial_aug_pool.pt \
        --model  sinr/pretrained_models/model_an_full_input_enc_sin_cos_distilled_from_env.pt \
        --seed   42

Pool layout (axis 0, 24 entries):
    indices  0– 7: distance 10 km, 8 random bearings
    indices  8–15: distance 20 km, same 8 bearings
    indices 16–23: distance 50 km, same 8 bearings
"""

import sys
import os
import argparse

import numpy as np
import torch

SINR_DIR = os.path.join(os.path.dirname(__file__), "sinr", "sinr")
sys.path.insert(0, SINR_DIR)

import models
import utils

DISTANCES_KM = [10.0, 20.0, 50.0]
N_BEARINGS   = 8
DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__),
    "sinr", "pretrained_models",
    "model_an_full_input_enc_sin_cos_distilled_from_env.pt",
)


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    params = checkpoint["params"]
    model = models.get_model(params)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    enc = utils.CoordEncoder(params["input_enc"])
    return model, enc


def shifted_coords(lons, lats, distance_km, bearings):
    """
    Compute shifted (lon, lat) for each (lon, lat, bearing) combination.

    Parameters
    ----------
    lons, lats : np.ndarray, shape (N,)
    distance_km : float
    bearings : np.ndarray, shape (N, n_bearings)  — radians

    Returns
    -------
    lons_out, lats_out : np.ndarray, shape (N, n_bearings)
    """
    d_lat = distance_km / 111.32                          # degrees
    d_lon = distance_km / (111.32 * np.cos(np.radians(lats)))  # degrees

    lats_out = lats[:, None] + d_lat * np.cos(bearings)
    lons_out = lons[:, None] + d_lon[:, None] * np.sin(bearings)
    return lons_out, lats_out


def extract_batch(lons_flat, lats_flat, model, enc, device, batch_size=4096):
    """Extract SINR embeddings for a flat array of coordinates. Returns (N, 256)."""
    results = []
    for start in range(0, len(lons_flat), batch_size):
        end = min(start + batch_size, len(lons_flat))
        locs = torch.from_numpy(
            np.stack([lons_flat[start:end], lats_flat[start:end]], axis=1).astype(np.float32)
        )
        locs_enc = enc.encode(locs.clone())
        with torch.no_grad():
            emb = model(locs_enc.to(device), return_feats=True)
        results.append(emb.cpu().numpy())
    return np.concatenate(results, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute spatially-shifted SINR embedding pools"
    )
    parser.add_argument("--input",  required=True, help="CSV with laz_path, longitude, latitude")
    parser.add_argument("--output", required=True, help="Output .pt path")
    parser.add_argument("--model",  default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument(
        "--distances_km", type=float, nargs="+", default=DISTANCES_KM,
        help="Shift distances in km (default: 10 20 50)"
    )
    parser.add_argument(
        "--n_bearings", type=int, default=N_BEARINGS,
        help="Random bearings per distance (default: 8)"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading model: {args.model}")
    model, enc = load_model(args.model, device)

    import csv
    print(f"Reading {args.input}")
    rows = []
    with open(args.input, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"  {len(rows):,} trees")

    lons = np.array([float(r["longitude"]) for r in rows], dtype=np.float64)
    lats = np.array([float(r["latitude"])  for r in rows], dtype=np.float64)
    stems = [os.path.splitext(os.path.basename(r["laz_path"]))[0] for r in rows]

    n_trees    = len(rows)
    n_dist     = len(args.distances_km)
    n_bear     = args.n_bearings
    n_aug      = n_dist * n_bear   # 24 by default
    feat_dim   = 256

    # Sample bearings once, shared across all distances (same 8 directions per tree)
    rng = np.random.default_rng(args.seed)
    # Shape: (n_trees, n_bearings)
    bearings = rng.uniform(0, 2 * np.pi, size=(n_trees, n_bear))

    # Collect ALL shifted (lon, lat) pairs: shape (n_trees * n_aug,)
    all_lons = np.empty(n_trees * n_aug, dtype=np.float32)
    all_lats = np.empty(n_trees * n_aug, dtype=np.float32)

    for d_idx, dist_km in enumerate(args.distances_km):
        sl, sr = d_idx * n_bear, (d_idx + 1) * n_bear
        lons_shifted, lats_shifted = shifted_coords(lons, lats, dist_km, bearings)
        # lons_shifted: (n_trees, n_bearings)
        # Interleave into flat array: tree0_d0_b0, tree0_d0_b1, ..., tree1_d0_b0, ...
        # Layout: for each tree, distances are grouped → index = tree*n_aug + d*n_bear + b
        for b_idx in range(n_bear):
            flat_idx = np.arange(n_trees) * n_aug + sl + b_idx
            all_lons[flat_idx] = lons_shifted[:, b_idx]
            all_lats[flat_idx] = lats_shifted[:, b_idx]

    print(f"Extracting {n_trees * n_aug:,} embeddings "
          f"({n_trees} trees × {n_dist} distances × {n_bear} bearings)...")
    all_emb = extract_batch(all_lons, all_lats, model, enc, device, args.batch_size)
    # all_emb: (n_trees * n_aug, feat_dim)

    # Reshape to (n_trees, n_aug, feat_dim) and build lookup
    all_emb = all_emb.reshape(n_trees, n_aug, feat_dim).astype(np.float32)

    pool = {stem: all_emb[i] for i, stem in enumerate(stems)}

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(pool, args.output)
    print(f"Saved pool to {args.output}")
    print(f"  Pool shape per tree: {next(iter(pool.values())).shape}")
    print(f"  Layout: indices 0–{n_bear-1} = {args.distances_km[0]} km, "
          f"{n_bear}–{2*n_bear-1} = {args.distances_km[1]} km, "
          f"{2*n_bear}–{3*n_bear-1} = {args.distances_km[2]} km")


if __name__ == "__main__":
    main()
