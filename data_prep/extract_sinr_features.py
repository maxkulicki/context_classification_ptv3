"""
Extract SINR (Spatial Implicit Neural Representation) location embeddings.

SINR maps (lon, lat) → 256-dim learned embedding capturing species-distribution
context. These can be used as location-based features for downstream classifiers.

Usage:
    # Single location
    python extract_sinr_features.py --lon 21.0 --lat 50.0

    # Batch from CSV (must have 'longitude' and 'latitude' columns)
    python extract_sinr_features.py --input data/all_trees_unified.csv --output data/sinr_features.csv

    # Choose model variant
    python extract_sinr_features.py --input data/all_trees_unified.csv \
        --model sinr/pretrained_models/model_an_full_input_enc_sin_cos_distilled_from_env.pt

Model variants (all in sinr/pretrained_models/):
    model_an_full_input_enc_sin_cos_distilled_from_env.pt  [DEFAULT, recommended]
        Distilled from a model that saw BioCLIM + elevation → richest embeddings
    model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt
        Trained with up to 1000 samples/species
    model_an_full_input_enc_sin_cos_hard_cap_num_per_class_100.pt
    model_an_full_input_enc_sin_cos_hard_cap_num_per_class_10.pt
        Fewest samples/species, sharpest decision boundaries but less stable

Architecture: ResidualFCNet
    sin_cos encode → Linear(4→256) → ReLU → 4× ResLayer(256) → [256-dim embedding]
                                                                  ↓
                                                      Linear(256→num_classes) + sigmoid
    return_feats=True stops before the classification head.

Resolution note: the encoding is mathematically continuous (sin/cos), but the
meaningful resolution of the learned representations is ~1–10 km. The model was
trained on iNaturalist GPS observations; species distributions don't shift at
sub-km scales, so embeddings for trees <1 km apart are nearly identical.
At ~10 km separation embeddings begin to diverge meaningfully (different climate
cells, habitat patches). At >50 km differences are large. In practice, trees in
the same forest plot will receive essentially the same embedding — use it as a
site-level context signal, not a per-tree one.
"""

import sys
import os
import argparse
import csv
import math

import numpy as np
import torch

# Add sinr package to path
SINR_DIR = os.path.join(os.path.dirname(__file__), "sinr", "sinr")
sys.path.insert(0, SINR_DIR)

import models
import utils

DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__),
    "sinr", "pretrained_models",
    "model_an_full_input_enc_sin_cos_distilled_from_env.pt",
)


def load_model(model_path: str, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)
    params = checkpoint["params"]
    model = models.get_model(params)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    enc = utils.CoordEncoder(params["input_enc"])
    return model, enc, params


def extract_features(
    lons: np.ndarray,
    lats: np.ndarray,
    model,
    enc: utils.CoordEncoder,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """
    Returns float32 array of shape [N, 256].
    lons/lats: 1-D arrays of WGS84 longitude/latitude.
    Rows where lon or lat is NaN are returned as NaN rows.
    """
    N = len(lons)
    feat_dim = None
    out = None

    valid_mask = ~(np.isnan(lons) | np.isnan(lats))

    # First pass: figure out embedding dim from a dummy forward
    dummy = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    dummy_enc = enc.encode(dummy.clone())
    with torch.no_grad():
        dummy_emb = model(dummy_enc.to(device), return_feats=True)
    feat_dim = dummy_emb.shape[1]

    out = np.full((N, feat_dim), np.nan, dtype=np.float32)

    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return out

    valid_lons = lons[valid_idx].astype(np.float32)
    valid_lats = lats[valid_idx].astype(np.float32)

    results = []
    for start in range(0, len(valid_idx), batch_size):
        end = min(start + batch_size, len(valid_idx))
        batch_lons = valid_lons[start:end]
        batch_lats = valid_lats[start:end]

        locs = torch.from_numpy(np.stack([batch_lons, batch_lats], axis=1))
        locs_enc = enc.encode(locs.clone())  # clone because normalize_coords mutates in-place

        with torch.no_grad():
            emb = model(locs_enc.to(device), return_feats=True)

        results.append(emb.cpu().numpy())

    out[valid_idx] = np.concatenate(results, axis=0)
    return out


def main():
    parser = argparse.ArgumentParser(description="Extract SINR location embeddings")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to .pt checkpoint")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Torch device")
    parser.add_argument("--batch_size", type=int, default=4096)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--lon", type=float, help="Single longitude (WGS84)")
    mode.add_argument("--input", help="CSV with 'longitude' and 'latitude' columns")

    parser.add_argument("--lat", type=float, help="Single latitude (required with --lon)")
    parser.add_argument("--output", help="Output CSV path (required with --input)")
    parser.add_argument(
        "--lon_col", default="longitude", help="Column name for longitude in input CSV"
    )
    parser.add_argument(
        "--lat_col", default="latitude", help="Column name for latitude in input CSV"
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading model from {args.model}")
    model, enc, params = load_model(args.model, device)
    print(
        f"  Architecture : {params['model']} | depth={params['depth']} | num_filts={params['num_filts']}"
    )
    print(f"  Input encoding: {params['input_enc']}")
    print(f"  Embedding dim : {params['num_filts']}")
    print(f"  Num species   : {params['num_classes']:,}")

    # --- Single location mode ---
    if args.lon is not None:
        if args.lat is None:
            parser.error("--lat is required when using --lon")
        lons = np.array([args.lon], dtype=np.float32)
        lats = np.array([args.lat], dtype=np.float32)
        feats = extract_features(lons, lats, model, enc, device, args.batch_size)
        print(f"\nLocation: lon={args.lon:.6f}, lat={args.lat:.6f}")
        print(f"Embedding shape: {feats.shape}")
        print(f"Embedding (first 16 dims): {feats[0, :16]}")
        print(f"Embedding norm: {np.linalg.norm(feats[0]):.4f}")
        return

    # --- CSV batch mode ---
    if args.output is None:
        parser.error("--output is required when using --input")

    print(f"\nReading {args.input} ...")
    rows = []
    with open(args.input, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    print(f"  {len(rows):,} rows")

    lons = np.array(
        [float(r[args.lon_col]) if r[args.lon_col] else np.nan for r in rows], dtype=np.float32
    )
    lats = np.array(
        [float(r[args.lat_col]) if r[args.lat_col] else np.nan for r in rows], dtype=np.float32
    )

    valid = int((~np.isnan(lons) & ~np.isnan(lats)).sum())
    print(f"  {valid:,} rows with valid coordinates ({len(rows)-valid:,} will be NaN)")

    feats = extract_features(lons, lats, model, enc, device, args.batch_size)
    feat_dim = feats.shape[1]
    feat_cols = [f"sinr_{i:03d}" for i in range(feat_dim)]

    print(f"\nWriting {args.output} ...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_fieldnames = list(fieldnames) + feat_cols

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            for j, col in enumerate(feat_cols):
                v = feats[i, j]
                row[col] = "" if np.isnan(v) else f"{v:.6f}"
            writer.writerow(row)

    print(f"Done. {len(rows):,} rows × {feat_dim} SINR features → {args.output}")


if __name__ == "__main__":
    main()
