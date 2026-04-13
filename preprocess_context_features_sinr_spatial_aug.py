"""
Build context_features_sinr_spatial_aug.pth from:
  - data/snapshot_v1/features.csv          — AE, topo, GPN columns
  - data/snapshot_v1/context_features.pth  — canonical ctx_sinr (distilled_from_env)
  - data/snapshot_v1/sinr_spatial_aug_pool.pt — pool of 24 shifted embeddings per tree,
      produced by extract_sinr_features_spatial_aug.py

Run extraction first (from context_classification_ptv3/):
    conda run -n context_baseline python extract_sinr_features_spatial_aug.py \
        --input  data/snapshot_v1/split.csv \
        --output data/snapshot_v1/sinr_spatial_aug_pool.pt \
        --model  sinr/pretrained_models/model_an_full_input_enc_sin_cos_distilled_from_env.pt

    Or via apptainer (if conda env is unavailable):
    apptainer exec --nv \
        --env PYTHONPATH=/net/pr2/projects/plgrid/plggtreeseg/context_classification_ptv3/Pointcept \
        /net/pr2/projects/plgrid/plggtreeseg/ptv3.sif \
        python /net/pr2/projects/plgrid/plggtreeseg/context_classification_ptv3/extract_sinr_features_spatial_aug.py \
            --input  /net/pr2/projects/plgrid/plggtreeseg/context_classification_ptv3/data/snapshot_v1/split.csv \
            --output /net/pr2/projects/plgrid/plggtreeseg/context_classification_ptv3/data/snapshot_v1/sinr_spatial_aug_pool.pt \
            --model  /net/pr2/projects/plgrid/plggtreeseg/context_classification_ptv3/sinr/pretrained_models/model_an_full_input_enc_sin_cos_distilled_from_env.pt

Then build the .pth (from context_classification_ptv3/):
    conda run -n context_baseline python preprocess_context_features_sinr_spatial_aug.py

    Or via apptainer:
    apptainer exec --nv \
        /net/pr2/projects/plgrid/plggtreeseg/ptv3.sif \
        python /net/pr2/projects/plgrid/plggtreeseg/context_classification_ptv3/preprocess_context_features_sinr_spatial_aug.py

Output:
    data/snapshot_v1/context_features_sinr_spatial_aug.pth

Keys per stem:
    ctx_ae        — (64,)   AlphaEarth embedding
    ctx_topo      — (6,)    topographic variables
    ctx_sinr      — (256,)  canonical SINR embedding (original coordinates)
    ctx_gpn       — (18,)   GeoPlantNet logits (NaN → 0)
    ctx_sinr_pool — (24, 256) spatially-shifted SINR embeddings
                              [0:8]  10 km shifts, 8 random bearings
                              [8:16] 20 km shifts, same bearings
                              [16:24] 50 km shifts, same bearings
"""

import os
import csv
import numpy as np
import torch

FEATURES_CSV   = "data/snapshot_v1/features.csv"
CANONICAL_PTH  = "data/snapshot_v1/context_features.pth"
POOL_PT        = "data/snapshot_v1/sinr_spatial_aug_pool.pt"
OUT_PATH       = "data/snapshot_v1/context_features_sinr_spatial_aug.pth"

_META_COLS = {
    "dataset", "tree_id", "genus", "species", "latitude", "longitude",
    "split", "laz_path", "elevation", "slope", "northness", "eastness",
    "tri", "tpi",
}
TOPO_COLS = ["elevation", "slope", "northness", "eastness", "tri", "tpi"]

print(f"Loading features.csv: {FEATURES_CSV}")
rows = []
with open(FEATURES_CSV, newline="") as f:
    reader = csv.DictReader(f)
    cols = reader.fieldnames
    ae_cols  = [c for c in cols if c.startswith("A") and c[1:].isdigit()]
    gpn_cols = [c for c in cols if c not in _META_COLS
                and not c.startswith("A") and not c.startswith("sinr_")]
    for row in reader:
        rows.append(row)
print(f"  {len(rows):,} rows  |  AE: {len(ae_cols)}  Topo: {len(TOPO_COLS)}  GPN: {len(gpn_cols)}")

print(f"Loading canonical SINR: {CANONICAL_PTH}")
canonical = torch.load(CANONICAL_PTH, weights_only=False)
print(f"  {len(canonical):,} entries")

print(f"Loading spatial aug pool: {POOL_PT}")
pool = torch.load(POOL_PT, weights_only=False)
sample_pool = next(iter(pool.values()))
print(f"  {len(pool):,} entries  |  pool shape per tree: {sample_pool.shape}")

missing_canonical = 0
missing_pool = 0

lookup = {}
n = len(rows)
for i, row in enumerate(rows):
    if i % 1000 == 0:
        print(f"  Building lookup: {i:,}/{n:,}", end="\r", flush=True)

    stem = os.path.splitext(os.path.basename(row["laz_path"]))[0]

    sinr_vec = canonical.get(stem, {}).get("ctx_sinr")
    if sinr_vec is None:
        missing_canonical += 1
        sinr_vec = np.zeros(256, dtype=np.float32)

    sinr_pool = pool.get(stem)
    if sinr_pool is None:
        missing_pool += 1
        sinr_pool = np.zeros((sample_pool.shape[0], sample_pool.shape[1]), dtype=np.float32)

    def _parse(keys):
        return np.array([float(row[c]) if row[c] else 0.0 for c in keys], dtype=np.float32)

    gpn_raw = np.array(
        [float(row[c]) if row[c] else float("nan") for c in gpn_cols], dtype=np.float64
    )

    lookup[stem] = {
        "ctx_ae":        _parse(ae_cols),
        "ctx_topo":      _parse(TOPO_COLS),
        "ctx_sinr":      sinr_vec,
        "ctx_gpn":       np.nan_to_num(gpn_raw, nan=0.0).astype(np.float32),
        "ctx_sinr_pool": sinr_pool,
    }

print(f"  Building lookup: {n:,}/{n:,}")
print(f"Built {len(lookup):,} entries")
if missing_canonical:
    print(f"WARNING: {missing_canonical} trees missing canonical SINR — zeroed")
if missing_pool:
    print(f"WARNING: {missing_pool} trees missing pool — zeroed")

print(f"Saving to {OUT_PATH} ...")
torch.save(lookup, OUT_PATH)
print(f"Done.")

sample_stem, sample = next(iter(lookup.items()))
print(f"\nSample stem: {sample_stem}")
for k, v in sample.items():
    print(f"  {k}: shape={v.shape}  dtype={v.dtype}  min={v.min():.3f}  max={v.max():.3f}")
