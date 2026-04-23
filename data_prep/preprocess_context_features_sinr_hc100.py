"""
Build context_features_sinr_hc100.pth from:
  - data/snapshot_v1/features.csv        — AE, topo, GPN columns
  - data/snapshot_v1/sinr_features_hc100.csv — SINR cols extracted with
      model_an_full_input_enc_sin_cos_hard_cap_num_per_class_100.pt

Run extraction first (from context_classification_ptv3/):
    conda run -n context_baseline python extract_sinr_features.py \
        --input data/snapshot_v1/split.csv \
        --output data/snapshot_v1/sinr_features_hc100.csv \
        --model sinr/pretrained_models/model_an_full_input_enc_sin_cos_hard_cap_num_per_class_100.pt

Then build the .pth (from context_classification_ptv3/):
    conda run -n context_baseline python preprocess_context_features_sinr_hc100.py

Output:
    data/snapshot_v1/context_features_sinr_hc100.pth
"""

import os
import numpy as np
import torch
import pandas as pd

FEATURES_CSV    = "data/snapshot_v1/features.csv"
SINR_HC100_CSV  = "data/snapshot_v1/sinr_features_hc100.csv"
OUT_PATH        = "data/snapshot_v1/context_features_sinr_hc100.pth"

df_feat = pd.read_csv(FEATURES_CSV)
df_sinr = pd.read_csv(SINR_HC100_CSV)

cols_feat = list(df_feat.columns)
cols_sinr = list(df_sinr.columns)

ae_cols   = [c for c in cols_feat if c.startswith("A") and c[1:].isdigit()]
topo_cols = ["elevation", "slope", "northness", "eastness", "tri", "tpi"]
gpn_cols  = [c for c in cols_feat if c not in {
    "dataset", "tree_id", "genus", "species", "latitude", "longitude",
    "split", "laz_path", "elevation", "slope", "northness", "eastness",
    "tri", "tpi",
} and not c.startswith("A") and not c.startswith("sinr_")]
sinr_cols = [c for c in cols_sinr if c.startswith("sinr_")]

print(f"AE:   {len(ae_cols)} cols")
print(f"Topo: {len(topo_cols)} cols")
print(f"SINR (hc100): {len(sinr_cols)} cols")
print(f"GPN:  {len(gpn_cols)} cols")
print(f"features.csv rows: {len(df_feat)}")
print(f"sinr_hc100 rows:   {len(df_sinr)}")

# Build stem→sinr lookup from hc100 CSV
sinr_by_stem = {}
for _, row in df_sinr.iterrows():
    stem = os.path.splitext(os.path.basename(str(row["laz_path"])))[0]
    sinr_by_stem[stem] = row[sinr_cols].values.astype(np.float32)

lookup = {}
missing_sinr = 0
for _, row in df_feat.iterrows():
    stem = os.path.splitext(os.path.basename(str(row["laz_path"])))[0]
    sinr_vec = sinr_by_stem.get(stem)
    if sinr_vec is None:
        missing_sinr += 1
        sinr_vec = np.zeros(len(sinr_cols), dtype=np.float32)
    lookup[stem] = {
        "ctx_ae":   row[ae_cols].values.astype(np.float32),
        "ctx_topo": row[topo_cols].values.astype(np.float32),
        "ctx_sinr": sinr_vec,
        "ctx_gpn":  np.nan_to_num(
            row[gpn_cols].values.astype(np.float64), nan=0.0
        ).astype(np.float32),
    }

if missing_sinr:
    print(f"WARNING: {missing_sinr} trees had no hc100 SINR entry — zeroed out")

torch.save(lookup, OUT_PATH)
print(f"Saved {len(lookup)} entries to {OUT_PATH}")

# Sanity check
sample_stem, sample = next(iter(lookup.items()))
print(f"\nSample stem: {sample_stem}")
for k, v in sample.items():
    print(f"  {k}: shape={v.shape}  dtype={v.dtype}  "
          f"min={v.min():.3f}  max={v.max():.3f}")
