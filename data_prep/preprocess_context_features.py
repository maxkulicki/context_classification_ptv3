"""
Preprocess context features CSV into a .pth lookup dict.

Converts the features.csv into a dict:
    {filename_stem: {"ctx_ae": np.float32, "ctx_topo": ..., "ctx_sinr": ..., "ctx_gpn": ...}}
and saves it as a .pth file that the dataset can load without pandas.

Usage (from context_classification_ptv3/):
    conda run -n context_baseline python preprocess_context_features.py

Output:
    data/snapshot_v1/context_features.pth
"""

import os
import numpy as np
import torch
import pandas as pd

CSV_PATH = "data/snapshot_v1/features.csv"
OUT_PATH = "data/snapshot_v1/context_features.pth"

df = pd.read_csv(CSV_PATH)
cols = list(df.columns)

ae_cols   = [c for c in cols if c.startswith("A") and c[1:].isdigit()]
topo_cols = ["elevation", "slope", "northness", "eastness", "tri", "tpi"]
sinr_cols = [c for c in cols if c.startswith("sinr_")]
gpn_cols  = [c for c in cols if c not in {
    "dataset", "tree_id", "genus", "species", "latitude", "longitude",
    "split", "laz_path", "elevation", "slope", "northness", "eastness",
    "tri", "tpi",
} and not c.startswith("A") and not c.startswith("sinr_")]

print(f"AE:   {len(ae_cols)} cols")
print(f"Topo: {len(topo_cols)} cols")
print(f"SINR: {len(sinr_cols)} cols")
print(f"GPN:  {len(gpn_cols)} cols")
print(f"Total rows: {len(df)}")

lookup = {}
for _, row in df.iterrows():
    stem = os.path.splitext(os.path.basename(str(row["laz_path"])))[0]
    lookup[stem] = {
        "ctx_ae":   row[ae_cols].values.astype(np.float32),
        "ctx_topo": row[topo_cols].values.astype(np.float32),
        "ctx_sinr": row[sinr_cols].values.astype(np.float32),
        "ctx_gpn":  np.nan_to_num(
            row[gpn_cols].values.astype(np.float64), nan=0.0
        ).astype(np.float32),
    }

torch.save(lookup, OUT_PATH)
print(f"Saved {len(lookup)} entries to {OUT_PATH}")

# Quick sanity check
sample_stem, sample = next(iter(lookup.items()))
print(f"\nSample stem: {sample_stem}")
for k, v in sample.items():
    print(f"  {k}: shape={v.shape}  dtype={v.dtype}  "
          f"min={v.min():.3f}  max={v.max():.3f}")
