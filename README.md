# PTv3 Tree Species Classification with Context Fusion

Point Transformer V3 (PTv3) for individual tree species classification from airborne LiDAR point clouds, with optional geospatial context fusion (satellite imagery embeddings, topography, species distribution models).

## Overview

Classifies tree species from individual tree point clouds using a PTv3 backbone, optionally fused with precomputed geospatial context features. Experiments use a standardized multi-dataset snapshot with a **dual-val** evaluation scheme: one in-distribution validation set (same forest districts as train) and one out-of-distribution validation set (held-out districts).

**10 target genera:** Abies, Acer, Alnus, Betula, Carpinus, Fagus, Larix, Picea, Pinus, Quercus

## Models

### Classifiers

| Model | Key | Description |
|-------|-----|-------------|
| **PTv3 baseline** | `DefaultCls-v1m1` | Geometry-only (XYZ), no context |
| **Late-concat context** | `CtxCls-v1m1` | PTv3 backbone → mean pool → concat with encoded context → MLP head |
| **Mid-fusion context** | `MidFusionCtxCls-v1m1` | AE embedding injected mid-backbone (after enc2 or enc3) via concat + Linear projection |
| **Universal multi-source** | `UniversalCtxCls-v1m1` | Late concat supporting arbitrary combinations of context sources |

### Context Encoders

| Encoder | Key | Input | Output |
|---------|-----|-------|--------|
| `AEEncoder` | `ctx_ae` | AlphaEarth 64-d raw embedding | 256-d |
| `SINREncoder` | `ctx_sinr` | SINR precomputed 256-d features | 256-d |
| `TopoEncoder` | `ctx_topo` | 6 topographic variables | configurable |

### Context Sources

| Source | Dim | Description |
|--------|-----|-------------|
| **AlphaEarth** | 64 | Google Satellite Embedding V1 Annual — annual 10m satellite composite |
| **SINR** | 256 | Species distribution model backbone features (ResidualFCNet) |
| **Topo** | 6 | Elevation, slope, northness, eastness, TRI, TPI |
| **GeoPlantNet** | 18 | Species logit scores |

## Context Augmentation

Augmentations applied to context embeddings during training only; val always uses canonical embeddings.

| Transform | Applies to | Description |
|-----------|-----------|-------------|
| `CtxVMFAugment` | `ctx_ae` | von Mises–Fisher rotation noise on unit sphere (kappa=100 ≈ 8° deviation) |
| `CtxGaussianNoise` | `ctx_sinr` | Scale-relative isotropic Gaussian noise (sigma=0.1, p_drop=0.2) |
| `CtxPoolSample` | any pool key | Randomly sample one embedding from a precomputed pool of K augmented variants (p_drop=0.2) |

### Precomputed Augmentation Pools

Three augmented context `.pth` files extend the canonical `context_features.pth`:

| File | Pool shape | Description |
|------|-----------|-------------|
| `context_features_temporal_aug.pth` | (9, 64) | One embedding per available year (2017–2025) at the original location |
| `context_features_spatial_aug.pth` | (15, 64) | 5 spatial shifts × 3 distances (500m/1km/2km), each a random year |
| `context_features_combined_aug.pth` | (50, 64) | 10 spatial shifts × 5 random years per shift |

## Data

### Point Clouds

Individual tree point clouds stored as `.npy` files (N×3, XYZ only), organised by genus:
```
data/snapshot_10class_dual_val_npy/
├── Abies/      *.npy
├── Acer/       *.npy
├── ...
├── Quercus/    *.npy
├── standardized_dataset_train.txt
├── standardized_dataset_val_id.txt
└── standardized_dataset_val_ood.txt
```

### Context Features

All context features are precomputed and stored in a single lookup:
```
context_classification_ptv3/data/snapshot_v1/
├── features.csv                          # raw features per tree (AE, topo, SINR, GPN)
├── context_features.pth                  # {stem: {ctx_ae, ctx_topo, ctx_sinr, ctx_gpn}}
├── context_features_temporal_aug.pth     # {stem: {ctx_ae, ctx_ae_pool (9×64)}}
├── context_features_spatial_aug.pth      # {stem: {ctx_ae, ctx_ae_pool (15×64)}}
└── context_features_combined_aug.pth     # {stem: {ctx_ae, ctx_ae_pool (50×64)}}
```

## Setup

### Environment

```bash
# Main training environment (inside apptainer container ptv3.sif)
# Container handles all PTv3 dependencies

# For preprocessing scripts (outside container):
conda run -n context_baseline python preprocess_context_features.py
```

### AlphaEarth Data Pipeline

```bash
# 1. Fetch all available years per location (GEE — run on device with GEE auth)
python fetch_alphaearth_embeddings_all_years.py --project my-gcp-project
# → data/trees_alphaearth_all_years.csv

# 2. Fetch spatial shifts (500m/1km/2km, 5 samples each, random year)
python fetch_alphaearth_embeddings_augmented.py --project my-gcp-project
# → data/trees_alphaearth_augmented.csv

# 3. Fetch combined spatial+temporal (10 shifts × 5 years each)
python fetch_alphaearth_embeddings_combined.py --project my-gcp-project
# → data/trees_alphaearth_combined_aug.csv

# 4. Build .pth lookup files (run on HPC after copying CSVs)
conda run -n context_baseline python preprocess_context_features.py
conda run -n context_baseline python preprocess_context_features_temporal.py
conda run -n context_baseline python preprocess_context_features_spatial.py
conda run -n context_baseline python preprocess_context_features_combined_aug.py
```

GEE authentication: `earthengine authenticate --auth_mode=notebook` (headless). Credentials cached at `~/.config/earthengine/credentials`.

## Training

All training runs via SLURM on 4×A100 GPUs using an apptainer container (`ptv3.sif`). Standard run: 120 epochs, ~5 hours.

```bash
cd context_classification_ptv3/sbatch_scripts
sbatch sbatch_ptv3_ctx_ae_4gpu_120ep_10class_dual_val.sh
```

### Available Experiments

| sbatch script | Model | Context | Augmentation |
|---------------|-------|---------|--------------|
| `..._ptv3_4gpu_120ep_...` | baseline | — | — |
| `..._ctx_ae_4gpu_120ep_...` | late concat | AE | — |
| `..._ctx_ae_vmf_...` | late concat | AE | vMF |
| `..._ctx_ae_midfusion_...` | mid-fusion enc3 | AE | — |
| `..._ctx_ae_midfusion_vmf_...` | mid-fusion enc3 | AE | vMF |
| `..._ctx_ae_midfusion_stage2_...` | mid-fusion enc2 | AE | — |
| `..._ctx_ae_spatial_aug_...` | late concat | AE | spatial pool |
| `..._ctx_ae_temporal_aug_...` | late concat | AE | temporal pool |
| `..._ctx_ae_combined_aug_vmf_...` | late concat | AE | combined pool + vMF |
| `..._ctx_sinr_4gpu_...` | late concat | SINR | — |
| `..._ctx_sinr_gauss_...` | late concat | SINR | Gaussian noise |
| `..._ctx_ae_sinr_cat_...` | late concat | AE + SINR | — |

### Training Internals

- Optimizer: AdamW (lr=0.004, weight\_decay=0.02)
- Scheduler: OneCycleLR (pct\_start=0.05, div\_factor=10, final\_div\_factor=1000)
- Separate lr for transformer blocks: 0.0004
- Class-balanced loss: sqrt-inverse-frequency weights
- Evaluator: `DualValClsEvaluator` — reports ID and OOD accuracy each epoch

## Evaluation

```bash
# Summarise results across all experiments
conda run -n context_baseline python tools/summarize_dual_val_results.py

# Plot ID vs OOD accuracy
conda run -n context_baseline python tools/plot_dual_val_results.py

# GradCAM attribution analysis
conda run -n context_baseline python Pointcept/tools/gradcam_attribution_3branch.py
```

## Project Structure

```
context_classification_ptv3/
├── Pointcept/
│   ├── configs/standardized_dataset/    # one config per experiment
│   ├── pointcept/
│   │   ├── datasets/
│   │   │   ├── standardized_dataset.py  # StandardizedDataset + context injection
│   │   │   └── transform.py             # CtxVMFAugment, CtxGaussianNoise, CtxPoolSample
│   │   └── models/tree_context/
│   │       └── classifier.py            # CtxCls, MidFusionCtxCls, UniversalCtxCls
│   └── tools/
│       ├── train.py / test.py
│       ├── gradcam_attribution_3branch.py
│       ├── plot_dual_val_results.py
│       └── summarize_dual_val_extended.py
├── sbatch_scripts/                      # SLURM job scripts (4×A100, 5h)
├── sinr/                                # SINR model + feature extractor
├── data/
│   ├── all_trees_unified.csv            # master tree list with coordinates
│   ├── trees_alphaearth_all_years.csv   # AE embeddings per year
│   ├── trees_alphaearth_augmented.csv   # spatially shifted AE embeddings
│   ├── trees_alphaearth_combined_aug.csv# combined spatial+temporal AE
│   └── snapshot_v1/
│       ├── features.csv
│       ├── context_features.pth
│       ├── context_features_temporal_aug.pth
│       ├── context_features_spatial_aug.pth
│       └── context_features_combined_aug.pth
├── fetch_alphaearth_embeddings.py          # single-year AE fetch
├── fetch_alphaearth_embeddings_all_years.py
├── fetch_alphaearth_embeddings_augmented.py
├── fetch_alphaearth_embeddings_combined.py
├── preprocess_context_features.py
├── preprocess_context_features_temporal.py
├── preprocess_context_features_spatial.py
├── preprocess_context_features_combined_aug.py
├── extract_sinr_features.py
├── extract_topo_features.py
└── tools/
    ├── summarize_dual_val_results.py
    └── plot_dual_val_results.py
```

## Acknowledgements

Built on the [Pointcept](https://github.com/Pointcept/Pointcept) framework (MIT License).
AlphaEarth embeddings from [Google Satellite Embedding V1 Annual](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL) via Google Earth Engine.
SINR model from [Mac Aodha et al. 2024](https://github.com/elijahcole/sinr).
