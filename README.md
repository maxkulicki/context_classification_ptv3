# PTv3 Tree Species Classification with Context Fusion

Point Transformer V3 (PTv3) for individual tree species classification from airborne LiDAR point clouds, fused with geospatial context features (satellite imagery embeddings, species distribution models, topography).

## Overview

Classifies tree species from individual tree point clouds using a PTv3 backbone, optionally fused with precomputed geospatial context features. Experiments use a standardized multi-dataset snapshot with a **dual-val** evaluation scheme: one in-distribution validation set (same forest districts as train) and one out-of-distribution validation set (held-out districts). A separate held-out **test** set is used for final evaluation only.

**10 target genera:** Abies, Acer, Alnus, Betula, Carpinus, Fagus, Larix, Picea, Pinus, Quercus

## Models

### Classifiers

| Model key | Class | Description |
|-----------|-------|-------------|
| `DefaultCls-v1m1` | — | Geometry-only (XYZ), no context |
| `CtxCls-v1m1` | `ContextFusionClassifier` | PTv3 → mean pool → concat with single encoded context → MLP head |
| `MultiCatCtxCls-v1m1` | `MultiCatCtxCls` | Late concat: PTv3 (512) + N context encoders → MLP head. No aux heads |
| `MultiCatCtxCls-v1m2` | `MultiCatCtxClsAux` | v1m1 + per-branch auxiliary classification heads (lidar, each ctx source). `aux_loss_weight` controls their contribution |
| `MultiCatCtxCls-v1m3` | `MultiCatCtxClsAuxModalityDrop` | v1m2 + per-sample modality dropout during training: one randomly chosen context source zeroed with probability `modality_dropout_p`. Aux heads always see full embeddings |
| `MultiXAttnCtxCls-v1m1` | `MultiXAttnCtxCls` | Cross-attention fusion: PTv3 features attend to each context embedding before concat |
| `MidFusionCtxCls-v1m1` | `MidFusionCtxCls` | AE embedding injected mid-backbone (after enc2 or enc3) via concat + linear projection |
| `UniversalCtxCls-v1m1` | `UniversalCtxCls` | Late concat supporting arbitrary combinations of context sources with optional aux heads and modality dropout |

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

Augmentations applied to context embeddings during training only; val/test always uses canonical embeddings.

| Transform | Applies to | Description |
|-----------|-----------|-------------|
| `CtxVMFAugment` | `ctx_ae` | von Mises–Fisher rotation noise on unit sphere (kappa=100 ≈ 8° deviation) |
| `CtxGaussianNoise` | `ctx_sinr` | Scale-relative isotropic Gaussian noise (sigma=0.1, p_drop=0.2) |
| `CtxPoolSample` | any pool key | Randomly sample one embedding from a precomputed pool of K augmented variants (p_drop=0.2) |

### Precomputed Augmentation Pools (AlphaEarth)

| File | Pool shape | Description |
|------|-----------|-------------|
| `context_features_temporal_aug.pth` | (9, 64) | One embedding per available year (2017–2025) at original location |
| `context_features_spatial_aug.pth` | (15, 64) | 5 spatial shifts × 3 distances (500m/1km/2km), each a random year |
| `context_features_combined_aug.pth` | (50, 64) | 10 spatial shifts × 5 random years per shift |

### Precomputed SINR Variants

| File | Description |
|------|-------------|
| `context_features.pth` | Canonical SINR 256-d features (standard spatial resolution) |
| `context_features_sinr_hc100.pth` | SINR features with HC-100 spatial resolution |
| `context_features_sinr_hc1000.pth` | SINR features with HC-1000 spatial resolution |
| `context_features_sinr_spatial_aug.pth` | SINR features at spatially shifted locations (augmentation pool) |

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
├── standardized_dataset_val_ood.txt
└── standardized_dataset_test.txt
```

### Context Features

All context features are precomputed and stored as a single lookup dict `{stem: {ctx_ae, ctx_topo, ctx_sinr, ctx_gpn, ...}}`:
```
context_classification_ptv3/data/snapshot_v1/
├── features.csv
├── context_features.pth                  # canonical AE + SINR + topo + GPN
├── context_features_temporal_aug.pth     # AE pool (9×64)
├── context_features_spatial_aug.pth      # AE pool (15×64)
├── context_features_combined_aug.pth     # AE pool (50×64)
├── context_features_sinr_hc100.pth       # SINR HC-100 variant
├── context_features_sinr_hc1000.pth      # SINR HC-1000 variant
└── context_features_sinr_spatial_aug.pth # SINR spatial-shift pool
```

## Setup

### Environment

```bash
# Main training environment (inside apptainer container ptv3.sif)
# Container handles all PTv3 / Flash Attention / spconv dependencies

# For preprocessing scripts (outside container):
conda run -n context_baseline python preprocess_context_features.py

# For UMAP visualizations:
conda env create -f Pointcept/env_umap_vis.yml
conda run -n umap_vis python Pointcept/tools/visualize_umap.py ...
```

## Training

All training runs via SLURM on 4×A100 GPUs using an apptainer container (`ptv3.sif`). Standard run: 120 epochs, ~5 hours.

```bash
cd context_classification_ptv3/sbatch_scripts
sbatch sbatch_ptv3_ctx_ae_sinr_cat_aux05_4gpu_120ep_10class_dual_val.sh
```

### Current Experiment Set (standardized 10-class dual-val, 4-GPU, 120ep)

| sbatch script | Model | Context | Augmentation | Notes |
|---------------|-------|---------|--------------|-------|
| `..._ptv3_small_4gpu_120ep_...` | `DefaultCls-v1m1` | — | — | geometry-only baseline |
| `..._ctx_ae_4gpu_100ep_...` | `CtxCls-v1m1` | AE | — | |
| `..._ctx_ae_4gpu_100ep_vmf_...` | `CtxCls-v1m1` | AE | vMF | |
| `..._ctx_ae_midfusion_...` | `MidFusionCtxCls-v1m1` | AE | — | inject after enc3 |
| `..._ctx_ae_midfusion_stage2_...` | `MidFusionCtxCls-v1m1` | AE | — | inject after enc2 |
| `..._ctx_ae_midfusion_vmf_...` | `MidFusionCtxCls-v1m1` | AE | vMF | |
| `..._ctx_ae_spatial_aug_...` | `CtxCls-v1m1` | AE | spatial pool | |
| `..._ctx_ae_temporal_aug_...` | `CtxCls-v1m1` | AE | temporal pool | |
| `..._ctx_ae_combined_aug_vmf_...` | `CtxCls-v1m1` | AE | combined pool + vMF | |
| `..._ctx_sinr_4gpu_200ep_...` | `CtxCls-v1m1` | SINR | — | |
| `..._ctx_sinr_gauss_...` | `CtxCls-v1m1` | SINR | Gaussian noise | |
| `..._ctx_sinr_hc100_gauss_...` | `CtxCls-v1m1` | SINR HC-100 | Gaussian noise | |
| `..._ctx_sinr_hc1000_gauss_...` | `CtxCls-v1m1` | SINR HC-1000 | Gaussian noise | |
| `..._ctx_sinr_spatial_aug_...` | `CtxCls-v1m1` | SINR | spatial pool | |
| `..._ctx_ae_sinr_cat_4gpu_120ep_...` | `MultiCatCtxCls-v1m1` | AE + SINR | vMF (AE) | no aux heads |
| `..._ctx_ae_sinr_cat_sinrgauss_...` | `MultiCatCtxCls-v1m1` | AE + SINR | vMF (AE) + Gauss (SINR) | no aux heads |
| `..._ctx_ae_sinr_cat_aux01_...` | `MultiCatCtxCls-v1m2` | AE + SINR | vMF (AE) | aux weight=0.1 |
| `..._ctx_ae_sinr_cat_aux05_...` | `MultiCatCtxCls-v1m2` | AE + SINR | vMF (AE) | aux weight=0.5 |
| `..._ctx_ae_sinr_cat_aux05_sinrgauss_...` | `MultiCatCtxCls-v1m2` | AE + SINR | vMF (AE) + Gauss (SINR) | aux weight=0.5 |
| `..._ctx_ae_sinr_cat_aux1_...` | `MultiCatCtxCls-v1m2` | AE + SINR | vMF (AE) | aux weight=1.0 |
| `..._ctx_ae_sinr_cat_aux_4gpu_200ep_...` | `MultiCatCtxCls-v1m2` | AE + SINR | vMF (AE) | aux weight=0.25, 200ep |
| `..._ctx_ae_sinr_cat_aux_mdrop_...` | `MultiCatCtxCls-v1m3` | AE + SINR | vMF (AE) | aux weight=0.25 + modality dropout p=0.25 |
| `..._ctx_ae_sinr_xattn_aux_...` | `MultiXAttnCtxCls-v1m1` | AE + SINR | vMF (AE) | cross-attention fusion |

### Training Internals

- Optimizer: AdamW (lr=0.004, weight\_decay=0.02)
- Scheduler: OneCycleLR (pct\_start=0.05, div\_factor=10, final\_div\_factor=1000)
- Separate lr for transformer blocks: 0.0004 (via `param_dicts`)
- Class-balanced loss: sqrt-inverse-frequency weights
- Evaluator: `DualValClsEvaluator` — logs val\_id and val\_ood metrics each epoch; saves `model_best_ood.pth` at the epoch with highest val\_ood allAcc

### WandB Metrics

Each run logs to `wandb_project=pointcept` under entity `makskulicki`. Key metrics per split:

- `loss`, `mIoU`, `mAcc`, `allAcc`, `macro_f1`, `weighted_f1`
- Per-class: `f1_{Genus}`, `acc_{Genus}`
- Per-dataset: `ds_acc_{Dataset}`
- Aux branch (v1m2/v1m3): `loss_aux_{lidar,ae,sinr}`, `mAcc_aux_{...}`, `allAcc_aux_{...}`
- Confusion matrix and per-dataset bar chart images

**Recommended paper metrics:** Macro F1 (balances precision + recall across classes), allAcc (overall accuracy), and per-class F1. mAcc (macro recall) is complementary but insensitive to false positives.

## Evaluation

### Val + Test Evaluation

```bash
# Evaluate best checkpoint on the held-out test set (runs inside apptainer)
sbatch sbatch_scripts/sbatch_eval_test.sh           # single model
sbatch sbatch_scripts/sbatch_eval_test_multi.sh     # multiple models in sequence

# Generates: Pointcept/eval_results/{exp_name}/metrics.json, confusion_matrix.png, predictions.csv
```

### Visualization

```bash
# Compare test results across models (per-genus F1/acc, per-dataset acc, heatmaps)
conda run -n context_baseline python visualize_results.py

# Compare val_id / val_ood / test at best_ood epoch across models (side-by-side splits)
conda run -n context_baseline python visualize_results_splits.py

# GradCAM branch attribution analysis
conda run -n context_baseline python gradcam_analysis/plot_attribution.py

# UMAP embeddings (requires umap_vis env)
conda run -n umap_vis python Pointcept/tools/visualize_umap.py \
    --exp_dir Pointcept/exp/snapshot_10class_dual_val/ptv3_ctx_ae_sinr_cat_aux_4gpu_200ep \
    --tag ae_sinr_best_umap
```

Output directories:
- `Pointcept/eval_results/` — per-model test metrics and confusion matrices
- `Pointcept/eval_results/comparison/` — cross-model comparison plots (test)
- `Pointcept/eval_results/comparison_splits/` — val\_id / val\_ood / test side-by-side plots
- `Pointcept/umap_vis/` — UMAP embedding visualizations

## Project Structure

```
context_classification_ptv3/
├── Pointcept/
│   ├── configs/standardized_dataset/    # one .py config per experiment
│   ├── eval_results/                    # test evaluation outputs + comparison plots
│   ├── umap_vis/                        # UMAP visualizations
│   ├── env_umap_vis.yml                 # conda env for UMAP/visualization
│   ├── eval_test.py                     # standalone test evaluator
│   ├── pointcept/
│   │   ├── datasets/
│   │   │   ├── standardized_dataset.py  # StandardizedDataset + context injection
│   │   │   └── transform.py             # CtxVMFAugment, CtxGaussianNoise, CtxPoolSample
│   │   └── models/tree_context/
│   │       └── classifier.py            # all classifier models:
│   │                                    #   CtxCls-v1m1, MultiCatCtxCls-v1m{1,2,3},
│   │                                    #   MultiXAttnCtxCls-v1m1, MidFusionCtxCls-v1m1,
│   │                                    #   UniversalCtxCls-v1m1
│   └── tools/
│       ├── train.py / test.py
│       ├── gradcam_attribution_3branch.py
│       └── visualize_umap.py
├── sbatch_scripts/                      # SLURM job scripts (4×A100, 5h)
├── sinr/                                # SINR model + feature extractor
├── gradcam_analysis/                    # GradCAM CSVs and attribution plots
├── feature_visualization/               # PCA/UMAP plots of raw context features
├── results/                             # context-only baseline results
├── data/
│   ├── all_trees_unified.csv
│   └── snapshot_v1/
│       ├── features.csv
│       ├── context_features.pth
│       ├── context_features_temporal_aug.pth
│       ├── context_features_spatial_aug.pth
│       ├── context_features_combined_aug.pth
│       ├── context_features_sinr_hc100.pth
│       ├── context_features_sinr_hc1000.pth
│       └── context_features_sinr_spatial_aug.pth
├── visualize_results.py                 # test set cross-model comparison
├── visualize_results_splits.py          # val_id / val_ood / test comparison at best_ood epoch
├── build_standardized_dataset.py        # build npy dataset from LAZ sources
├── convert_laz_to_npy_normals.py
├── create_test_split_txt.py
├── fetch_alphaearth_embeddings.py
├── fetch_alphaearth_embeddings_all_years.py
├── fetch_alphaearth_embeddings_augmented.py
├── fetch_alphaearth_embeddings_combined.py
├── fetch_alphaearth_embeddings_expanded.py
├── preprocess_context_features.py
├── preprocess_context_features_temporal.py
├── preprocess_context_features_spatial.py
├── preprocess_context_features_combined_aug.py
├── extract_sinr_features.py
├── extract_sinr_features_spatial_aug.py
├── preprocess_context_features_sinr_hc100.py
├── preprocess_context_features_sinr_hc1000.py
├── preprocess_context_features_sinr_spatial_aug.py
└── extract_topo_features.py
```

## Acknowledgements

Built on the [Pointcept](https://github.com/Pointcept/Pointcept) framework (MIT License).
AlphaEarth embeddings from [Google Satellite Embedding V1 Annual](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL) via Google Earth Engine.
SINR model from [Mac Aodha et al. 2024](https://github.com/elijahcole/sinr).
