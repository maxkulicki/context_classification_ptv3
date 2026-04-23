# Guide: PTv3 + AlphaEarth Integration

## 1. Extracting AE Embeddings for a Location

All extraction scripts are in `data_prep/`. They call the Google Earth Engine API (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`) and output CSVs with 64-dim embeddings (columns `A00`–`A63`).

| Script | What it does |
|--------|-------------|
| `fetch_alphaearth_embeddings.py` | Baseline: one embedding per plot location (PUWG92→WGS84 reprojection). Output: 9 plot attrs + 64 dims. |
| `fetch_alphaearth_embeddings_augmented.py` | Spatial augmentation: 15 samples per location (5×500m + 5×1000m + 5×2000m random offsets), each from a random year. |
| `fetch_alphaearth_embeddings_combined.py` | Spatial + temporal: `n_shifts` offsets × `n_years` random years per location (default: 10×5). |
| `fetch_alphaearth_embeddings_all_years.py` | Batch fetch across all years. |

After fetching, run the corresponding preprocessing script to build the `.pth` file the dataset loader expects:

| Script | Output |
|--------|--------|
| `preprocess_context_features_combined_aug.py` | `context_features_combined_aug.pth`: `{stem: {"ctx_ae": (64,), "ctx_ae_pool": (S×T, 64)}}` |
| `preprocess_context_features_spatial.py` | Spatial-only augmentation pool. |
| `preprocess_context_features_temporal.py` | Temporal-only augmentation pool. |

---

## 2. Model Architecture Files

All architecture lives under `Pointcept/pointcept/models/`.

### PTv3 Backbone

`Pointcept/pointcept/models/point_transformer_v3/point_transformer_v3m1_base.py`

Encoder-only PTv3 with serialized attention and Hilbert ordering. 5 encoder stages, channels `(32, 64, 128, 256, 512)`. Point-wise features are mean-pooled to `(B, 512)` for classification.

### AE Encoder

`Pointcept/pointcept/models/tree_context/encoders.py`

- **`AlphaEarthEncoder`**: `Linear(64→256) + LayerNorm` — projects 64-dim AE embedding to a 256-dim context token.

### Classifier combining PTv3 + AE

`Pointcept/pointcept/models/tree_context/classifier.py`

| Class | Registered as | Fusion |
|-------|--------------|--------|
| `ContextFusionClassifier` | `CtxCls-v1m1` | PTv3(512) + AEEncoder(256) → concat(768) → Linear → BN → ReLU → Dropout(0.5) → Linear(→classes) |

---

## 3. Custom AE Augmentations

Augmentation happens at two levels:

**At fetch time** (the scripts above): spatial offsets (shifting the query location by 500m/1km/2km) and temporal sampling (random year selection from available annual embeddings).

**At training time**, a vMF (von Mises-Fisher) perturbation adds noise on the unit hypersphere to the AE embedding. See the `vmf` config variants listed below.

The `ctx_ae_pool` key (shape `(K, 64)`) is the augmented pool stored per stem — at training time the dataset randomly samples one vector from this pool instead of the canonical embedding.

---

## 4. Config Files

All configs are in `Pointcept/configs/standardized_dataset/`.

**Baseline (no context):**
- `cls-ptv3-baseline-genus-small.py` — pure PTv3, useful as reference

**Single-source AE:**
- `cls-ptv3-ctx-ae-10class-dual-val-4gpu.py` — canonical AE, `CtxCls-v1m1`, recommended starting point
- `cls-ptv3-ctx-ae-spatial-aug-10class-dual-val-4gpu.py` — AE with spatial aug pool
- `cls-ptv3-ctx-ae-temporal-aug-10class-dual-val-4gpu.py` — AE with temporal aug pool
- `cls-ptv3-ctx-ae-combined-aug-vmf-10class-dual-val-4gpu.py` — spatial+temporal aug + vMF perturbation

---

## 5. Dataset Loader

`Pointcept/pointcept/datasets/standardized_dataset.py`

Loads `context_features.pth`, looks up `ctx_ae` (or `ctx_ae_pool`) per stem, and injects it into the data dict for each sample. The config controls this via `context_pth` and `context_sources=["alphaearth"]`.

---

## Summary: AlphaEarth Integration Flow

```
Google Earth Engine (GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL)
         ↓  data_prep/fetch_alphaearth_embeddings.py
     64-dim embeddings per location (CSV)
         ↓  data_prep/fetch_alphaearth_embeddings_combined.py
     Spatial & temporal augmentations (CSV)
         ↓  data_prep/preprocess_context_features_combined_aug.py
     context_features.pth: {stem: {"ctx_ae": (64,), "ctx_ae_pool": (K, 64)}}
         ↓  StandardizedDataset
     Data dict with "ctx_ae" key injected per sample
         ↓  AlphaEarthEncoder: 64 → 256
     256-dim context embedding
         ↓  CtxCls-v1m1
     Fused with PTv3 point cloud features (512) → concat(768) → classification
```

---

## Minimal Files to Reuse This Architecture

1. `data_prep/fetch_alphaearth_embeddings.py` + `data_prep/fetch_alphaearth_embeddings_combined.py`
2. `data_prep/preprocess_context_features_combined_aug.py`
3. `Pointcept/pointcept/models/tree_context/encoders.py` (just `AlphaEarthEncoder`)
4. `Pointcept/pointcept/models/tree_context/classifier.py` (just `CtxCls-v1m1`)
5. `Pointcept/pointcept/datasets/standardized_dataset.py` (context-injection logic)
6. `Pointcept/configs/standardized_dataset/cls-ptv3-ctx-ae-10class-dual-val-4gpu.py` — clean config template
