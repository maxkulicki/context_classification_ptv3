# Experiment Plan: Multimodal Tree Species Classification from TLS Point Clouds

**Author:** Maks
**Date:** March 2026
**Context:** Multi-dataset, multi-country tree species classification using PTv3 backbone with contextual features and inter-tree attention

---

## 1. Overview

This plan evaluates multimodal fusion strategies for tree species classification from terrestrial laser scanning (TLS) data. The core architecture is a Point Transformer v3 (PTv3) backbone operating on individual tree point clouds, augmented with contextual features derived from remote sensing, species distribution models, and topography.

The key research questions, in order:

1. **Location prior:** How well do context features alone (no point cloud) predict species identity? Quantifies what location encodes before any geometry is involved.
2. **Marginal source value:** Which individual context sources improve classification when fused with PTv3?
3. **Universal model:** Does a single model trained with source dropout over all sources and a CLS-attention context encoder match or exceed individually-tuned models, and which source combinations drive performance?
4. **Fusion depth:** Does mid-fusion (FiLM conditioning) outperform late fusion for the best context configuration?
5. **Inter-tree attention:** Does neighborhood-level reasoning improve classification, especially for ambiguous individuals?

The plan is structured in sequential phases. Each phase resolves one question before the next begins.

---

## 2. Datasets

### 2.1 Overview

| Dataset | Country | Plots | Trees | Species | Scanner | Georef. |
|---------|---------|------:|------:|--------:|---------|---------|
| **TreeScanPL** | Poland | 271 | 6,845 | 18 | TLS (Riegl VZ-400i) | Yes |
| **BioDiv-3DTrees** | Germany | 27 | 4,952 | 19 | TLS + ULS | Sensitive† |
| **LAUTx** | Austria | 6 | 434 | 6 | PLS (handheld) | Yes |
| **Weiser et al.** | Germany | 12 | 264 | 12 | TLS (Riegl VZ-400) | Yes |
| **NIBIO** | Norway | 20 | 481 | 3 | ULS | Yes |
| **CULS** | Czech Republic | — | 50 | 1 | ULS | Yes |
| **Frey 2022** | Germany | 15 | 472 | 6 | TLS | Yes |
| **Junttila/Yrttimaa** | Finland | 20 | 51 | 1 | ULS | Yes |
| **Puliti MLS** | Italy | 1 | 67 | 1 | MLS | Approx. |
| **Puliti ULS 2** | Norway/Finland | — | 621 | 3 | ULS | Yes |
| **Saarinen 2021** | Finland | 10 | 1,976 | 1 | MLS | Yes |
| **Wytham Woods** | UK | 1 | 769 | 6 | TLS | Approx. |
| **Total** | 9 countries | 383+ | 16,982 | — | | |

†BioDiv-3DTrees plot coordinates must be requested through BExIS (sensitive biodiversity data), which affects availability of geolocation-dependent context features (AlphaEarth, SINR, GeoPlantNet).

The collection spans 16,982 trees across 9 countries. Dominant species cross-dataset: *Pinus sylvestris* (6,495 trees, 8 datasets), *Fagus sylvatica* (3,789, 4 datasets), *Picea abies* (2,576, 7 datasets). Full species overlap matrix is in `dataset_overview_species_classification.md`.

### 2.2 Dataset Roles

**Core training dataset:** TreeScanPL (Poland) — largest single source, full geolocation, complete contextual feature availability, plot structure enabling inter-tree attention. 18 species across 271 circular 15 m plots.

**Multi-species external datasets** (primary cross-country generalization targets): BioDiv-3DTrees (Germany, 19 species), LAUTx (Austria, 6 species), Weiser et al. (Germany, 12 species), Frey 2022 (Germany, 6 species), Wytham Woods (UK, 6 species), NIBIO (Norway, 3 species).

**Single-species or low-diversity datasets** (CULS, Junttila/Yrttimaa, Puliti MLS, Puliti ULS 2, Saarinen 2021): limited value for classification training; useful for cross-scanner consistency checks and as additional samples for shared species.

**Scanner diversity** is a significant domain shift factor: TLS (stationary, dense stem detail), ULS (UAV, strong canopy, sparse stem), MLS (mobile, SLAM-based), and PLS (handheld) are all represented.

### 2.3 Context Feature Availability per Dataset

| Dataset | AlphaEarth | SINR / GeoPlantNet | Topo |
|---------|------------|-------------------|------|
| TreeScanPL | Yes | Yes | Yes |
| BioDiv-3DTrees | Pending BExIS | Pending BExIS | Pending BExIS |
| LAUTx | Yes | Yes | Yes |
| Weiser et al. | Yes | Yes | Yes |
| NIBIO | Yes | Yes | Yes |
| CULS | Yes | Yes | Yes |
| Frey 2022 | Yes | Yes | Yes |
| Junttila/Yrttimaa | Yes | Yes | Yes |
| Puliti MLS | Approx. only | Approx. only | Approx. only |
| Puliti ULS 2 | Yes | Yes | Yes |
| Saarinen 2021 | Yes | Yes | Yes |
| Wytham Woods | Approx. only | Approx. only | Approx. only |

Datasets with approximate-only coordinates receive context features computed from the plot centroid rather than individual tree positions. Given that SINR and topographic features are meaningful at ≥1 km resolution, centroid-based features introduce negligible error for intra-plot variation.

### 2.4 Data Split

The dataset is divided into a fixed train/val/test partition. **The test set is held out completely** — it is never used for model selection or hyperparameter tuning, only for final evaluation.

**Test set** — chosen to cover distinct geographic and ecological conditions:

| Held-out unit | Dataset | Rationale |
|--------------|---------|-----------|
| Wytham Woods (entire) | UK | Most geographically distant location; different species composition — species absent from training will be skipped in evaluation |
| 1 forest district | TreeScanPL (Poland) | Within-country held-out; tests generalization within the core training domain |
| Subset of plots from 1 area | Weiser et al. (Germany) | Partial holdout from one spatially contiguous area; remaining Weiser plots stay in training |
| Subset of plots from 1 area | NIBIO (Norway) | Partial holdout from one spatially contiguous area; remaining NIBIO plots stay in training |

**Validation set** — ~15% of remaining data, sampled at geographic unit level (whole plots or plot groups), stratified by country and scanner type. Individual trees are never split across train and val.

**Training set** — all remaining data after test and val exclusion.

The geographic unit of splitting (plots, districts, entire datasets) ensures that no spatial autocorrelation leaks between splits. Trees within the same plot are always in the same partition.

### 2.5 Standardization Principles

- The pipeline operates on **individual tree point clouds** as the primary input unit.
- Neighboring trees (for inter-tree attention in Phase 4) are retrieved by spatial query within a **15 m radius** of each target tree's stem position, regardless of original plot geometry.
- All trees across all datasets receive the same contextual feature vector (Section 3), with absent sources masked out at the source level.

---

## 3. Contextual Features

All non-point-cloud features are encoded into a **single unified 256-dim context embedding** before fusion with the point cloud encoder. The output dimensionality is invariant to which combination of sources is active.

### 3.1 Feature Inventory

| Feature | Dimensions | Type | Notes |
|---------|-----------|------|-------|
| **AlphaEarth** | 64 | Continuous | Satellite-derived per-location embedding |
| **SINR** | 256 | Continuous | Species distribution backbone features (not logits) |
| **GeoPlantNet** | N_species | Continuous | Species probability logits from coordinates |
| **Topographic variables** | 6 | Continuous | Elevation, slope, northness, eastness, TRI, TPI — from FABDEM V1-2 |

SINR and GeoPlantNet both encode species distribution priors from coordinates but in different forms (generic habitat embedding vs. species logits). Phase 1 evaluates their individual contribution; the universal model in Phase 2 includes both as separate source paths and can use either or both via masking.

### 3.2 Context Encoder Architecture

Each source has an independent encoder projecting to 256-dim with LayerNorm, producing a **source token**. A learned `[CTX]` token then aggregates information across available tokens via self-attention:

```
AlphaEarth (64)  ──► Linear(64→256) + LN  ──► token_ae   ─┐
SINR (256)       ──► Linear(256→256) + LN ──► token_sinr  ─┤  (absent sources
GeoPlantNet (N)  ──► log + MLP(N→256) + LN──► token_gpn   ─┤   excluded from
Topo (6)         ──► MLP(6→128→256) + LN  ──► token_topo  ─┘   token sequence)
                                                 │
                          prepend learned [CTX] token
                                                 │
                     1-layer self-attention (4 heads, 256-dim)
                     [CTX] attends to all source tokens
                                                 │
                          [CTX] output → 256-dim context embedding
```

Missing sources are simply absent from the token sequence — the [CTX] token attends to however many sources are available. This naturally handles any combination without zero-filling or architectural changes.

GeoPlantNet logits are log-transformed before the MLP (`log(p + ε)`) to handle near-zero values.

**Source dropout during training** (p=0.25 per source, independently) forces the model to operate well with any subset of sources and ensures each source token carries individually meaningful signal.

---

## 4. Fusion Strategies

The PTv3 backbone produces a per-tree feature vector (512-dim after global pooling); the context encoder produces a 256-dim context embedding.

### 4.1 Late Fusion

Context combined with PTv3 output after the backbone:

```
Point Cloud ──► PTv3 Backbone ──► tree_feat (512) ─┐
                                                     ├──► concat (768) ──► Linear(768→512) ──► ReLU ──► classifier
Context ──► Context Encoder ──► ctx (256) ──────────┘
```

### 4.2 Mid Fusion (FiLM)

Context injected after PTv3 stage 2, where local geometry is established but the global tree-level representation is still forming:

```
Point Cloud ──► PTv3 Stages 1–2
                      │
      Context ──► ctx (256) ──► Linear(256→1024) ──► (γ, β)
                      │
              FiLM: feat = γ * feat + β
                      │
              PTv3 Stages 3–end ──► tree_feat (512) ──► classifier
```

The FiLM injection layer is determined empirically on the validation set once and fixed for all mid-fusion experiments.

---

## 5. Inter-Tree Attention Module

Applied after per-tree feature extraction (backbone + fusion), before the classification head. All trees within a 15 m radius form a set of tokens.

```
Per-tree features
    │
    ▼
┌─────────────────────────────────────────┐
│         Inter-Tree Attention            │
│                                         │
│  Token = tree_feat + pos_encoding(XY)   │
│                                         │
│  1-layer multi-head self-attention      │
│  with learned Euclidean distance bias   │
│  (bias = -α·dist, learned α)            │
│                                         │
│  Residual: out = attn(in) + in          │
└─────────────────┬───────────────────────┘
                  │
          Refined tree features
                  │
          Classification Head
```

Single layer, 4–8 heads. Residual connection preserves well-classified trees while refining ambiguous ones.

---

## 6. Experimental Phases

### Phase 0: Context-Only Baselines

**Goal:** Quantify the location prior — how much species identity is predictable from location alone, without any point cloud geometry.

**Architecture:** Per-source encoder → 256-dim → Linear classifier. No PTv3. These are small, fast models.

**Same train/val/test split as all other phases.** Trees are still individual labeled instances, just without geometry as input.

| ID | Context sources | Purpose |
|----|----------------|---------|
| 0.0 | None (majority class) | Floor |
| 0.1 | AlphaEarth only | Satellite context alone |
| 0.2 | Topo only | Topographic context alone |
| 0.3 | SINR only | Species distribution prior, embedding form |
| 0.4 | GeoPlantNet only | Species prior, logit form |

**Key interpretive question:** What is the gap between the best context-only result and the PTv3-only baseline (1.0)? A large gap means geometry and location are complementary signals. A small gap means location dominates and geometry adds little — which would be a finding in itself.

#### Phase 0 Results (completed)

Train: 13,288 trees · Val: 1,606 trees (NIBIO, LAUTx, Weiser, Puliti ULS 2) · Test: 1,864 trees (Wytham Woods + TreeScanPL Milicz district)

| Source | Val wF1 | Test wF1 | Notes |
|--------|---------|----------|-------|
| Majority class | 0.020 | 0.267 | Test set is heavily Pinus-dominated (~44%) |
| AlphaEarth | **0.470** | **0.403** | Clear best; meaningful signal on both splits |
| GeoPlantNet | 0.347 | 0.262 | Second on val; degrades on test (weaker UK coverage) |
| Topo | 0.039 | 0.192 | Below majority on val; weak standalone |
| SINR | 0.032 | 0.002 | Near-random on both splits; fails as standalone |

**Key findings:**

- **AlphaEarth** is the strongest standalone source by a clear margin. Satellite appearance directly encodes forest composition and generalises across regions.
- **SINR fails entirely as a standalone classifier.** It is a generic habitat embedding not designed for direct classification, and it carries almost no discriminative signal alone. This is an important prior for Phase 1: SINR's marginal contribution over PTv3 may also be limited.
- **GeoPlantNet** is competitive on val but degrades on test, likely because its species distribution model has weaker coverage for Wytham Woods (UK) and the held-out Polish district. Its utility may be region-dependent.
- **Topography** alone is insufficient — terrain at this spatial scale does not resolve species identity.
- The large gap between the best context-only result (AE val wF1≈0.47) and a typical point-cloud baseline suggests geometry and location are genuinely complementary signals, motivating the fusion experiments in Phases 1–4.

---

### Phase 1: Individual Source + PTv3

**Goal:** Measure the marginal contribution of each context source when added individually to PTv3.

**Architecture:** PTv3 → single-source context encoder (simple linear projection, no CLS attention) → late concat fusion → classifier. One model per source, no combinations.

| ID | Context sources | Purpose |
|----|----------------|---------|
| 1.0 | None | PTv3 geometric baseline |
| 1.1 | AlphaEarth | Satellite context |
| 1.2 | Topo | Topographic context |
| 1.3 | SINR | Species distribution prior (SINR) |
| 1.4 | GeoPlantNet | Species distribution prior (GPN) |

**Analysis:** Per-source gain over 1.0. Compare 1.3 vs 1.4 — this tells us whether SINR or GeoPlantNet is the stronger species prior as an individual signal.

---

### Phase 2: Universal Model (Late Fusion)

**Goal:** Train a single model that handles all source combinations via masking and evaluate whether it matches or exceeds the individually-tuned Phase 1 models.

**Architecture:** PTv3 → full CLS-attention context encoder (all 4 source paths, structured source dropout) → late concat fusion → classifier. One training run.

**Evaluation:** Mask sources at inference to evaluate all combinations of interest without retraining:

| Mask configuration | Comparison |
|-------------------|------------|
| All sources active | Full context ceiling |
| AE only | vs 1.1 (individually trained) |
| Topo only | vs 1.2 |
| SINR only | vs 1.3 |
| GPN only | vs 1.4 |
| AE + Topo | Non-species context combined |
| AE + SINR | Best non-GPN combo |
| AE + GPN | Best GPN combo |
| Topo + SINR | |
| Topo + GPN | |
| AE + Topo + SINR | |
| AE + Topo + GPN | Full without SINR/GPN overlap |
| No sources (PTv3 only) | vs 1.0 — tests universal model PTv3 baseline |

The comparison between individual-source mask results and Phase 1 models tests whether multi-task training with source dropout degrades single-source performance (it typically does not, but worth verifying).

#### Source Dropout Sampling Scheme

Simple independent Bernoulli dropout underrepresents sparse combinations (1–2 sources get only ~5% of batches at p_drop=0.25). Instead, use **two-stage cardinality sampling**:

1. With probability 0.35: use all 4 sources (protects the primary deployment scenario).
2. Otherwise: sample cardinality k uniformly from {1, 2, 3}, then sample k sources uniformly from the 4 available.

This gives roughly: full set 35% · 3-source 22% · 2-source 22% · 1-source 22%. At least one source is always active (all-masked is never sampled — it is a degenerate case not seen at inference).

**Training curriculum for source dropout:**

| Epochs | Schedule |
|--------|----------|
| 1–5 | All sources always present (no dropout) — establish stable representations |
| 6–10 | Ramp up: p_full=0.8, sparse combos rare |
| 11+ | Full sampling scheme (p_full=0.35) |

Starting with no dropout prevents early gradient instability while the CLS attention and source encoders are randomly initialised.

**Key decision:** Identify the best-performing source combination(s) to carry into Phase 3.

---

### Phase 3: Mid Fusion (FiLM)

**Goal:** Test whether injecting context earlier in the PTv3 computation (FiLM conditioning) outperforms late fusion for the best context configuration identified in Phase 2.

**Architecture:** Same universal model setup as Phase 2, but context injected via FiLM after PTv3 stage 2 instead of concatenation after pooling. One training run.

| ID | Fusion | Context |
|----|--------|---------|
| 3.0 | Late concat | Best Phase 2 config (reuse Phase 2 result) |
| 3.1 | Mid FiLM | Best Phase 2 config |

**Analysis:** Does early injection — allowing context to modulate how geometry is processed rather than appending after the fact — improve performance? Especially examine per-species effects: do ambiguous species (confusable by geometry alone) benefit more from mid-fusion?

---

### Phase 4: Inter-Tree Attention

**Goal:** Test whether neighborhood-level reasoning improves classification.

**Fixed:** Best configuration from Phase 3 (best fusion strategy + best source combination).

| ID | Configuration | Inter-Tree Attention |
|----|--------------|---------------------|
| 4.0 | Phase 3 winner | None (reuse Phase 3 result) |
| 4.1 | Phase 3 winner | 1-layer self-attention with distance bias |

**Analysis:**
- Overall metrics vs 4.0.
- Stratified by prediction confidence: does attention help most for low-confidence trees?
- Stratified by stand structure: dense mixed stands vs. monocultures.
- Attention weight visualization: which neighbors does a tree attend to?

---

### Phase 5: Final Evaluation on Test Set

**Goal:** Unbiased evaluation of the final system on the held-out test set.

All preceding phases use only train and val splits. The test set (Wytham Woods, 1 TreeScanPL district, ~4 Weiser plots, ~5 NIBIO plots) is evaluated once, at the end, for the following configurations:

1. **PTv3 only** (Exp 1.0) — geometric reference
2. **Phase 3 winner** — best fusion without inter-tree attention
3. **Phase 4 winner** — full system with inter-tree attention

**Analysis:**
- Per held-out unit: how does performance vary across Wytham (UK, very different ecosystem), Polish district (within-domain), German plots, Norwegian plots?
- Per-species confusion matrices on each held-out unit.
- For Wytham specifically: does context help more than geometry, given the species there are either absent or rare in training data?

---

## 7. Evaluation Protocol

### 7.1 Metrics

| Metric | Level | Purpose |
|--------|-------|---------|
| Overall Accuracy (OA) | Global | Basic performance |
| Weighted F1 | Global | Accounts for class imbalance |
| Per-species F1 | Per-class | Which species benefit from context |
| Confusion matrix | Per-class | Full error structure |

### 7.2 Reporting

- All experiments logged to W&B (or MLflow), tagged with: phase, experiment ID, context sources (as bitmask), fusion strategy, dataset version.
- Per-species metrics logged, not just aggregates.
- Training curves (loss, val F1) to check for overfitting.
- For Phase 2 universal model: log per-mask-configuration val metrics to track which combinations are strong vs. weak.

---

## 8. Possible Future Ablation: BDL Ecological Classes

The Polish BDL forest database provides categorical site-quality variables (fertility classes, moisture classes) for Polish forest districts. Excluded from the main plan because they are Poland-only (incompatible with the cross-country generalization focus) and categorical (requiring a different sub-encoder than the continuous source paths used here).

If Phase 2–3 results show strong performance on Polish data but poor cross-country transfer, BDL could be revisited as a Poland-only ablation to test whether ecological site classifications add signal beyond what AlphaEarth and topography already capture for Polish forests.
