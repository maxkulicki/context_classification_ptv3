# Experiment Plan: Multimodal Tree Species Classification from TLS Point Clouds

**Author:** Maks
**Date:** March 2026
**Context:** Multi-dataset, multi-country tree species classification using PTv3 backbone with contextual features and inter-tree attention

---

## 1. Overview

This plan addresses the systematic evaluation of multimodal fusion strategies for tree species classification from terrestrial laser scanning (TLS) data. The core architecture is a Point Transformer v3 (PTv3) backbone operating on individual tree point clouds, augmented with contextual features derived from remote sensing, species distribution models, and topography.

The key research questions, in order:

1. **Location prior:** How well do context features alone (no point cloud) predict species identity? This quantifies the location prior and contextualizes all subsequent gains.
2. **Feature selection:** Which combination of contextual features best improves species classification when fused with point cloud features?
3. **Fusion strategy:** Given the best feature set, does mid-fusion or late fusion perform better?
4. **Inter-tree attention:** Does a neighborhood attention layer that allows trees to exchange information before final prediction improve accuracy?

The plan is structured in sequential phases. Each phase resolves one question before the next begins, avoiding combinatorial explosion.

---

## 2. Datasets

### 2.1 Core Dataset

**TreeScanPL-10K** — 272 circular plots (15 m radius), 10,000+ manually segmented trees across Central European forests (Poland). Full TLS point clouds with species labels, stem XY positions, DBH, and height.

### 2.2 External Datasets

| Dataset | Structure | Coverage | Notes |
|---------|-----------|----------|-------|
| Dataset B | Individually segmented trees with geolocation from one contiguous area | TBD country | No plot structure; neighbors retrieved by spatial query within radius |
| Dataset C | Mix of circular and rectangular plots | TBD country | Clip/query neighbors within 15 m of each target tree |

### 2.3 Standardization Principles

- The pipeline operates on **individual tree point clouds** as the primary input unit.
- Neighboring trees (for inter-tree attention in Phase 3) are retrieved by spatial query within a **15 m radius** of each target tree's stem position, regardless of original plot geometry.
- All trees across all datasets receive the same contextual feature vector (see Section 3), with missing features masked or zero-filled.

---

## 3. Contextual Features

All non-point-cloud features are encoded into a **single unified 256-dim context embedding** before fusion with the point cloud encoder. The 256-dim output is invariant to which combination of sources is active.

### 3.1 Feature Inventory

| Feature | Dimensions | Type | Availability | Notes |
|---------|-----------|------|-------------|-------|
| **AlphaEarth embeddings** | 64 | Continuous | Global | Satellite-derived, per-location |
| **SINR embeddings** | 256 | Continuous | Global | Species distribution backbone features; not logits |
| **GeoPlantNet predictions** | N_species logits | Continuous | Global (precomputed) | Species probability distribution from coordinates |
| **Topographic variables** | 6 | Continuous | Global | Elevation, slope, northness, eastness, TRI, TPI — derived from FABDEM V1-2 |

SINR and GeoPlantNet are **substitutes** — both encode a species distribution prior from coordinates, differing in form (generic habitat embedding vs. species logits). They are never combined in the same experiment; whichever performs better in Phase 1 is carried forward.

### 3.2 Context Encoder Architecture

Each source has an independent encoder that projects to 256-dim with LayerNorm. The context embedding is the **masked mean** of available source embeddings:

```
AlphaEarth (64)   ──► Linear(64→256) + LN  ──► 256 ─┐
SINR (256)        ──► Linear(256→256) + LN ──► 256 ──┤
  OR                                                   ├──► masked mean ──► 256-dim context
GeoPLN (N_sp)     ──► log + MLP(N→256) + LN──► 256 ──┤
Topo (6)          ──► MLP(6→128→256) + LN  ──► 256 ─┘
```

**Why masked mean:** The output is always 256-dim regardless of which sources are present. Missing sources are excluded from the mean — they contribute nothing rather than being zero-filled. This makes any combination of sources produce a representation in the same embedding space, and ablations reduce to simply toggling source masks.

**Source dropout during training** (p=0.25 per source, independently): prevents the model from over-relying on any single source and ensures each individual projection carries meaningful signal on its own.

GeoPlantNet logits are log-transformed before the MLP (`log(p + ε)`) to handle near-zero values.

---

## 4. Fusion Strategies

Two fusion strategies are compared. The PTv3 backbone produces a per-tree feature vector (512-dim after global pooling); the context encoder produces a 256-dim context embedding.

### 4.1 Late Fusion

Context embedding combined with PTv3 output after the backbone:

```
Point Cloud ──► PTv3 Backbone ──► tree_feat (512) ─┐
                                                     ├──► concat (768) ──► Linear(768→512) ──► ReLU ──► classifier
Context ──► Context Encoder ──► ctx (256) ──────────┘
```

Variant: **gated addition** — both streams projected to 256 first, then `out = tree_feat_proj + σ(W·ctx) ⊙ ctx_proj`. Lets the model learn how much context to trust per-instance.

### 4.2 Mid Fusion (FiLM)

Context injected after PTv3 stage 2 (after first major downsampling, where local geometry is established but global representation is still forming):

```
Point Cloud ──► PTv3 Stages 1–2
                      │
      Context ──► ctx (256) ──► Linear(256→1024) ──► (γ, β)
                      │
              FiLM: feat = γ * feat + β
                      │
              PTv3 Stages 3–end ──► tree_feat (512) ──► classifier
```

The injection layer is fixed empirically once and held constant across all mid-fusion experiments.

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
│  with Euclidean distance bias           │
│  (closer neighbors → higher attention)  │
│                                         │
│  Residual: out = attn(in) + in          │
└─────────────────┬───────────────────────┘
                  │
          Refined tree features
                  │
          Classification Head
```

Single layer, 4–8 heads. Distance bias: `bias = -α · dist` (learned α) added to attention logits before softmax. Residual connection preserves well-classified trees while refining ambiguous ones.

---

## 6. Experimental Phases

### Phase 0: Context-Only Baselines

**Goal:** Quantify the location prior. How well does each context source predict species identity without any point cloud geometry? Establishes a ceiling for "what location alone tells you" and contextualizes all Phase 1 gains.

**Architecture:** Context encoder (same as all other phases) → 256-dim → Linear classifier. No PTv3, no point cloud.

**Same splits and metrics as all other phases** — trees are still individual labeled instances, just without geometry.

| ID | Context sources | Purpose |
|----|----------------|---------|
| 0.0 | None (majority class) | Floor |
| 0.1 | SINR only | Species distribution prior, embedding form |
| 0.2 | GeoPlantNet only | Species prior, logit form — expected strongest here |
| 0.3 | AlphaEarth + Topo | Non-species context: satellite + terrain alone |
| 0.4 | AE + Topo + SINR | Best non-GPN combination |
| 0.5 | AE + Topo + GeoPlantNet | Full context-only ceiling |

**Key question:** What is the gap between the best context-only result (0.5) and the PTv3-only baseline (1.0)? This gap measures how complementary geometry and location are as signal sources.

### Phase 1: Feature Selection

**Goal:** Determine which contextual features best improve classification when fused with point cloud features.

**Fixed architecture:** PTv3 → late concat fusion → classifier. Simplest viable fusion, used as a stable testbed.

**Fixed evaluation:** Single geographic split — hold out 2–3 districts as test set, remaining for train/val.

| ID | Context sources | Purpose |
|----|----------------|---------|
| 1.0 | None | PTv3 geometric baseline |
| 1.1 | AlphaEarth | Satellite context alone |
| 1.2 | Topo | Topographic context alone |
| 1.3 | SINR | Species distribution prior (SINR) |
| 1.4 | GeoPlantNet | Species distribution prior (GPN) |
| 1.5 | AE + Topo | Non-species context combined |
| 1.6 | AE + winner(1.3 vs 1.4) | Best species prior + remote sensing |
| 1.7 | Topo + winner | Best species prior + topography |
| 1.8 | AE + Topo + winner | Full context |

Experiments 1.6–1.8 use whichever of SINR or GeoPlantNet wins in 1.3 vs 1.4. The loser is dropped and not carried further.

**Decision rule:** Carry forward the top 2–3 feature sets that improve weighted F1 by ≥ 1pp over the next-best subset, or that improve per-species F1 on hard/ambiguous species even at similar overall F1.

### Phase 2: Fusion Strategy

**Goal:** Determine whether mid or late fusion is superior for the selected feature sets.

**Feature sets:** Top 2 from Phase 1.

| ID | Feature set | Fusion |
|----|------------|--------|
| 2.1 | Best_1 | Late concat (reuse Phase 1 result) |
| 2.2 | Best_1 | Late gated addition |
| 2.3 | Best_1 | Mid FiLM |
| 2.4 | Best_2 | Late gated addition |
| 2.5 | Best_2 | Mid FiLM |

**Analysis:** Per-species F1 delta relative to Phase 1 winner. Gate/attention visualizations: which trees benefit most from context?

### Phase 3: Inter-Tree Attention

**Goal:** Test whether neighborhood-level reasoning improves classification.

**Fixed:** Best configuration from Phase 2.

| ID | Configuration | Inter-Tree Attention |
|----|--------------|---------------------|
| 3.0 | Phase 2 winner | None (already run) |
| 3.1 | Phase 2 winner | 1-layer self-attention with distance bias |

**Analysis:** Stratified by prediction confidence (does attention help most for low-confidence trees?) and stand structure (dense mixed vs. monoculture).

### Phase 4: Final Validation

**Goal:** Robust evaluation of the final 2–3 best configurations.

**Evaluation protocol:** Leave-one-district-out cross-validation. Only the final candidates go through this — it is expensive.

**Configurations:**
1. PTv3 alone (Exp 1.0) — geometric reference
2. Phase 2 winner — best fusion, no inter-tree attention
3. Phase 3 winner — best fusion + inter-tree attention

**Analysis:** Per-district performance, cross-country generalization, per-species confusion matrices, statistical significance (Wilcoxon across folds).

---

## 7. Evaluation Protocol

### 7.1 Metrics

| Metric | Level | Purpose |
|--------|-------|---------|
| Overall Accuracy (OA) | Global | Basic performance |
| Weighted F1 | Global | Accounts for class imbalance |
| Per-species F1 | Per-class | Which species benefit from context |
| Confusion matrix | Per-class | Full error structure |

### 7.2 Splits

- **Phases 0–3:** Fixed geographic split. Hold out 2–3 districts diverse in species composition and site conditions. 80/20 train/val within training districts.
- **Phase 4:** Leave-one-district-out across all available districts/regions.

### 7.3 Reporting

- All experiments logged to W&B (or MLflow), tagged with: phase, experiment ID, feature set, fusion strategy, dataset version.
- Per-species metrics logged, not just aggregates.
- Training curves (loss, val F1) to check for overfitting.

---

## 8. Possible Future Ablation: BDL Ecological Classes

The Polish BDL forest database provides categorical site-quality variables (fertility classes, moisture classes) for Polish forest districts. These were excluded from the main experiment plan because they are Poland-only (limiting generalization experiments) and categorical (requiring embedding tables rather than the continuous encoder used for all other sources).

If Phase 1–2 results show strong performance on Polish data but poor cross-country transfer, BDL could be revisited as a Poland-only ablation to test whether ecological site classifications add signal beyond what AlphaEarth and topography already capture. This would require only adding a categorical sub-encoder to the context module and running the ablation on the Polish subset of the data.
