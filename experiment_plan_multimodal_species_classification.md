# Experiment Plan: Multimodal Tree Species Classification from TLS Point Clouds

**Author:** Maks  
**Date:** March 2026  
**Context:** Multi-dataset, multi-country tree species classification using PTv3 backbone with contextual features and inter-tree attention

---

## 1. Overview

This plan addresses the systematic evaluation of multimodal fusion strategies for tree species classification from terrestrial laser scanning (TLS) data. The core architecture is a Point Transformer v3 (PTv3) backbone operating on individual tree point clouds, augmented with contextual features derived from remote sensing, ecological databases, and species distribution models.

The key research questions, in order:

1. **Feature selection:** Which combination of contextual features best improves species classification when fused with point cloud features via a simple baseline architecture?
2. **Fusion strategy:** Given the best feature set, does mid-fusion or late fusion perform better?
3. **Inter-tree attention:** Does a neighborhood attention layer that allows trees to exchange information before final prediction improve accuracy, especially for ambiguous individuals?

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

All non-point-cloud features are encoded into a **single unified context embedding** before fusion with the point cloud encoder. The features are organized by availability:

### 3.1 Feature Inventory

| Feature | Dimensions | Type | Availability | Notes |
|---------|-----------|------|-------------|-------|
| **AlphaEarth embeddings** | 64 | Continuous | Global | Satellite-derived, per-location |
| **SNIR embeddings** | TBD (backbone features) | Continuous | Global | Species distribution model; backbone features used as embedding, not final probabilities |
| **GeoPlantNet predictions** | N_species probabilities | Continuous | Study areas (precomputed) | Species probability distributions from coordinates |
| **BDL ecological classes** | 4 fertility + 2 moisture | Categorical | Poland only | Learned embedding tables; masked for non-Polish datasets |
| **XY position** | 2 | Continuous | All datasets | Relative to plot/neighborhood center |
| **Elevation** | 1 | Continuous | All datasets | From DEM |
| **Slope** | 1 | Continuous | Derivable | From DEM |
| **Aspect** | 2 (sin, cos encoded) | Continuous | Derivable | Circular encoding to avoid discontinuity |

### 3.2 Context Encoder Architecture

Features are grouped by data type, each with a tailored sub-encoder. Group outputs are concatenated and fused into a single context embedding.

**Group A — Learned embeddings** (AlphaEarth, SNIR backbone features): Already latent representations. Each gets a linear projection + LayerNorm → 64-dim.

**Group B — Categoricals** (BDL fertility, moisture): Learned embedding tables (fertility: 4 → 16-dim, moisture: 2 → 8-dim), concatenated → 24-dim.

**Group C — Probability distributions** (GeoPlantNet, optionally SNIR output probs): Log-transformed first (log(p + ε) to handle near-zero values), then passed through a small MLP → 32-dim per source.

**Group D — Continuous scalars** (XY, elevation, slope, aspect as sin/cos): Concatenated (6-dim) → small MLP → 32-dim.

```
Group A: AE ──► Linear+LN ──► 64 ─┐
         SNIR ► Linear+LN ──► 64 ─┤
Group B: BDL ──► EmbedTables ► 24 ─┤
Group C: GeoPLN ► log ► MLP ► 32 ──┼──► Concat ──► Fusion MLP (D_cat → 256 → D_ctx) ──► Context embedding
Group D: Scalars ► MLP ────► 32 ───┘
```

**Modularity:** Each group encoder is independent. Absent feature groups are zero-filled at the group output level with a binary availability mask bit per group, so any combination of features works without architectural changes. The same context embedding feeds into both mid and late fusion variants.

---

## 4. Fusion Strategies

Two fusion strategies are compared. In both cases, the PTv3 backbone produces a per-tree feature vector (512-dim after global pooling), and the context encoder produces a context embedding (D_ctx dims).

### 4.1 Late Fusion

Context embedding is combined with the PTv3 output **after** the backbone, before the classification head.

```
Point Cloud ──► PTv3 Backbone ──► tree_feat (512)
                                        │
Context ──► Context Encoder ──► ctx (D_ctx)  │
                                        │
                              ┌─────────▼──────────┐
                              │  Projection + Concat │
                              │  or Gated Addition   │
                              └─────────┬──────────┘
                                        │
                                  Classification Head
```

**Combination sub-variants** (tested within late fusion):

- **Concat + project:** [tree_feat; ctx] → Linear(512 + D_ctx, 512) → ReLU → classifier
- **Project-then-concat:** Both projected to 128-dim first → [128; 128] → classifier (current best baseline)
- **Gated addition:** tree_feat + σ(W·ctx) ⊙ ctx_projected — lets the model learn how much context to trust

### 4.2 Mid Fusion

Context embedding is injected at an intermediate PTv3 layer where points are being aggregated into superpoints. The context vector modulates the intermediate representation via:

- **FiLM conditioning:** context → (γ, β) applied as affine transform on intermediate features
- **Cross-attention:** intermediate point features attend to the context embedding

The injection point should be after the first major downsampling stage, where local geometric features have been extracted but global tree-level representation is still forming. The exact layer is determined empirically but fixed across experiments once chosen.

```
Point Cloud ──► PTv3 Layers 1..L_mid
                        │
        Context ──► Context Encoder ──► ctx
                        │                │
                        ▼                │
                ┌───────────────┐        │
                │  Mid-Fusion   │◄───────┘
                │ (FiLM / XAttn)│
                └───────┬───────┘
                        │
                PTv3 Layers L_mid+1..L
                        │
                   tree_feat (512)
                        │
                  Classification Head
```

---

## 5. Inter-Tree Attention Module

Applied **after** the per-tree feature extraction (backbone + fusion) and **before** the classification head. All trees within a 15 m radius neighborhood form a set of tokens.

### 5.1 Architecture

```
Per-tree features (from backbone + fusion)
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
│  Residual connection: out = attn + in   │
└─────────────────┬───────────────────────┘
                  │
          Refined tree features
                  │
          Classification Head
```

### 5.2 Design Decisions

- **Positional encoding:** Euclidean XY offset from plot center, encoded via small MLP or sinusoidal encoding, added to tree feature tokens.
- **Distance bias:** Pairwise Euclidean distances converted to attention bias (e.g., `bias = -α · dist`, learned α), added to attention logits before softmax.
- **Single layer, 4–8 heads.** Keep it lightweight — the goal is contextual refinement, not deep processing.
- **Residual connection** so that well-classified trees pass through unharmed while ambiguous ones get refined.
- **Batch unit:** All trees in a neighborhood (typically 20–60 trees). Manageable for standard attention.

---

## 6. Experimental Phases

### Phase 1: Feature Selection (Simple Late Fusion)

**Goal:** Determine which contextual features contribute to classification accuracy.

**Fixed architecture:** PTv3 backbone → project-then-concat late fusion (both streams to 128-dim) → classifier. This is the simplest viable architecture and serves as a testbed for feature utility.

**Fixed evaluation:** Single geographic split — hold out 2–3 districts as test set, remaining for train/val. Fast iteration, still tests geographic generalization.

**Experiments:**

| ID | Point Cloud | Context Features | Purpose |
|----|------------|-----------------|---------|
| 1.0 | PTv3 | None (baseline) | Pure geometric baseline |
| 1.1 | PTv3 | AlphaEarth | Satellite context |
| 1.2 | PTv3 | SNIR | Species distribution prior |
| 1.3 | PTv3 | GeoPlantNet | Alternative species prior |
| 1.4 | PTv3 | BDL | Ecological site classes (Poland only) |
| 1.5 | PTv3 | XY + Elevation + Slope + Aspect | Topographic context |
| 1.6 | PTv3 | AE + SNIR | Best remote sensing combo? |
| 1.7 | PTv3 | AE + GeoPlantNet | AE + species prior |
| 1.8 | PTv3 | AE + SNIR + GeoPlantNet | All distribution info |
| 1.9 | PTv3 | AE + BDL + Topo | AE + ecological + topographic |
| 1.10 | PTv3 | All features | Kitchen sink |

**Analysis:**

- Per-species F1 and overall weighted F1 / OA.
- Marginal contribution of each feature group (compare 1.1–1.5 against 1.0).
- Redundancy: does adding SNIR on top of GeoPlantNet help? Does topo help on top of AE?
- Select **top 2–3 configurations** (best single, best pair, best full set) to carry forward.

**Decision rule:** A feature set is carried forward if it improves weighted F1 by ≥ 1 percentage point over the next-best subset, or if it improves accuracy on hard/ambiguous species even at similar overall F1.

### Phase 2: Fusion Strategy Comparison

**Goal:** Determine whether mid or late fusion is superior for the selected feature sets.

**Feature sets:** Top 2–3 from Phase 1.

**Experiments:**

| ID | Feature Set | Fusion | Combination Method |
|----|------------|--------|--------------------|
| 2.1a | Best_1 | Late | Project-then-concat (128+128) |
| 2.1b | Best_1 | Late | Gated addition |
| 2.1c | Best_1 | Mid | FiLM conditioning |
| 2.1d | Best_1 | Mid | Cross-attention |
| 2.2a | Best_2 | Late | Project-then-concat |
| 2.2b | Best_2 | Late | Gated addition |
| 2.2c | Best_2 | Mid | FiLM conditioning |
| 2.2d | Best_2 | Mid | Cross-attention |
| ... | Best_3 | ... | ... |

**Analysis:**

- Same metrics as Phase 1.
- Attention/gate visualizations: what does the model attend to? Which trees benefit most from context?
- Select **1 best configuration** (feature set + fusion strategy + combination method).

### Phase 3: Inter-Tree Attention

**Goal:** Test whether neighborhood-level reasoning improves classification, especially for ambiguous trees.

**Fixed:** Best configuration from Phase 2.

**Experiments:**

| ID | Configuration | Inter-Tree Attention |
|----|--------------|---------------------|
| 3.0 | Phase 2 winner | None (baseline from Phase 2) |
| 3.1 | Phase 2 winner | 1-layer self-attention with distance bias |

**Analysis:**

- Overall metrics (F1, OA).
- **Stratified analysis by tree ambiguity:** Split test trees by confidence of the Phase 2 baseline prediction (e.g., max softmax probability). Does attention help most for low-confidence trees?
- **Stratified by structural context:** Does attention help more in dense mixed stands vs. monocultures?
- Attention weight visualization: which neighbors does a tree attend to? Are they same-species or informative heterospecific neighbors?

### Phase 4: Final Validation

**Goal:** Robust evaluation of the final 2–3 best configurations.

**Evaluation protocol:** Leave-one-district-out (or leave-one-region-out for multi-country data). This is expensive — only the final candidates go through this.

**Configurations to validate:**

1. Phase 2 winner (best fusion, no inter-tree attention) — the clean multimodal baseline.
2. Phase 3 winner (best fusion + inter-tree attention) — the full system.
3. Optionally: PTv3 alone (Exp 1.0) as a reference baseline.

**Analysis:**

- Per-district / per-region performance.
- Cross-country generalization: how does the model trained on Polish data perform on external datasets?
- Per-species confusion matrices.
- Statistical significance testing (paired t-test or Wilcoxon across folds).

---

## 7. Evaluation Protocol Details

### 7.1 Metrics

| Metric | Level | Purpose |
|--------|-------|---------|
| Overall Accuracy (OA) | Global | Basic performance |
| Weighted F1 | Global | Accounts for class imbalance |
| Per-species F1 | Per-class | Identifies which species benefit from context |
| Per-species Precision/Recall | Per-class | Disentangles confusion patterns |
| Confusion matrix | Per-class | Full error structure |

### 7.2 Splits

- **Phases 1–3:** Fixed geographic split. Hold out 2–3 districts (diverse in species composition and site conditions). Use remaining districts for train/val (random 80/20 within training districts).
- **Phase 4:** Leave-one-district-out cross-validation across all available districts/regions.

### 7.3 Reporting

- All experiments logged to W&B (or MLflow).
- Each run tagged with: phase, experiment ID, feature set, fusion strategy, dataset version.
- Per-species metrics logged, not just aggregates.
- Training curves (loss, val F1) to check for overfitting, especially with richer feature sets.


