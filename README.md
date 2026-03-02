# PTv3 Tree Species Classification

Point Transformer V3 (PTv3) for individual tree species classification from airborne LiDAR point clouds, with optional AlphaEarth satellite context fusion.

## Overview

This project classifies tree species from individual tree point clouds extracted from Polish forest plots (TreeScanPL dataset). It uses a PTv3 backbone pretrained on FOR-species20K, finetuned on TreeScanPL with district-level 6-fold cross-validation.

**10 target genera:** Acer, Alnus, Betula, Carpinus, Fagus, Larix, Picea, Pinus, Quercus, Tilia

**6 forest districts (folds):** Gorlice, Herby, Katrynka, Milicz, Piensk, Suprasl

### Models

| Model | Description |
|-------|-------------|
| **PTv3 baseline** | Geometry-only (XYZ + normals), pretrained backbone finetuned on TreeScanPL |
| **PTv3 + AlphaEarth (projected)** | Late fusion — backbone features (512d) and satellite embeddings (64d) projected to 128d each before merging |
| **PTv3 + AlphaEarth (direct)** | Late fusion — direct concatenation (512 + 64 = 576d) |

## Setup

### Environment

```bash
conda env create -f Pointcept/environment.yml
conda activate pointcept-torch2.5.0-cu12.4

# Build custom ops
cd Pointcept/libs/pointops && python setup.py install && cd ../../..
cd Pointcept/libs/pointgroup_ops && python setup.py install && cd ../../..
```

A separate `lidar` conda environment with `laspy` is needed only for data preparation scripts.

### Data Preparation

The pipeline converts raw LAZ plot files into per-tree `.npy` files (N×6: XYZ + normals):

```bash
# 1. Extract individual trees from plot-level LAZ files
conda run -n lidar python prepare_treescanpl.py \
    --input_dir /path/to/TreeScanPL_2cm \
    --output_dir Pointcept/data/treescanpl \
    --species_csv /path/to/species_id_names.csv

# 2. Generate district-level 6-fold splits
conda run -n lidar python generate_district_folds.py \
    --input_dir /path/to/TreeScanPL_2cm \
    --species_csv /path/to/species_id_names.csv \
    --output_dir Pointcept/data/treescanpl

# 3. (For context models) Generate sample→plot_id mapping
conda run -n lidar python generate_sample_plotid_mapping.py \
    --input_dir /path/to/TreeScanPL_2cm \
    --species_csv /path/to/species_id_names.csv \
    --output_dir Pointcept/data/treescanpl

# 4. (For context models) Copy AlphaEarth embeddings
cp /path/to/plots_alphaearth_2018.csv Pointcept/data/treescanpl/
```

Expected data directory structure:
```
Pointcept/data/treescanpl/
├── Acer/Acer_0001.npy ... Acer_0060.npy
├── Alnus/ ...
├── Betula/ ...
├── ...
├── Tilia/
├── treescanpl_fold{0-5}_{train,test}.txt
├── sample_plotid_mapping.csv          # for context models
└── plots_alphaearth_2018.csv          # for context models
```

### FOR-species20K Pretraining

The backbone is pretrained on FOR-species20K (20K trees, 33 species). See `prepare_forspecies20k.py` for dataset preparation. The pretrained checkpoint is expected at:

```
Pointcept/exp/forspecies20k/cls-ptv3-v1m1-0-base/model/model_best.pth
```

## Training

### Baseline PTv3 (6-fold CV)

```bash
bash run_district_kfold.sh
```

### PTv3 + AlphaEarth Context (6-fold CV)

```bash
# Variant A: Projected fusion (recommended)
bash run_district_kfold_context.sh --variant projected

# Variant B: Direct fusion
bash run_district_kfold_context.sh --variant direct
```

Each fold trains for 60 epochs (~10k steps) with AdamW + OneCycleLR. Results are saved to `Pointcept/exp/treescanpl/`.

### Single Fold

```bash
cd Pointcept
python tools/train.py \
    --config-file configs/treescanpl/cls-ptv3-v1m1-0-base-finetune-kfold.py \
    --num-gpus 1 \
    --options save_path=exp/treescanpl/fold0 \
              weight=exp/forspecies20k/cls-ptv3-v1m1-0-base/model/model_best.pth \
              data.train.fold=0 data.val.fold=0 data.test.fold=0
```

## Evaluation

```bash
# Aggregate confusion matrices across all 6 folds
python aggregate_kfold_results.py \
    --exp_base Pointcept/exp/treescanpl \
    --output_dir results/kfold
```

Reports overall accuracy, mean per-class accuracy, macro/weighted F1, and per-class precision/recall.

## Project Structure

```
ptv3_cls/
├── Pointcept/                        # Point cloud framework (based on Pointcept)
│   ├── configs/
│   │   ├── treescanpl/               # TreeScanPL classification configs
│   │   └── forspecies20k/            # FOR-species20K pretraining configs
│   ├── pointcept/
│   │   ├── datasets/treescanpl.py    # TreeScanPL + Context dataset classes
│   │   └── models/default.py         # DefaultClassifier + ContextClassifier
│   └── tools/                        # train.py, test.py
├── prepare_treescanpl.py             # LAZ → .npy extraction
├── prepare_forspecies20k.py          # FOR-species20K preparation
├── generate_district_folds.py        # 6-fold split generation
├── generate_sample_plotid_mapping.py # Sample → plot_id mapping
├── aggregate_kfold_results.py        # Cross-fold metric aggregation
├── run_district_kfold.sh             # Baseline 6-fold runner
└── run_district_kfold_context.sh     # Context fusion 6-fold runner
```

## Acknowledgements

Built on the [Pointcept](https://github.com/Pointcept/Pointcept) framework (MIT License).
