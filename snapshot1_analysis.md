# Snapshot 1 — Training Run Analysis
Generated: 2026-03-23 10:55

Metrics extracted from `train.log` (train_loss, val_mAcc, val_allAcc) and
TensorBoard tfevents (val_loss). Sparklines show 25-point subsampled trajectory.
Peak epoch is 1-indexed.

## 1. Configuration Overview

| Run | Context | Ctx dim | Ep | LR | Pretrained | Loss |
|-----|---------|---------|----|-----|------------|------|
| ptv3_baseline_8gpu_100ep | None | — | 100 | 0.008 | No | CE+Lovasz |
| ptv3_baseline_8gpu_200ep | None | — | 200 | 0.008 | No | CE+Lovasz |
| ptv3_normals_8gpu_200ep | None (normals) | — | 200 | 0.008 | No | CE+Lovasz |
| ptv3_ctx_ae_8gpu_100ep | AE | 64 | 100 | 0.004 | No | CE+Lovasz |
| ptv3_ctx_gpn_8gpu_100ep | GPN | 18 | 100 | 0.004 | No | CE+Lovasz |
| ptv3_ctx_sinr_8gpu_100ep | SINR | 256 | 100 | 0.004 | No | CE+Lovasz |
| ptv3_ctx_topo_8gpu_100ep | Topo | 6 | 100 | 0.004 | No | CE+Lovasz |
| ptv3_ctx_ae_pretrained_8gpu_200ep | AE | 64 | 200 | 0.004 | Yes | CE+Lovasz |
| ptv3_ctx_sinr_pretrained_8gpu_200ep | SINR | 256 | 200 | 0.004 | Yes | CE+Lovasz |
| ptv3_ctx_ae_pretrained_wce_8gpu_100ep | AE | 64 | 100 | 0.004 | Yes | WCE |
| ptv3_ctx_ae_pretrained_wce_8gpu_300ep | AE | 64 | 300 | 0.004 | Yes | WCE |
| ptv3_ctx_ae_sinr_cat_8gpu_100ep | AE+SINR (cat) | 64+256 | 200 | 0.001 | Yes | CE |
| ptv3_ctx_ae_sinr_cat_4gpu_200ep | AE+SINR (cat) | 64+256 | 200 | 0.001 | Yes | CE |
| ptv3_ctx_ae_sinr_cat_scratch_4gpu_200ep | AE+SINR (cat) | 64+256 | 200 | 0.001 | No | CE |
| ptv3_ctx_universal_8gpu_100ep | Universal (all 4) | all | 50 | 0.001 | Yes | CE+Lovasz |

## 2. Final & Best Metrics

Best = highest value seen across all epochs. Final = value at last epoch.
mAcc and allAcc from train.log; val_loss final from wandb-summary.

| Run | Best mAcc (ep) | Best allAcc (ep) | Final mAcc | Final allAcc | Final val_loss |
|-----|---------------|-----------------|------------|--------------|----------------|
| ptv3_baseline_8gpu_100ep | 0.356 @ep85 | 0.614 @ep36 | 0.087 | 0.236 | N/A |
| ptv3_baseline_8gpu_200ep | 0.353 @ep181 | 0.621 @ep54 | 0.344 | 0.440 | 3.970 |
| ptv3_normals_8gpu_200ep | 0.344 @ep132 | 0.619 @ep96 | 0.320 | 0.497 | 3.084 |
| ptv3_ctx_ae_8gpu_100ep | 0.418 @ep55 | 0.575 @ep18 | 0.396 | 0.443 | N/A |
| ptv3_ctx_gpn_8gpu_100ep | 0.346 @ep56 | 0.475 @ep56 | 0.337 | 0.442 | 3.480 |
| ptv3_ctx_sinr_8gpu_100ep | 0.434 @ep65 | 0.558 @ep71 | 0.394 | 0.484 | 2.662 |
| ptv3_ctx_topo_8gpu_100ep | 0.346 @ep58 | 0.540 @ep29 | 0.311 | 0.368 | 6.099 |
| ptv3_ctx_ae_pretrained_8gpu_200ep ⚠ | 0.429 @ep70 | 0.626 @ep70 | 0.077 | 0.048 | N/A |
| ptv3_ctx_sinr_pretrained_8gpu_200ep ⚠ | 0.383 @ep51 | 0.625 @ep36 | 0.077 | 0.048 | N/A |
| ptv3_ctx_ae_pretrained_wce_8gpu_100ep | 0.444 @ep14 | 0.673 @ep14 | 0.391 | 0.538 | 3.203 |
| ptv3_ctx_ae_pretrained_wce_8gpu_300ep ⚠ | 0.418 @ep31 | 0.666 @ep24 | 0.077 | 0.048 | N/A |
| ptv3_ctx_ae_sinr_cat_8gpu_100ep | N/A | N/A | N/A | N/A | N/A |
| ptv3_ctx_ae_sinr_cat_4gpu_200ep | 0.415 @ep112 | 0.653 @ep34 | 0.387 | 0.523 | 2.696 |
| ptv3_ctx_ae_sinr_cat_scratch_4gpu_200ep ⚠ | 0.390 @ep23 | 0.599 @ep9 | 0.077 | 0.048 | N/A |
| ptv3_ctx_universal_8gpu_100ep | 0.406 @ep41 | 0.539 @ep7 | 0.354 | 0.443 | 3.403 |

## 3. Per-Run Trajectory Summaries

For each metric: Start | Best (ep) | Final | [sparkline]

### ptv3_baseline_8gpu_100ep
*None context | 100 ep | lr=0.008 | CE+Lovasz | from scratch*

| Metric | Start | Best (ep) | Final | Trajectory |
|--------|-------|-----------|-------|------------|
| train_loss | 2.883 | 0.443 @ep98 | 0.519 | `█▄▃▃▃▃▃▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁` |
| val_loss | 4.298 | 1.656 @ep36 | 2.624 | `▄▃▃▂▃▂▁▂▂▁▁▁▁▁▃▂▁▁▂▂▂▂▁▂▂` |
| val_mAcc | 0.094 | 0.356 @ep85 | 0.087 | `▁▁▂▄▄▁▄▄▅▅▄▆▅▄▆▇▇▇▇▇▆█▇▇▁` |
| val_allAcc | 0.214 | 0.614 @ep36 | 0.236 | `▁▁▃▄▄▂▅▇▅▄▄▆▅▄▅▅▅▅▆▅▅▆▅▅▂` |

Last 20-ep mAcc spread: **0.2693** (noisy)
Instability dips at ep: 21, 100

### ptv3_baseline_8gpu_200ep
*None context | 200 ep | lr=0.008 | CE+Lovasz | from scratch*

| Metric | Start | Best (ep) | Final | Trajectory |
|--------|-------|-----------|-------|------------|
| train_loss | 3.044 | 0.260 @ep170 | 0.341 | `█▃▃▃▃▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁` |
| val_loss | 4.561 | 1.771 @ep54 | 3.970 | `▄▄▁▃▁▂▁▁▁▃▁█▂▄▃▂▃▂▃▂▃▃▃▃▃` |
| val_mAcc | 0.085 | 0.353 @ep181 | 0.344 | `▁▁▆▄▅▄▅▆▄▆▆▄▆▅▇▅▆▇▇▇▇▇▇▇▇` |
| val_allAcc | 0.227 | 0.621 @ep54 | 0.440 | `▁▁▄▂▆▃▃▅▃▄▆▂▄▃▄▄▄▅▄▅▅▄▅▅▅` |

Last 20-ep mAcc spread: **0.0284** (moderate)
No collapse or severe dips detected.

### ptv3_normals_8gpu_200ep
*None (normals) context | 200 ep | lr=0.008 | CE+Lovasz | from scratch*

| Metric | Start | Best (ep) | Final | Trajectory |
|--------|-------|-----------|-------|------------|
| train_loss | 2.950 | 0.268 @ep192 | 0.351 | `█▄▃▃▃▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁` |
| val_loss | 3.795 | 1.874 @ep53 | 3.084 | `▄▆▂▂▂▁▁▂▁▂▂▂▃▂▂▂▂▂▂▂▂▃▂▃▃` |
| val_mAcc | 0.087 | 0.344 @ep132 | 0.320 | `▁▁▄▃▃▆▅▅▅▅▇▆▅▆▆▇▆▆▇▇▇▇▇▇▇` |
| val_allAcc | 0.184 | 0.619 @ep96 | 0.497 | `▁▂▃▄▄▄▄▄▆▄▅▅▃▅▅▆▇▅▇▅▆▆▆▆▆` |

Last 20-ep mAcc spread: **0.0323** (moderate)
Instability dips at ep: 21

### ptv3_ctx_ae_8gpu_100ep
*AE context | 100 ep | lr=0.004 | CE+Lovasz | from scratch*

| Metric | Start | Best (ep) | Final | Trajectory |
|--------|-------|-----------|-------|------------|
| train_loss | 1.858 | 0.106 @ep88 | 0.131 | `█▄▃▂▂▂▂▂▂▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁` |
| val_loss | — | — | N/A | *tfevents missing* |
| val_mAcc | 0.199 | 0.418 @ep55 | 0.396 | `▁▂▅▅▄▆▆▄▄▅▅▅▅▅▅▆▇▆▇▆▇▇▇▇▇` |
| val_allAcc | 0.326 | 0.575 @ep18 | 0.443 | `▂▅▄▃▂▄▅▂▃▂▂▄▂▃▃▄▆▄▄▄▄▅▄▄▄` |

Last 20-ep mAcc spread: **0.0255** (moderate)
No collapse or severe dips detected.

### ptv3_ctx_gpn_8gpu_100ep
*GPN context | 100 ep | lr=0.004 | CE+Lovasz | from scratch*

| Metric | Start | Best (ep) | Final | Trajectory |
|--------|-------|-----------|-------|------------|
| train_loss | 2.086 | 0.149 @ep93 | 0.166 | `█▄▃▃▃▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁` |
| val_loss | 3.949 | 2.187 @ep56 | 3.480 | `▃▃▁▅▂▄▂▂▂█▃▄▁▂▂▃▃▃▂▂▂▃▃▂▃` |
| val_mAcc | 0.116 | 0.346 @ep56 | 0.337 | `▁▂▄▄▅▆▅▅▆▅▅▆▆▅▆▆▆▆▆▇▇▇▇▇▇` |
| val_allAcc | 0.245 | 0.475 @ep56 | 0.442 | `▁▂▅▃▄▃▂▄▄▃▃▅▇▄▆▆▅▅▆▆▆▆▆▆▇` |

Last 20-ep mAcc spread: **0.0176** (tight)
No collapse or severe dips detected.

### ptv3_ctx_sinr_8gpu_100ep
*SINR context | 100 ep | lr=0.004 | CE+Lovasz | from scratch*

| Metric | Start | Best (ep) | Final | Trajectory |
|--------|-------|-----------|-------|------------|
| train_loss | 1.968 | 0.139 @ep100 | 0.139 | `█▄▃▃▃▃▃▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁` |
| val_loss | 2.801 | 1.918 @ep14 | 2.662 | `▃▂▁▂▁▁▃▂▂▁▁▃▁▂▂▂▁▃▂▂▃▂▂▂▂` |
| val_mAcc | 0.222 | 0.434 @ep65 | 0.394 | `▃▂▄▅▆▆▅▅▅▆▆▄▅▆▅▆█▆▆▆▇▇▇▇▇` |
| val_allAcc | 0.361 | 0.558 @ep71 | 0.484 | `▄▄▇▆▆▆▄▅▅▅▆▅▅▆▅▆▇▅▆▆▆▇▆▆▆` |

Last 20-ep mAcc spread: **0.0243** (moderate)
Instability dips at ep: 4

### ptv3_ctx_topo_8gpu_100ep
*Topo context | 100 ep | lr=0.004 | CE+Lovasz | from scratch*

| Metric | Start | Best (ep) | Final | Trajectory |
|--------|-------|-----------|-------|------------|
| train_loss | 2.224 | 0.123 @ep86 | 0.184 | `█▄▃▃▃▂▂▂▂▂▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁` |
| val_loss | 3.763 | 2.678 @ep9 | 6.099 | `▁▆▁▁▂▂▃▁▂▃▁▁▁▁▃▂▂▂▁▂▂▂▂▂▂` |
| val_mAcc | 0.115 | 0.346 @ep58 | 0.311 | `▂▁▃▂▂▅▂▅▅▆▄▃▆▅▅▆▆▆▆▆▇▆▇▆▇` |
| val_allAcc | 0.285 | 0.540 @ep29 | 0.368 | `▄▃▄▅▂▂▃█▄▅▅▄▆▅▄▅▅▅▅▅▅▅▅▅▅` |

Last 20-ep mAcc spread: **0.0480** (moderate)
Instability dips at ep: 10–11, 14–15, 20, 27

### ptv3_ctx_ae_pretrained_8gpu_200ep
*AE context | 200 ep | lr=0.004 | CE+Lovasz | pretrained backbone*

| Metric | Start | Best (ep) | Final | Trajectory |
|--------|-------|-----------|-------|------------|
| train_loss | 0.936 | 0.140 @ep99 | 0.240 | `█▃▃▃▂▃▂▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁` |
| val_loss | 2.816 | 1.794 @ep41 | 4.753 | `▂▁▁▁▃▅▃▁▂▃▄▂▁▇▃▁?????????` |
| val_mAcc | 0.310 | 0.429 @ep70 | 0.077 | `▅▆▆▆▅▆▆▇▆▇▇▇█▇▆▆▂▁▁▁▁▁▁▁▁` |
| val_allAcc | 0.448 | 0.626 @ep70 | 0.048 | `▅▆▆▆▄▅▅▇▆▅▇▆█▆▅▅▂▁▁▁▁▁▁▁▁` |

Last 20-ep mAcc spread: **0.0000** (tight)
**COLLAPSE ep126–145** (duration 19 ep, did not recover)
Instability dips at ep: 92, 98–115

### ptv3_ctx_sinr_pretrained_8gpu_200ep
*SINR context | 200 ep | lr=0.004 | CE+Lovasz | pretrained backbone*

| Metric | Start | Best (ep) | Final | Trajectory |
|--------|-------|-----------|-------|------------|
| train_loss | 0.984 | 0.244 @ep71 | 0.292 | `█▂▃▃▂▂▂▃▂▂▂▁▂▁▁▁▁▁▁▁▁▁▁▁▁` |
| val_loss | 3.264 | 2.087 @ep34 | 3.613 | `▃▄█▄▄▁▅▁▂▃▃▃▃▄▃▆▄▂???????` |
| val_mAcc | 0.307 | 0.383 @ep51 | 0.077 | `▆▆▅▆▆▇▆▆▅▇▇▆▇█▆▆▆▅▃▁▁▁▁▁▁` |
| val_allAcc | 0.410 | 0.625 @ep36 | 0.048 | `▅▅▄▅▅▆▅▅▄▅▅▅▅▅▄▆▅▄▂▁▁▁▁▁▁` |

Last 20-ep mAcc spread: **0.0000** (tight)
**COLLAPSE ep74–97** (duration 23 ep, did not recover)

### ptv3_ctx_ae_pretrained_wce_8gpu_100ep
*AE context | 100 ep | lr=0.004 | WCE | pretrained backbone*

| Metric | Start | Best (ep) | Final | Trajectory |
|--------|-------|-----------|-------|------------|
| train_loss | 0.847 | 0.053 @ep84 | 0.070 | `█▄▃▃▃▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁` |
| val_loss | 2.759 | 1.034 @ep14 | 3.203 | `▄▅▂▃▃▄▆▆▅▃▃▄▄▃▆▄▃▄▅▅▄▅▆▅▅` |
| val_mAcc | 0.309 | 0.444 @ep14 | 0.391 | `▂▃▄▅▄▂▁▂▅▇▅▃▆▆▅▆▆▆▅▅▆▆▅▅▅` |
| val_allAcc | 0.403 | 0.673 @ep14 | 0.538 | `▂▂▆▄▆▂▂▃▃▆▄▃▃▆▄▅▅▄▅▅▅▅▅▅▅` |

Last 20-ep mAcc spread: **0.0207** (moderate)
No collapse or severe dips detected.

### ptv3_ctx_ae_pretrained_wce_8gpu_300ep
*AE context | 300 ep | lr=0.004 | WCE | pretrained backbone*

| Metric | Start | Best (ep) | Final | Trajectory |
|--------|-------|-----------|-------|------------|
| train_loss | 0.829 | 0.143 @ep67 | 0.313 | `█▂▂▃▂▂▃▂▃▂▂▂▂▁▁▁▁▁▁▁▁▂▁▁▂` |
| val_loss | 2.559 | 1.311 @ep27 | 30.940 | `▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▄?????` |
| val_mAcc | 0.310 | 0.418 @ep31 | 0.077 | `▅▆▇▇▆▇▇▆█▆▇▆▄▆▇▅▇▆▁▁▁▁▁▁▁` |
| val_allAcc | 0.408 | 0.666 @ep24 | 0.048 | `▅▅▅▅▅▅█▇▆▅▆▅▃▅▇▄▅▅▁▁▁▁▁▁▁` |

Last 20-ep mAcc spread: **0.0000** (tight)
**COLLAPSE ep71–73** (duration 2 ep, recovered)
**COLLAPSE ep75–96** (duration 21 ep, did not recover)

### ptv3_ctx_ae_sinr_cat_8gpu_100ep
*AE+SINR (cat) context | 200 ep | lr=0.001 | CE | pretrained backbone*

*No train.log or tfevents available.*

### ptv3_ctx_ae_sinr_cat_4gpu_200ep
*AE+SINR (cat) context | 200 ep | lr=0.001 | CE | pretrained backbone*

| Metric | Start | Best (ep) | Final | Trajectory |
|--------|-------|-----------|-------|------------|
| train_loss | 0.742 | 0.039 @ep197 | 0.057 | `█▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁` |
| val_loss | 1.738 | 1.233 @ep19 | 2.696 | `▂▄▁▃▂▂▅▂▃▇▃▃▅▄▃▄▄▄▄▄▄▄▅▄▅` |
| val_mAcc | 0.321 | 0.415 @ep112 | 0.387 | `▁▄▅▂▅▄▃▅▃▃▆▇▅▆▃▅▄▅▅▆▅▆▆▅▆` |
| val_allAcc | 0.444 | 0.653 @ep34 | 0.523 | `▂▂▇▂▄▄▂▄▄▂▄▄▃▅▅▄▄▄▅▄▄▄▄▄▄` |

Last 20-ep mAcc spread: **0.0390** (moderate)
No collapse or severe dips detected.

### ptv3_ctx_ae_sinr_cat_scratch_4gpu_200ep
*AE+SINR (cat) context | 200 ep | lr=0.001 | CE | from scratch*

| Metric | Start | Best (ep) | Final | Trajectory |
|--------|-------|-----------|-------|------------|
| train_loss | 1.169 | 0.236 @ep28 | 0.948 | `█▄▃▂▂▁▂▁▂▁▁▁▁▁▁▁▁▁▁▁▁▂▁▃▆` |
| val_loss | 1.737 | 1.065 @ep9 | 156.391 | `▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▄??` |
| val_mAcc | 0.308 | 0.390 @ep23 | 0.077 | `▆▇▆▆▆▆▅▄▄▇▅▄█▄▆▁▁▂▁▁▁▁▁▁▁` |
| val_allAcc | 0.455 | 0.599 @ep9 | 0.048 | `▆▆▆▅▆▄▄▂▃▅▄▆▆▅▆▃▁▁▁▃▃▁▃▁▁` |

Last 20-ep mAcc spread: **0.2598** (noisy)
**COLLAPSE ep36–47** (duration 11 ep, did not recover)

### ptv3_ctx_universal_8gpu_100ep
*Universal (all 4) context | 50 ep | lr=0.001 | CE+Lovasz | pretrained backbone*

| Metric | Start | Best (ep) | Final | Trajectory |
|--------|-------|-----------|-------|------------|
| train_loss | 2.141 | 0.469 @ep46 | 0.532 | `█▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁` |
| val_loss | 3.403 | 3.403 @ep1 | 3.403 | `▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁` |
| val_mAcc | 0.317 | 0.406 @ep41 | 0.354 | `▁▁▂▂▄▂▃▂▄▃▆▂▁▆▂▄▂▄▃▄█▄▂▂▄` |
| val_allAcc | 0.450 | 0.539 @ep7 | 0.443 | `▃▄▅█▄▁▃▄▃▂▅▃▃▄▂▄▄▄▃▂▃▂▂▂▂` |

Last 20-ep mAcc spread: **0.0782** (noisy)
No collapse or severe dips detected.

## 4. Collapse & Instability Warnings

- **ptv3_baseline_8gpu_100ep**: instability dips at ep 21, 100
- **ptv3_normals_8gpu_200ep**: instability dips at ep 21
- **ptv3_ctx_sinr_8gpu_100ep**: instability dips at ep 4
- **ptv3_ctx_topo_8gpu_100ep**: instability dips at ep 10–11, 14–15, 20, 27
- **ptv3_ctx_ae_pretrained_8gpu_200ep**: COLLAPSE ep126–145 (duration 19 ep, permanent — model did not recover)
- **ptv3_ctx_ae_pretrained_8gpu_200ep**: instability dips at ep 92, 98–115
- **ptv3_ctx_sinr_pretrained_8gpu_200ep**: COLLAPSE ep74–97 (duration 23 ep, permanent — model did not recover)
- **ptv3_ctx_ae_pretrained_wce_8gpu_300ep**: COLLAPSE ep71–73 (duration 2 ep, recovered)
- **ptv3_ctx_ae_pretrained_wce_8gpu_300ep**: COLLAPSE ep75–96 (duration 21 ep, permanent — model did not recover)
- **ptv3_ctx_ae_sinr_cat_scratch_4gpu_200ep**: COLLAPSE ep36–47 (duration 11 ep, permanent — model did not recover)

## 5. Cross-Run Comparison Notes

**Ranking by best val mAcc (descending):**

| Rank | Run | Best mAcc | Context | Pretrained | Loss |
|------|-----|-----------|---------|------------|------|
| 1 | ptv3_ctx_ae_pretrained_wce_8gpu_100ep | 0.444 | AE | Yes | WCE |
| 2 | ptv3_ctx_sinr_8gpu_100ep | 0.434 | SINR | No | CE+Lovasz |
| 3 | ptv3_ctx_ae_pretrained_8gpu_200ep | 0.429 | AE | Yes | CE+Lovasz |
| 4 | ptv3_ctx_ae_pretrained_wce_8gpu_300ep | 0.418 | AE | Yes | WCE |
| 5 | ptv3_ctx_ae_8gpu_100ep | 0.418 | AE | No | CE+Lovasz |
| 6 | ptv3_ctx_ae_sinr_cat_4gpu_200ep | 0.415 | AE+SINR (cat) | Yes | CE |
| 7 | ptv3_ctx_universal_8gpu_100ep | 0.406 | Universal (all 4) | Yes | CE+Lovasz |
| 8 | ptv3_ctx_ae_sinr_cat_scratch_4gpu_200ep | 0.390 | AE+SINR (cat) | No | CE |
| 9 | ptv3_ctx_sinr_pretrained_8gpu_200ep | 0.383 | SINR | Yes | CE+Lovasz |
| 10 | ptv3_baseline_8gpu_100ep | 0.356 | None | No | CE+Lovasz |
| 11 | ptv3_baseline_8gpu_200ep | 0.353 | None | No | CE+Lovasz |
| 12 | ptv3_ctx_gpn_8gpu_100ep | 0.346 | GPN | No | CE+Lovasz |
| 13 | ptv3_ctx_topo_8gpu_100ep | 0.346 | Topo | No | CE+Lovasz |
| 14 | ptv3_normals_8gpu_200ep | 0.344 | None (normals) | No | CE+Lovasz |

**Key observations:**

**Baselines:**
- ptv3_baseline_8gpu_100ep: best mAcc=0.356 @ep85, final=0.087
- ptv3_baseline_8gpu_200ep: best mAcc=0.353 @ep181, final=0.344
- ptv3_normals_8gpu_200ep: best mAcc=0.344 @ep132, final=0.320

**Single context, from scratch (100ep):**
- ptv3_ctx_sinr_8gpu_100ep: best mAcc=0.434 @ep65, final=0.394
- ptv3_ctx_ae_8gpu_100ep: best mAcc=0.418 @ep55, final=0.396
- ptv3_ctx_gpn_8gpu_100ep: best mAcc=0.346 @ep56, final=0.337
- ptv3_ctx_topo_8gpu_100ep: best mAcc=0.346 @ep58, final=0.311

**Pretrained backbone runs:**
- ptv3_ctx_ae_pretrained_wce_8gpu_100ep (100ep, WCE): best mAcc=0.444 @ep14, best allAcc=0.673 @ep14
- ptv3_ctx_ae_pretrained_8gpu_200ep (200ep, CE+Lovasz): best mAcc=0.429 @ep70, best allAcc=0.626 @ep70
- ptv3_ctx_ae_pretrained_wce_8gpu_300ep (300ep, WCE): best mAcc=0.418 @ep31, best allAcc=0.666 @ep24
- ptv3_ctx_ae_sinr_cat_4gpu_200ep (200ep, CE): best mAcc=0.415 @ep112, best allAcc=0.653 @ep34
- ptv3_ctx_universal_8gpu_100ep (50ep, CE+Lovasz): best mAcc=0.406 @ep41, best allAcc=0.539 @ep7
- ptv3_ctx_sinr_pretrained_8gpu_200ep (200ep, CE+Lovasz): best mAcc=0.383 @ep51, best allAcc=0.625 @ep36
- ptv3_ctx_ae_sinr_cat_8gpu_100ep (200ep, CE): best mAcc=N/A @epNone, best allAcc=N/A @epNone
