#!/bin/bash
#SBATCH --job-name=gradcam_3branch
#SBATCH --time=02:00:00
#SBATCH --account=plgteecls-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=gradcam_3branch.%j.out
#SBATCH --error=gradcam_3branch.%j.err

# ---------------------------------------------------------------------------
# Usage:
#   cd sbatch_scripts && sbatch sbatch_gradcam_3branch.sh
#
# Or override at submission time:
#   sbatch --export=CONFIG=...,CHECKPOINT=...,OUT_DIR=... sbatch_gradcam_3branch.sh
#
# Runs gradcam_attribution_3branch.py over train / val_id / val_ood / test splits
# and writes one CSV per split to OUT_DIR.
# ---------------------------------------------------------------------------

set -euo pipefail

ROOT=/net/pr2/projects/plgrid/plggtreeseg
WD=$ROOT/context_classification_ptv3/Pointcept
SIF=$ROOT/ptv3.sif

CONFIG=${CONFIG:-$WD/configs/standardized_dataset/cls-ptv3-ctx-ae-sinr-cat-aux-10class-dual-val-4gpu.py}
CHECKPOINT=${CHECKPOINT:-$WD/exp/snapshot_10class_dual_val/ptv3_ctx_ae_sinr_cat_aux_4gpu_200ep/model/model_best.pth}
OUT_DIR=${OUT_DIR:-$ROOT/context_classification_ptv3/gradcam_analysis}

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Config:     $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Out dir:    $OUT_DIR"

export APPTAINER_BINDPATH="$ROOT"
export SPCONV_ALGO=native

apptainer exec --nv --pwd "$WD" \
  --env PYTHONPATH="$WD:${PYTHONPATH:-}" "$SIF" \
  python -u tools/gradcam_attribution_3branch.py \
    --config     "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --out-dir    "$OUT_DIR" \
    --prefix     best_model

echo "Job completed at: $(date)"
