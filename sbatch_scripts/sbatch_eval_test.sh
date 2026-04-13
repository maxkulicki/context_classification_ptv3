#!/bin/bash
#SBATCH --job-name=eval_test
#SBATCH --time=00:30:00
#SBATCH --account=plgteecls-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=eval_test.%j.out
#SBATCH --error=eval_test.%j.err

# ---------------------------------------------------------------------------
# Usage:
#   Edit CONFIG, CHECKPOINT, and OUTPUT_NAME below, then:
#       cd sbatch_scripts && sbatch sbatch_eval_test.sh
#
# Or pass them at submission time via --export:
#   sbatch --export=CONFIG=...,CHECKPOINT=...,OUTPUT_NAME=... sbatch_eval_test.sh
# ---------------------------------------------------------------------------

set -euo pipefail

ROOT=/net/pr2/projects/plgrid/plggtreeseg
WD=$ROOT/context_classification_ptv3/Pointcept
SIF=$ROOT/ptv3.sif

# --- Edit these for each model to evaluate ---
CONFIG=${CONFIG:-$WD/configs/standardized_dataset/cls-ptv3-ctx-sinr-gauss-10class-dual-val-4gpu.py}
CHECKPOINT=${CHECKPOINT:-$WD/exp/snapshot_10class_dual_val/ptv3_ctx_sinr_gauss_4gpu_120ep/model/model_best.pth}
OUTPUT_NAME=${OUTPUT_NAME:-ptv3_ctx_sinr_gauss}
# --------------------------------------------

OUTPUT_DIR=$WD/eval_results/$OUTPUT_NAME

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Config:     $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Output:     $OUTPUT_DIR"

export APPTAINER_BINDPATH="$ROOT"
export SPCONV_ALGO=native

apptainer exec --nv --pwd "$WD" \
  --env PYTHONPATH="$WD:${PYTHONPATH:-}" "$SIF" \
  python -u eval_test.py \
    --config      "$CONFIG" \
    --checkpoint  "$CHECKPOINT" \
    --output_dir  "$OUTPUT_DIR" \
    --batch_size  256 \
    --num_workers 8

echo "Job completed at: $(date)"
