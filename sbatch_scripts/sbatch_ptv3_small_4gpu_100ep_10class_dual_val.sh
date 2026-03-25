#!/bin/bash
#SBATCH --job-name=ptv3_10class_dual_val
#SBATCH --time=01:00:00
#SBATCH --account=plgteecls-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=ptv3_small_4gpu_100ep_10class_dual_val.%j.out
#SBATCH --error=ptv3_small_4gpu_100ep_10class_dual_val.%j.err

set -euo pipefail

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

ROOT=/net/pr2/projects/plgrid/plggtreeseg
WD=$ROOT/context_classification_ptv3/Pointcept
SIF=$ROOT/ptv3.sif

CONFIG=$WD/configs/standardized_dataset/cls-ptv3-baseline-genus-small-10class-dual-val.py
EXP_DIR=$WD/exp/snapshot_10class_dual_val/ptv3_small_4gpu_100ep

ls -ld "$ROOT" "$WD" || true

export APPTAINER_BINDPATH="$ROOT"

WANDB_KEY_FILE=$ROOT/wandb_key.txt
export WANDB_API_KEY="$(cat ${WANDB_KEY_FILE})"
export WANDB_ENTITY="makskulicki"
export WANDB_PROJECT="context_classification"
export SPCONV_ALGO=native  # skip autotuning to avoid OOM on first forward pass

# NOTE: to resume a future run correctly, pass BOTH:
#   resume=True weight="$EXP_DIR/model/model_last.pth"
#
# To evaluate on val_ood instead of val_id, add:
#   "data.val.split=val_ood"

apptainer exec --nv --pwd "$WD" \
  --env PYTHONPATH="$WD:${PYTHONPATH:-}" "$SIF" \
  python -u tools/train.py \
    --config-file "$CONFIG" \
    --num-gpus 4 \
    --options save_path="$EXP_DIR" epoch=50 eval_epoch=50 num_worker=16

echo "Job completed at: $(date)"
