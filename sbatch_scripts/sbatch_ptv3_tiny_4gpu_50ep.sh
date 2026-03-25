#!/bin/bash
#SBATCH --job-name=ptv3_tiny_4gpu_50ep
#SBATCH --time=02:00:00
#SBATCH --account=plgteecls-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=ptv3_tiny_4gpu_50ep.%j.out
#SBATCH --error=ptv3_tiny_4gpu_50ep.%j.err

set -euo pipefail

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

ROOT=/net/pr2/projects/plgrid/plggtreeseg
WD=$ROOT/context_classification_ptv3/Pointcept
SIF=$ROOT/ptv3.sif

CONFIG=$WD/configs/standardized_dataset/cls-ptv3-baseline-genus-tiny.py
EXP_DIR=$WD/exp/baseline/ptv3_tiny_4gpu_50ep

ls -ld "$ROOT" "$WD" || true

export APPTAINER_BINDPATH="$ROOT"

WANDB_KEY_FILE=$ROOT/wandb_key.txt
export WANDB_API_KEY="$(cat ${WANDB_KEY_FILE})"
export WANDB_ENTITY="makskulicki"
export WANDB_PROJECT="context_classification"
export SPCONV_ALGO=native  # skip autotuning to avoid OOM on first forward pass

# NOTE: to resume a future run correctly, pass BOTH:
#   resume=True weight="$EXP_DIR/model/model_last.pth"

apptainer exec --nv --pwd "$WD" \
  --env PYTHONPATH="$WD:${PYTHONPATH:-}" "$SIF" \
  python -u tools/train.py \
    --config-file "$CONFIG" \
    --num-gpus 4 \
    --options save_path="$EXP_DIR" epoch=50 eval_epoch=50 num_worker=16

echo "Job completed at: $(date)"
