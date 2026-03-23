#!/bin/bash
#SBATCH --job-name=ptv3_ctx_ae_sinr_cat
#SBATCH --time=08:00:00
#SBATCH --account=plgteecls-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=ptv3_ctx_ae_sinr_cat.%j.out
#SBATCH --error=ptv3_ctx_ae_sinr_cat.%j.err

set -euo pipefail

echo "Job started at:  $(date)"
echo "Running on node: $(hostname)"
echo "Job ID:          $SLURM_JOB_ID"

ROOT=/net/pr2/projects/plgrid/plggtreeseg
WD=$ROOT/context_classification_ptv3/Pointcept
SIF=$ROOT/ptv3_laspy.sif

CONFIG=$WD/configs/standardized_dataset/cls-ptv3-ctx-ae-sinr-cat-snapshot1-genus.py
EXP_DIR=$WD/exp/snapshot1/ptv3_ctx_ae_sinr_cat_4gpu_200ep

ls -ld "$ROOT" "$WD" || true

export APPTAINER_BINDPATH="$ROOT"

WANDB_KEY_FILE=$ROOT/wandb_key.txt
export WANDB_API_KEY="$(cat ${WANDB_KEY_FILE})"
export WANDB_ENTITY="makskulicki"
export WANDB_PROJECT="context_classification"
export SPCONV_ALGO=native

apptainer exec --nv --pwd "$WD" \
  --env PYTHONPATH="$WD:${PYTHONPATH:-}" "$SIF" \
  python -u tools/train.py \
    --config-file "$CONFIG" \
    --num-gpus 4 \
    --options save_path="$EXP_DIR" num_worker=16

echo "Job completed at: $(date)"
