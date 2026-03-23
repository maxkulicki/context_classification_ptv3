#!/bin/bash
#SBATCH --job-name=ptv3_treescanpl_dbg
#SBATCH --time=01:30:00
#SBATCH --account=plgteecls-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=ptv3_treescanpl.%j.out
#SBATCH --error=ptv3_treescanpl.%j.err

set -euo pipefail

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# module load apptainer  # if required by your cluster

ROOT=/net/pr2/projects/plgrid/plggtreeseg
WD=$ROOT/context_classification_ptv3/Pointcept
SIF=$ROOT/ptv3.sif

CONFIG=$WD/configs/treescanpl/cls-ptv3-v1m1-0-base.py
EXP_DIR=$WD/exp/treescanpl/debug_run

ls -ld "$ROOT" "$WD" || true

export APPTAINER_BINDPATH="$ROOT"

# W&B settings (API key file, entity, project)
WANDB_KEY_FILE=/net/pr2/projects/plgrid/plggtreeseg/wandb_key.txt
export WANDB_API_KEY="$(cat ${WANDB_KEY_FILE})"
export WANDB_ENTITY="makskulicki"
export WANDB_PROJECT="context_classification"

apptainer exec --nv --pwd "$WD" \
  --env PYTHONPATH="$WD:${PYTHONPATH:-}" "$SIF" \
  python -u tools/train.py \
    --config-file "$CONFIG" \
    --num-gpus 1 \
    --options save_path="$EXP_DIR" epoch=50 eval_epoch=10

echo "Job completed at: $(date)"
