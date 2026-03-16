#!/bin/bash
#SBATCH --job-name=ptv3_std_200ep
#SBATCH --time=02:00:00
#SBATCH --account=plgteecls-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=ptv3_std_test.%j.out
#SBATCH --error=ptv3_std_test.%j.err

set -euo pipefail

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# module load apptainer  # if required by your cluster

# Paths
ROOT=/net/pr2/projects/plgrid/plggtreeseg
WD=$ROOT/context_classification_ptv3/Pointcept
SIF=$ROOT/ptv3.sif

# Config selection (species by default; switch to genus if needed)
#CONFIG=$WD/configs/standardized_dataset/cls-ptv3-v1m1-0-base.py
CONFIG=$WD/configs/standardized_dataset/cls-ptv3-v1m1-0-base-genus.py

# Experiment output directory
EXP_DIR=$WD/exp/standardized_dataset/run_200ep

# Force autofs to mount before container enters its own mount namespace
ls -ld "$ROOT" "$WD" || true

# Make these paths visible inside the container (same absolute paths)
export APPTAINER_BINDPATH="$ROOT"

# W&B settings (API key file, entity, project)
WANDB_KEY_FILE=/net/pr2/projects/plgrid/plggtreeseg/wandb_key.txt
export WANDB_API_KEY="$(cat ${WANDB_KEY_FILE})"
export WANDB_ENTITY="makskulicki"
export WANDB_PROJECT="context_classification"

# Run training (50 epochs test run)
apptainer exec --nv --pwd "$WD" \
  --env PYTHONPATH="$WD:${PYTHONPATH:-}" "$SIF" \
  python -u tools/train.py \
    --config-file "$CONFIG" \
    --num-gpus 1 \
    --options save_path="$EXP_DIR" epoch=200 eval_epoch=20 num_worker=8


echo "Job completed at: $(date)"
