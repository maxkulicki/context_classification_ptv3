#!/bin/bash
#SBATCH --job-name=ptv3_ctx_universal_resume
#SBATCH --time=02:00:00
#SBATCH --account=plgteecls-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=ptv3_ctx_universal_resume.%j.out
#SBATCH --error=ptv3_ctx_universal_resume.%j.err

set -euo pipefail

echo "Job started at:  $(date)"
echo "Running on node: $(hostname)"
echo "Job ID:          $SLURM_JOB_ID"

ROOT=/net/pr2/projects/plgrid/plggtreeseg
WD=$ROOT/context_classification_ptv3/Pointcept
SIF=$ROOT/ptv3_laspy.sif

CONFIG=$WD/configs/standardized_dataset/cls-ptv3-ctx-universal-snapshot1-genus-resume.py
EXP_DIR=$WD/exp/snapshot1/ptv3_ctx_universal_8gpu_100ep

ls -ld "$ROOT" "$WD" || true

export APPTAINER_BINDPATH="$ROOT"

WANDB_KEY_FILE=$ROOT/wandb_key.txt
export WANDB_API_KEY="$(cat ${WANDB_KEY_FILE})"
export WANDB_ENTITY="makskulicki"
export WANDB_PROJECT="context_classification"
export SPCONV_ALGO=native

# Resume the existing W&B run (run ID from exp dir wandb/run-*-<id>/)
export WANDB_RUN_ID="q5tbhiqv"
export WANDB_RESUME="must"

apptainer exec --nv --pwd "$WD" \
  --env PYTHONPATH="$WD:${PYTHONPATH:-}" "$SIF" \
  python -u tools/train.py \
    --config-file "$CONFIG" \
    --num-gpus 8 \
    --options save_path="$EXP_DIR" num_worker=32

echo "Job completed at: $(date)"
