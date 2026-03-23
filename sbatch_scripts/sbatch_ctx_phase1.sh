#!/bin/bash
# Phase 1 context fusion experiments — single sbatch script, source passed as argument.
#
# Usage:
#   sbatch sbatch_ctx_phase1.sh ae
#   sbatch sbatch_ctx_phase1.sh topo
#   sbatch sbatch_ctx_phase1.sh sinr
#   sbatch sbatch_ctx_phase1.sh gpn
#
# Source must be one of: ae  topo  sinr  gpn

#SBATCH --job-name=ptv3_ctx_%a     # overridden below via --job-name flag
#SBATCH --time=05:00:00
#SBATCH --account=plgteecls-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=ptv3_ctx_%x.%j.out
#SBATCH --error=ptv3_ctx_%x.%j.err

set -euo pipefail

SOURCE="${1:-}"
if [[ -z "$SOURCE" ]]; then
    echo "ERROR: no source argument provided."
    echo "Usage: sbatch $0 <ae|topo|sinr|gpn>"
    exit 1
fi

case "$SOURCE" in
    ae|topo|sinr|gpn) ;;
    *)
        echo "ERROR: unknown source '$SOURCE'. Must be one of: ae topo sinr gpn"
        exit 1
        ;;
esac

echo "Job started at:  $(date)"
echo "Running on node: $(hostname)"
echo "Job ID:          $SLURM_JOB_ID"
echo "Context source:  $SOURCE"

ROOT=/net/pr2/projects/plgrid/plggtreeseg
WD=$ROOT/context_classification_ptv3/Pointcept
SIF=$ROOT/ptv3.sif

CONFIG=$WD/configs/standardized_dataset/cls-ptv3-ctx-${SOURCE}-snapshot1-genus.py
EXP_DIR=$WD/exp/snapshot1/ptv3_ctx_${SOURCE}_8gpu_100ep

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
    --num-gpus 8 \
    --options save_path="$EXP_DIR" epoch=100 eval_epoch=100 num_worker=32

echo "Job completed at: $(date)"
