#!/bin/bash
#SBATCH --job-name=eval_val_splits
#SBATCH --time=01:30:00
#SBATCH --account=plgteecls-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=eval_val_splits_multi.%j.out
#SBATCH --error=eval_val_splits_multi.%j.err

set -euo pipefail

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

ROOT=/net/pr2/projects/plgrid/plggtreeseg
WD=$ROOT/context_classification_ptv3/Pointcept
SIF=$ROOT/ptv3.sif
EXP=$WD/exp/snapshot_10class_dual_val

export APPTAINER_BINDPATH="$ROOT"
export SPCONV_ALGO=native

run_eval() {
    local config="$1"
    local checkpoint="$2"
    local output_dir="$3"
    local split="$4"
    echo ""
    echo "--- Evaluating: $(basename $output_dir) [${split}] ---"
    apptainer exec --nv --pwd "$WD" \
      --env PYTHONPATH="$WD:${PYTHONPATH:-}" "$SIF" \
      python -u eval_test.py \
        --config      "$config" \
        --checkpoint  "$checkpoint" \
        --output_dir  "$output_dir" \
        --split       "$split" \
        --batch_size  256 \
        --num_workers 8
}

for SPLIT in val_id val_ood; do
  run_eval \
    "$EXP/ptv3_small_4gpu_120ep/config.py" \
    "$EXP/ptv3_small_4gpu_120ep/model/model_best_ood.pth" \
    "$WD/eval_results/${SPLIT}/ptv3_baseline_120ep" \
    "$SPLIT"

  run_eval \
    "$EXP/ptv3_ctx_sinr_gauss_4gpu_120ep/config.py" \
    "$EXP/ptv3_ctx_sinr_gauss_4gpu_120ep/model/model_best_ood.pth" \
    "$WD/eval_results/${SPLIT}/ptv3_sinr_gauss_120ep" \
    "$SPLIT"

  run_eval \
    "$EXP/ptv3_ctx_ae_combined_aug_vmf_4gpu_120ep/config.py" \
    "$EXP/ptv3_ctx_ae_combined_aug_vmf_4gpu_120ep/model/model_best_ood.pth" \
    "$WD/eval_results/${SPLIT}/ptv3_ae_combined_aug_vmf_120ep" \
    "$SPLIT"

  run_eval \
    "$EXP/ptv3_ctx_ae_sinr_cat_aux_4gpu_200ep/config.py" \
    "$EXP/ptv3_ctx_ae_sinr_cat_aux_4gpu_200ep/model/model_best_ood.pth" \
    "$WD/eval_results/${SPLIT}/ptv3_ae_sinr_cat_aux_200ep" \
    "$SPLIT"
done

echo ""
echo "All evaluations completed at: $(date)"
