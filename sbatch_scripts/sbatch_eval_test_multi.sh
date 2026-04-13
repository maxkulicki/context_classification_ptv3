#!/bin/bash
#SBATCH --job-name=eval_test_multi
#SBATCH --time=00:30:00
#SBATCH --account=plgteecls-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=eval_test_multi.%j.out
#SBATCH --error=eval_test_multi.%j.err

# ---------------------------------------------------------------------------
# Evaluate multiple trained models on the test set in a single job.
# Each model runs sequentially on the same GPU.
#
# Add/remove entries in the MODELS array below. Format per entry:
#   "output_name|config_path|checkpoint_path"
#
# Outputs saved to: $WD/eval_results/{output_name}/
# ---------------------------------------------------------------------------

set -euo pipefail

ROOT=/net/pr2/projects/plgrid/plggtreeseg
WD=$ROOT/context_classification_ptv3/Pointcept
SIF=$ROOT/ptv3.sif
CFG=$WD/configs/standardized_dataset
EXP=$WD/exp/snapshot_10class_dual_val

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# ---------------------------------------------------------------------------
# Model list — edit here
# Format: "output_name|config|checkpoint"
# ---------------------------------------------------------------------------
MODELS=(
    "ptv3_baseline_120ep|\
$CFG/cls-ptv3-baseline-genus-small-10class-dual-val-4gpu-120ep.py|\
$EXP/ptv3_small_4gpu_120ep/model/model_best_ood.pth"

    "ptv3_sinr_gauss_120ep|\
$CFG/cls-ptv3-ctx-sinr-gauss-10class-dual-val-4gpu.py|\
$EXP/ptv3_ctx_sinr_gauss_4gpu_120ep/model/model_best_ood.pth"

    "ptv3_ae_combined_aug_vmf_120ep|\
$CFG/cls-ptv3-ctx-ae-combined-aug-vmf-10class-dual-val-4gpu.py|\
$EXP/ptv3_ctx_ae_combined_aug_vmf_4gpu_120ep/model/model_best_ood.pth"

    "ptv3_ae_sinr_cat_aux_200ep|\
$CFG/cls-ptv3-ctx-ae-sinr-cat-aux-10class-dual-val-4gpu.py|\
$EXP/ptv3_ctx_ae_sinr_cat_aux_4gpu_200ep/model/model_best_ood.pth"
)
# ---------------------------------------------------------------------------

export APPTAINER_BINDPATH="$ROOT"
export SPCONV_ALGO=native

N=${#MODELS[@]}
echo "Models to evaluate: $N"
echo ""

FAILED=()

for i in "${!MODELS[@]}"; do
    entry="${MODELS[$i]}"
    # Strip embedded newlines/backslashes from multiline entries
    entry="${entry//\\$'\n'/}"
    entry="${entry//$'\n'/}"

    IFS='|' read -r NAME CONFIG CHECKPOINT <<< "$entry"
    OUTPUT_DIR="$WD/eval_results/$NAME"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[$((i+1))/$N] $NAME"
    echo "  Config:     $CONFIG"
    echo "  Checkpoint: $CHECKPOINT"
    echo "  Output:     $OUTPUT_DIR"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ ! -f "$CHECKPOINT" ]; then
        echo "SKIP: checkpoint not found — $CHECKPOINT"
        FAILED+=("$NAME (checkpoint missing)")
        echo ""
        continue
    fi

    if [ ! -f "$CONFIG" ]; then
        echo "SKIP: config not found — $CONFIG"
        FAILED+=("$NAME (config missing)")
        echo ""
        continue
    fi

    apptainer exec --nv --pwd "$WD" \
      --env PYTHONPATH="$WD:${PYTHONPATH:-}" "$SIF" \
      python -u eval_test.py \
        --config      "$CONFIG" \
        --checkpoint  "$CHECKPOINT" \
        --output_dir  "$OUTPUT_DIR" \
        --batch_size  256 \
        --num_workers 8
    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "All done at: $(date)"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "SKIPPED (${#FAILED[@]}):"
    for f in "${FAILED[@]}"; do echo "  - $f"; done
else
    echo "All $N models evaluated successfully."
fi
