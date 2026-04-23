#!/bin/bash
#SBATCH --job-name=eval_test_all
#SBATCH --time=02:00:00
#SBATCH --account=plgteecls-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=eval_test_all_120ep.%j.out
#SBATCH --error=eval_test_all_120ep.%j.err

# ---------------------------------------------------------------------------
# Evaluate all 120-epoch runs on the test set (excluding models that already
# have results in eval_results/).
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

MODELS=(
    "ptv3_sinr_hc100_gauss_120ep|\
$CFG/cls-ptv3-ctx-sinr-hc100-gauss-10class-dual-val-4gpu.py|\
$EXP/ptv3_ctx_sinr_hc100_gauss_4gpu_120ep/model/model_best_ood.pth"

    "ptv3_sinr_hc1000_gauss_120ep|\
$CFG/cls-ptv3-ctx-sinr-hc1000-gauss-10class-dual-val-4gpu.py|\
$EXP/ptv3_ctx_sinr_hc1000_gauss_4gpu_120ep/model/model_best_ood.pth"

    "ptv3_ae_spatial_aug_120ep|\
$CFG/cls-ptv3-ctx-ae-spatial-aug-10class-dual-val-4gpu.py|\
$EXP/ptv3_ctx_ae_spatial_aug_4gpu_120ep/model/model_best_ood.pth"

    "ptv3_ae_temporal_aug_120ep|\
$CFG/cls-ptv3-ctx-ae-temporal-aug-10class-dual-val-4gpu.py|\
$EXP/ptv3_ctx_ae_temporal_aug_4gpu_120ep/model/model_best_ood.pth"

    "ptv3_ae_sinr_cat_120ep|\
$CFG/cls-ptv3-ctx-ae-sinr-cat-10class-dual-val-4gpu-120ep.py|\
$EXP/ptv3_ctx_ae_sinr_cat_4gpu_120ep/model/model_best_ood.pth"

    "ptv3_ae_sinr_cat_aux01_120ep|\
$CFG/cls-ptv3-ctx-ae-sinr-cat-aux01-10class-dual-val-4gpu-120ep.py|\
$EXP/ptv3_ctx_ae_sinr_cat_aux01_4gpu_120ep/model/model_best_ood.pth"

    "ptv3_ae_sinr_cat_aux025_mdrop_120ep|\
$CFG/cls-ptv3-ctx-ae-sinr-cat-aux-mdrop-10class-dual-val-4gpu-120ep.py|\
$EXP/ptv3_ctx_ae_sinr_cat_aux_mdrop_4gpu_120ep/model/model_best_ood.pth"

    "ptv3_ae_sinr_cat_aux05_120ep|\
$CFG/cls-ptv3-ctx-ae-sinr-cat-aux05-10class-dual-val-4gpu-120ep.py|\
$EXP/ptv3_ctx_ae_sinr_cat_aux05_4gpu_120ep/model/model_best_ood.pth"

    "ptv3_ae_sinr_cat_aux1_120ep|\
$CFG/cls-ptv3-ctx-ae-sinr-cat-aux1-10class-dual-val-4gpu-120ep.py|\
$EXP/ptv3_ctx_ae_sinr_cat_aux1_4gpu_120ep/model/model_best_ood.pth"

    "ptv3_ae_sinr_cat_sinrgauss_120ep|\
$CFG/cls-ptv3-ctx-ae-sinr-cat-sinrgauss-10class-dual-val-4gpu-120ep.py|\
$EXP/ptv3_ctx_ae_sinr_cat_sinrgauss_4gpu_120ep/model/model_best_ood.pth"

    "ptv3_ae_sinr_cat_aux05_sinrgauss_120ep|\
$CFG/cls-ptv3-ctx-ae-sinr-cat-aux05-sinrgauss-10class-dual-val-4gpu-120ep.py|\
$EXP/ptv3_ctx_ae_sinr_cat_aux05_sinrgauss_4gpu_120ep/model/model_best_ood.pth"

    "ptv3_ae_sinr_cat_aux05_combaug_120ep|\
$CFG/cls-ptv3-ctx-ae-sinr-cat-aux05-combaug-10class-dual-val-4gpu-120ep.py|\
$EXP/ptv3_ctx_ae_sinr_cat_aux05_combaug_4gpu_120ep/model/model_best_ood.pth"

    "ptv3_ae_sinr_cat_aux05_noaug_120ep|\
$CFG/cls-ptv3-ctx-ae-sinr-cat-aux05-noaug-10class-dual-val-4gpu-120ep.py|\
$EXP/ptv3_ctx_ae_sinr_cat_aux05_noaug_4gpu_120ep/model/model_best_ood.pth"
)

export APPTAINER_BINDPATH="$ROOT"
export SPCONV_ALGO=native

N=${#MODELS[@]}
echo "Models to evaluate: $N"
echo ""

FAILED=()

for i in "${!MODELS[@]}"; do
    entry="${MODELS[$i]}"
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
