#!/bin/bash
# District-level 6-fold cross-validation with PTv3 + AlphaEarth context
#
# Each fold holds out one district as the test set.
# Uses FOR-species20K pretrained backbone, 60 epochs (~10k steps) per fold.
#
# Usage:
#   bash run_district_kfold_context.sh                    # default: projected
#   bash run_district_kfold_context.sh --variant projected
#   bash run_district_kfold_context.sh --variant direct

set -e

# Parse arguments
VARIANT="projected"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)
            VARIANT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash run_district_kfold_context.sh [--variant projected|direct]"
            exit 1
            ;;
    esac
done

if [[ "$VARIANT" != "projected" && "$VARIANT" != "direct" ]]; then
    echo "Error: --variant must be 'projected' or 'direct', got '${VARIANT}'"
    exit 1
fi

cd "$(dirname "$0")" || exit 1
ROOT_DIR=$(pwd)

DISTRICTS=("Gorlice" "Herby" "Katrynka" "Milicz" "Piensk" "Suprasl")
CONFIG="configs/treescanpl/cls-ptv3-v1m1-0-base-context-${VARIANT}-kfold.py"
WEIGHT="exp/forspecies20k/cls-ptv3-v1m1-0-base/model/model_best.pth"
PYTHON=python

echo "======================================================"
echo "District-level 6-fold CV: PTv3 + AlphaEarth (${VARIANT})"
echo "Config: ${CONFIG}"
echo "Weight: ${WEIGHT}"
echo "======================================================"

for FOLD in 0 1 2 3 4 5; do
    DISTRICT=${DISTRICTS[$FOLD]}
    EXP_NAME="cls-ptv3-v1m1-0-base-context-${VARIANT}-kfold-fold${FOLD}-${DISTRICT}"
    EXP_DIR="exp/treescanpl/${EXP_NAME}"

    echo ""
    echo "===== Fold ${FOLD}: Hold out ${DISTRICT} ====="
    echo "Experiment dir: ${EXP_DIR}"

    # Clean previous run if exists
    rm -rf "${EXP_DIR}"
    mkdir -p "${EXP_DIR}/model" "${EXP_DIR}/code"

    # Copy code snapshot (same as train.sh)
    cp -r scripts tools pointcept "${EXP_DIR}/code"

    # Run training
    PYTHONPATH=./${EXP_DIR}/code ${PYTHON} "${EXP_DIR}/code/tools/train.py" \
        --config-file "${CONFIG}" \
        --num-gpus 1 \
        --options \
            save_path="${EXP_DIR}" \
            weight="${WEIGHT}" \
            data.train.fold=${FOLD} \
            data.val.fold=${FOLD} \
            data.test.fold=${FOLD}

    echo "===== Fold ${FOLD} (${DISTRICT}) complete ====="
done

echo ""
echo "======================================================"
echo "All 6 folds complete."
echo "Results in: exp/treescanpl/cls-ptv3-v1m1-0-base-context-${VARIANT}-kfold-fold*/"
echo "======================================================"
