#!/bin/bash
# District-level 6-fold cross-validation with pretrained PTv3
#
# Each fold holds out one district as the test set.
# Uses FOR-species20K pretrained backbone, 60 epochs (~10k steps) per fold.
#
# Usage: bash run_district_kfold.sh

set -e

cd "$(dirname "$0")/Pointcept" || exit 1
ROOT_DIR=$(pwd)

DISTRICTS=("Gorlice" "Herby" "Katrynka" "Milicz" "Piensk" "Suprasl")
CONFIG="configs/treescanpl/cls-ptv3-v1m1-0-base-finetune-kfold.py"
WEIGHT="exp/forspecies20k/cls-ptv3-v1m1-0-base/model/model_best.pth"
PYTHON=python

echo "======================================================"
echo "District-level 6-fold cross-validation"
echo "Config: ${CONFIG}"
echo "Weight: ${WEIGHT}"
echo "======================================================"

for FOLD in 0 1 2 3 4 5; do
    DISTRICT=${DISTRICTS[$FOLD]}
    EXP_NAME="cls-ptv3-v1m1-0-base-finetune-kfold-fold${FOLD}-${DISTRICT}"
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
            data.val.fold=${FOLD}

    echo "===== Fold ${FOLD} (${DISTRICT}) complete ====="
done

echo ""
echo "======================================================"
echo "All 6 folds complete."
echo "Results in: exp/treescanpl/cls-ptv3-v1m1-0-base-finetune-kfold-fold*/"
echo "======================================================"
