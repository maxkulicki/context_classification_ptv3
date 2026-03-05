#!/bin/bash
# District-level 6-fold cross-validation with PTv3 + AlphaEarth + BDL
#
# Three-stream fusion: point cloud + satellite context + fertility/moisture.
# Each fold holds out one district as the test set.
#
# Usage:
#   bash run_full_context_kfold.sh

set -e

cd "$(dirname "$0")" || exit 1

DISTRICTS=("Gorlice" "Herby" "Katrynka" "Milicz" "Piensk" "Suprasl")
CONFIG="configs/treescanpl/cls-ptv3-v1m1-0-base-context-full-kfold.py"
WEIGHT="exp/forspecies20k/cls-ptv3-v1m1-0-base/model/model_best.pth"
PYTHON=python

echo "======================================================"
echo "District-level 6-fold CV: PTv3 + AlphaEarth + BDL"
echo "Config: ${CONFIG}"
echo "Weight: ${WEIGHT}"
echo "======================================================"

for FOLD in 0 1 2 3 4 5; do
    DISTRICT=${DISTRICTS[$FOLD]}
    EXP_NAME="cls-ptv3-v1m1-0-base-context-full-kfold-fold${FOLD}-${DISTRICT}"
    EXP_DIR="exp/treescanpl/${EXP_NAME}"

    echo ""
    echo "===== Fold ${FOLD}: Hold out ${DISTRICT} ====="
    echo "Experiment dir: ${EXP_DIR}"

    rm -rf "${EXP_DIR}"
    mkdir -p "${EXP_DIR}/model" "${EXP_DIR}/code"

    cp -r scripts tools pointcept "${EXP_DIR}/code"

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
echo "Results in: exp/treescanpl/cls-ptv3-v1m1-0-base-context-full-kfold-fold*/"
echo "======================================================"
