#!/bin/bash
# Plot-level split training with PTv3 + AlphaEarth + BDL
#
# Three-stream fusion: point cloud + satellite context + fertility/moisture.
# Uses the standard plot-level train/test split (11 genera incl. Abies).
#
# Usage:
#   bash run_full_context_plotsplit.sh

set -e

cd "$(dirname "$0")" || exit 1

CONFIG="configs/treescanpl/cls-ptv3-v1m1-0-base-context-full.py"
WEIGHT="exp/forspecies20k/cls-ptv3-v1m1-0-base/model/model_best.pth"
EXP_NAME="cls-ptv3-v1m1-0-base-context-full"
EXP_DIR="exp/treescanpl/${EXP_NAME}"
PYTHON=python

echo "======================================================"
echo "Plot-level split: PTv3 + AlphaEarth + BDL"
echo "Config: ${CONFIG}"
echo "Weight: ${WEIGHT}"
echo "Experiment dir: ${EXP_DIR}"
echo "======================================================"

rm -rf "${EXP_DIR}"
mkdir -p "${EXP_DIR}/model" "${EXP_DIR}/code"

cp -r scripts tools pointcept "${EXP_DIR}/code"

PYTHONPATH=./${EXP_DIR}/code ${PYTHON} "${EXP_DIR}/code/tools/train.py" \
    --config-file "${CONFIG}" \
    --num-gpus 1 \
    --options \
        save_path="${EXP_DIR}" \
        weight="${WEIGHT}" \
        epoch=5 \
        eval_epoch=5

echo ""
echo "======================================================"
echo "Training complete."
echo "Results in: ${EXP_DIR}/"
echo "======================================================"
