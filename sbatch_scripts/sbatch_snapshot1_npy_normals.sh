#!/bin/bash
#SBATCH --job-name=snap1_normals
#SBATCH --time=02:00:00
#SBATCH --account=plgteecls-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=snap1_normals.%j.out
#SBATCH --error=snap1_normals.%j.err

set -euo pipefail

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

ROOT=/net/pr2/projects/plgrid/plggtreeseg
WD=$ROOT/context_classification_ptv3
SIF=$ROOT/ptv3_laspy.sif

INPUT_ROOT=$ROOT/data/snapshot_1
OUTPUT_ROOT=$ROOT/data/snapshot_1_npy_normals

ls -ld "$ROOT" "$INPUT_ROOT" || true
export APPTAINER_BINDPATH="$ROOT"

apptainer exec --nv --pwd "$WD" \
  --env PYTHONPATH="$WD/Pointcept:${PYTHONPATH:-}" "$SIF" \
  python -u "$WD/convert_laz_to_npy_normals.py" \
    --input_root "$INPUT_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --voxel_size 0.05 \
    --n_final 8192 \
    --normal_k 20 \
    --workers 60 \
    --split_root "$ROOT/data/snapshot_1_npy_fps8192"

echo "Job completed at: $(date)"
