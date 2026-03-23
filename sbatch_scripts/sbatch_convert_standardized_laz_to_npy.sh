#!/bin/bash
#SBATCH --job-name=std_laz2npy
#SBATCH --time=02:30:00
#SBATCH --account=plgteecls-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=std_laz2npy.%j.out
#SBATCH --error=std_laz2npy.%j.err

set -euo pipefail

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# module load apptainer  # if required by your cluster

ROOT=/net/pr2/projects/plgrid/plggtreeseg
WD=$ROOT/context_classification_ptv3
SIF=$ROOT/ptv3_laspy.sif

INPUT_ROOT=$ROOT/data/standardized_dataset
OUTPUT_ROOT=$ROOT/data/standardized_dataset_npy_fps8192

# Force autofs to mount before container enters its own mount namespace
ls -ld "$ROOT" "$INPUT_ROOT" "$OUTPUT_ROOT" || true

# Make these paths visible inside the container (same absolute paths)
export APPTAINER_BINDPATH="$ROOT"

# Convert LAZ -> NPY with 2cm voxelization and FPS to 8192 points
apptainer exec --nv --pwd "$WD" \
  --env PYTHONPATH="$WD/Pointcept:${PYTHONPATH:-}" "$SIF" \
  python -u "$WD/convert_standardized_laz_to_npy.py" \
    --input_root "$INPUT_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --overwrite \
    --voxel_size 0.02 \
    --fps_points 8192

echo "Job completed at: $(date)"
