#!/bin/bash
#SBATCH --job-name=context_only_baseline
#SBATCH --time=02:00:00
#SBATCH --account=plgteecls-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=results/context_only/slurm_%j.out
#SBATCH --error=results/context_only/slurm_%j.err

set -euo pipefail

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

PYTHON=/net/pr2/projects/plgrid/plggtreeseg/conda/envs/context_baseline/bin/python
WD=/net/pr2/projects/plgrid/plggtreeseg/context_classification_ptv3

mkdir -p "$WD/results/context_only"

cd "$WD"

$PYTHON context_only_baseline.py \
    --source all \
    --data_dir data \
    --treescanpl_dir Pointcept/data/treescanpl \
    --results_dir results/context_only \
    --epochs 200 \
    --batch_size 256 \
    --seed 42

echo "Job completed at: $(date)"
