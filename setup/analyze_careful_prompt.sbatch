#!/bin/bash
# CPU-only, dependency-gated aggregation of the wall and glass prompt matrices.
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --job-name=cb_prompt_analysis
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1
cd "$HOME/crash_bench"

python scripts/analyze_careful_prompt.py

echo "DONE @ $(date)"
