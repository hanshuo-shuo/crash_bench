#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=cb_repeat_value
#SBATCH --output=%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/crash_bench}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
module purge
module load mamba/24.3.0 git/2.37.2
source activate "$HOME/crash_bench/envs/openvla"
repeat_value_commit=$(git rev-parse HEAD)
test -z "$(git status --porcelain=v1 --untracked-files=all)"
repeat_value_output="results/repeat_value/${repeat_value_commit:0:12}_${SLURM_JOB_ID}"
python scripts/expansion/analyze_repeat_value.py --output-dir "$repeat_value_output"
