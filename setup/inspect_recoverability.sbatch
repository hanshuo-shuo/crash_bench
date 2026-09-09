#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=recoverability_inspect
#SBATCH --output=%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/crash_bench}"
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python scripts/expansion/inspect_recoverability.py --run results/detour_benefit/a89f755da200_job5791132 --output "results/detour_benefit/recoverability_inspect_job${SLURM_JOB_ID}"
