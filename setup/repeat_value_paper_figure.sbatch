#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=cb_value_paper
#SBATCH --output=%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/crash_bench}"
module purge
module load mamba/24.3.0 git/2.37.2
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
paper_value_commit=$(git rev-parse HEAD)
test -z "$(git status --porcelain=v1 --untracked-files=all)"
python scripts/expansion/render_repeat_value_paper.py \
  --detour-analysis /projects/p33100/siosio/crashbench_repeat_value/bcaeda762d6c_detour_6240307/analysis \
  --refresh-analysis /projects/p33100/siosio/crashbench_repeat_value/bcaeda762d6c_refresh_6240316/analysis \
  --output "results/repeat_value/paper_${paper_value_commit:0:12}_${SLURM_JOB_ID}"
