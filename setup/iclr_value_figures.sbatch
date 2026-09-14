#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=cb_iclr_figures
#SBATCH --output=%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/crash_bench}"
module purge
module load mamba/24.3.0 git/2.47.0-gcc-12.4.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
figure_commit=$(git rev-parse HEAD)
test -z "$(git status --porcelain=v1 --untracked-files=all)"
figure_name="${figure_commit:0:12}_iclr_figures_${SLURM_JOB_ID}"
figure_output="/projects/p33100/siosio/crashbench_repeat_value/$figure_name"
test ! -e "$figure_output"
ln -s "$figure_output" "results/repeat_value/$figure_name"
python scripts/paper/render_iclr_value_figures.py \
    --fresh-run /projects/p33100/siosio/crashbench_repeat_value/2c5e6163a6bc_fresh_6243668 \
    --output "$figure_output"
