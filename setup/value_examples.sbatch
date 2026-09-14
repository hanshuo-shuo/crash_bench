#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --job-name=cb_value_examples
#SBATCH --output=%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/crash_bench}"
module purge
module load mamba/24.3.0 git/2.47.0-gcc-12.4.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
examples_commit=$(git rev-parse HEAD)
test -z "$(git status --porcelain=v1 --untracked-files=all)"
examples_name="${examples_commit:0:12}_value_examples_${SLURM_JOB_ID}"
examples_output="/projects/p33100/siosio/crashbench_repeat_value/$examples_name"
test ! -e "$examples_output"
ln -s "$examples_output" "results/repeat_value/$examples_name"
python scripts/paper/render_value_examples.py --output "$examples_output"
