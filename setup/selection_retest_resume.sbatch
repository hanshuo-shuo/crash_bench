#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=cb_selection_resume
#SBATCH --output=%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/crash_bench}"
module purge
module load mamba/24.3.0 git/2.37.2
source /projects/p33100/siosio/envs/openpi/bin/activate
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH="$HOME/crash_bench"
export OPENPI_DATA_HOME=/projects/p33100/siosio/openpi_assets
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
retest_commit=$(git rev-parse HEAD)
test -z "$(git status --porcelain=v1 --untracked-files=all)"
retest_name="${retest_commit:0:12}_${SLURM_JOB_ID}"
retest_output="/projects/p33100/siosio/crashbench_selection_retest/$retest_name"
test ! -e "$retest_output"
test ! -e "results/selection_retest/$retest_name"
ln -s "$retest_output" "results/selection_retest/$retest_name"
python scripts/expansion/resume_selection_retest.py \
  --parent-dir results/selection_retest/1dfc6046ae9c_5680180 --output-dir "$retest_output"
python scripts/expansion/analyze_selection_retest.py --run-dir "$retest_output"
nvidia-smi > "$retest_output/gpu.txt"
