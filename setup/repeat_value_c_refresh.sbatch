#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --job-name=cb_value_c_refresh
#SBATCH --output=%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/crash_bench}"
module purge
module load mamba/24.3.0 git/2.37.2
source /projects/p33100/siosio/envs/openpi/bin/activate
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH="$PWD"
export OPENPI_DATA_HOME=/projects/p33100/siosio/openpi_assets
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
repeat_c_commit=$(git rev-parse HEAD)
test -z "$(git status --porcelain=v1 --untracked-files=all)"
repeat_c_name="${repeat_c_commit:0:12}_refresh_${SLURM_JOB_ID}"
repeat_c_output="/projects/p33100/siosio/crashbench_repeat_value/$repeat_c_name"
mkdir -p results/repeat_value /projects/p33100/siosio/crashbench_repeat_value
test ! -e "$repeat_c_output"
ln -s "$repeat_c_output" "results/repeat_value/$repeat_c_name"
python scripts/expansion/run_repeat_value_c.py --panel candidate_refresh \
  --checkpoint-dir "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi0_libero" --output "$repeat_c_output"
python scripts/expansion/analyze_repeat_value_c.py --run-dir "$repeat_c_output"
