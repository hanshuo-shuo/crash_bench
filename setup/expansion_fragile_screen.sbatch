#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --job-name=cb_exp_fragile
#SBATCH --output=%x_%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/crash_bench}"
: "${CB_EXPANSION_RUN_ID:?wrapper must export CB_EXPANSION_RUN_ID}"
: "${SLURM_ARRAY_TASK_ID:?must submit as array 0-15}"

module purge
module load mamba/24.3.0
source /projects/p33100/siosio/envs/openpi/bin/activate
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH="$HOME/crash_bench"
export OPENPI_DATA_HOME=/projects/p33100/siosio/openpi_assets
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

task_cell=$((SLURM_ARRAY_TASK_ID / 8))
source_index=$((SLURM_ARRAY_TASK_ID % 8))
if [[ "$task_cell" -eq 0 ]]; then
  task_id=0
  mechanical_cell="cell_0_fragile_path_collision_v2_task0.json"
  nominal_cell="cell_0_fragile_path_collision_v2_task0.json"
else
  task_id=2
  mechanical_cell="cell_1_fragile_path_collision_v2_task2.json"
  nominal_cell="cell_1_fragile_path_collision_v2_task2.json"
fi
mechanical_root="results/expansion/d2_mechanical/d79973fd46e3_f2a8d011f62d_20260830T090458Z"
nominal_root="results/expansion/d2_preflight/77ea231ebf97_a1464cf4ddf0_20260830T084407Z"
output="results/expansion/d2_fragile_screen/${CB_EXPANSION_RUN_ID}/task${task_id}_source${source_index}.json"

python scripts/expansion/run_fragile_screen.py \
  --mechanical-cell "$mechanical_root/$mechanical_cell" \
  --nominal-cell "$nominal_root/$nominal_cell" \
  --source-index "$source_index" \
  --output "$output"
