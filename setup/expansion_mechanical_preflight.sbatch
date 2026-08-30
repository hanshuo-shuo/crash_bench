#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=cb_exp_d2_mech
#SBATCH --output=%x_%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/crash_bench}"
: "${CB_EXPANSION_RUN_ID:?wrapper must export CB_EXPANSION_RUN_ID}"
: "${SLURM_ARRAY_TASK_ID:?must submit as array 0-7}"

module purge
module load mamba/24.3.0
source /projects/p33100/siosio/envs/openpi/bin/activate
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH="$HOME/crash_bench"

read -r mechanism task_id < <(python - "$SLURM_ARRAY_TASK_ID" <<'PY'
import sys
from scripts.expansion.run_nominal_preflight import cell_for_array_index
print(*cell_for_array_index(int(sys.argv[1])))
PY
)
nominal_root="results/expansion/d2_preflight/77ea231ebf97_a1464cf4ddf0_20260830T084407Z"
nominal_cell="${nominal_root}/cell_${SLURM_ARRAY_TASK_ID}_${mechanism}_task${task_id}.json"
output="results/expansion/d2_mechanical/${CB_EXPANSION_RUN_ID}/cell_${SLURM_ARRAY_TASK_ID}_${mechanism}_task${task_id}.json"

python scripts/expansion/derive_mechanical_preflight.py \
  --nominal-cell "$nominal_cell" \
  --output "$output"
