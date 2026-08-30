#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --job-name=cb_exp_d2_nominal
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
export OPENPI_DATA_HOME=/projects/p33100/siosio/openpi_assets
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

read -r mechanism task_id < <(python - "$SLURM_ARRAY_TASK_ID" <<'PY'
import sys
from scripts.expansion.run_nominal_preflight import cell_for_array_index
print(*cell_for_array_index(int(sys.argv[1])))
PY
)
output="results/expansion/d2_preflight/${CB_EXPANSION_RUN_ID}/cell_${SLURM_ARRAY_TASK_ID}_${mechanism}_task${task_id}.json"
python scripts/expansion/run_nominal_preflight.py \
  --source-plan results/expansion/d2_sources/screen_plan_256ce74521254ec9f692b77cb5197b5aa29be73f58d91ce29a6c945841cba7de.json \
  --exposure-registry results/expansion/governance/exposure_registry.json \
  --mechanism-id "$mechanism" \
  --task-id "$task_id" \
  --output "$output"
