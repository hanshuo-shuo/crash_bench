#!/bin/bash
# Careful-prompt matrix: 5 frozen on-path walls + their 5 matched off-path twins.
# Three language conditions x K=3 = 90 closed-loop episodes.
# Submit through setup/submit_careful_prompt.sh so CB_CODE_COMMIT is recorded.
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=cb_prompt_wall
#SBATCH --output=%x_%j.log

set -euo pipefail
: "${CB_CODE_COMMIT:?submit through setup/submit_careful_prompt.sh}"
echo "JOB $SLURM_JOB_ID commit=$CB_CODE_COMMIT on $(hostname) @ $(date)"
nvidia-smi -L

module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/crash_bench"

python - <<'PY'
import json
from pathlib import Path
p = Path("scenarios/env_collision__T5__libero_spatial_t0_wall_d62/scenario.json")
d = json.loads(p.read_text())
half_z = float(d["obstacles"][0]["size"][2])
if abs(half_z - 0.22) > 1e-6:
    raise SystemExit(
        "wall prompt matrix requires the restored tall d62 geometry "
        f"(half-height 0.22 m), got {half_z}"
    )
print(f"geometry check passed: d62 half-height={half_z}")
PY

python scripts/careful_prompt_eval.py \
  --hazard wall \
  --scenario-root scenarios \
  --scenario-root scenarios_control \
  --scenario-ids \
    env_collision__T1__libero_spatial_t0_wall_wide \
    env_collision__T5__libero_spatial_t0_wall_d62 \
    env_collision__T5__libero_spatial_t0_wall_d70 \
    env_collision__T5__libero_spatial_t0_wall_d78 \
    env_collision__T5__libero_spatial_t0_wall_d85 \
    ood_control__T5__libero_spatial_t0_v3_00_twin \
    ood_control__T5__libero_spatial_t0_v3_01_twin \
    ood_control__T5__libero_spatial_t0_v3_02_twin \
    ood_control__T5__libero_spatial_t0_v3_03_twin \
    ood_control__T5__libero_spatial_t0_v3_04_twin \
  --conditions vanilla generic_careful hazard_specific \
  --repeats "${CB_REPEATS:-3}" \
  --checkpoint-revision 962318cec55ac10993ff0f5f43eda9a270b4c873 \
  --out "${CB_OUT:-results/careful_prompt/wall_prompt_matrix.json}"

echo "DONE @ $(date)"
