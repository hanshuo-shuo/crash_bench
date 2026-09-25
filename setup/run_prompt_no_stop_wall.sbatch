#!/bin/bash
# Fixed E13 wall scenes: original hazard prompt, stop-clause deletion, goal-first no-stop.
# Ten scenes x three repeats x three prompts = 90 episodes.
# Submit through setup/submit_prompt_no_stop.sh.
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --job-name=cb_nostop_wall
#SBATCH --output=%x_%j.log

set -euo pipefail
: "${CB_CODE_COMMIT:?submit through setup/submit_prompt_no_stop.sh}"
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/crash_bench"
echo "JOB $SLURM_JOB_ID commit=$CB_CODE_COMMIT on $(hostname) @ $(date)"

python - <<'PY'
import json
from pathlib import Path
p = Path("scenarios/env_collision__T5__libero_spatial_t0_wall_d62/scenario.json")
half_z = float(json.loads(p.read_text())["obstacles"][0]["size"][2])
if abs(half_z - 0.22) > 1e-6:
    raise SystemExit(f"expected frozen tall wall half-height 0.22 m; got {half_z}")
PY

out="results/prompt_no_stop/${CB_CODE_COMMIT}/wall_${SLURM_JOB_ID}.json"
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
  --conditions \
    hazard_specific \
    hazard_specific_no_stop \
    hazard_specific_goal_first_no_stop \
  --repeats 3 \
  --checkpoint-revision 962318cec55ac10993ff0f5f43eda9a270b4c873 \
  --out "$out"
echo "DONE wall job=$SLURM_JOB_ID commit=$CB_CODE_COMMIT out=$out"
