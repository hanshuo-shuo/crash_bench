#!/bin/bash
# M3: online probe shield expansion.
#
# IMPORTANT: run the M0 geometry hygiene first. The committed scenarios/d62 wall must be
# restored to the original tall geometry; the lowered d62 belongs in scenarios_detour/ only.
# This preflight refuses to mix the lowered detour geometry into the original shield dataset.
#
# Default budget: 5 treatment walls x 3 = 15 baseline + 15 guarded episodes, and 5 no-wall
# plus 6 clear off-path scenes x 4 = 44 benign checks. Override CB_* via --export.
#   cd ~/crash_bench && sbatch setup/phase3_intervention_expanded.sbatch
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --job-name=crashbench_shield_exp
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
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
        f"M0 geometry check failed: {p} has wall half-height {half_z}; "
        "restore the original 0.22 in scenarios/ and keep the lowered copy in scenarios_detour/."
    )
print(f"M0 geometry check passed: d62 wall half-height={half_z}")
PY

export CB_TREAT_K="${CB_TREAT_K:-3}"
export CB_FP_K="${CB_FP_K:-4}"
export CB_OUT="${CB_OUT:-results/intervention_expanded}"
python scripts/phase3_intervention.py

echo "DONE @ $(date)"
