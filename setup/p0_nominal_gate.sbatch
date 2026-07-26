#!/bin/bash
# P0 authoring stage A: provenance-complete nominal gate for candidate second tasks.
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=crashbench_p0_gate
#SBATCH --output=%x_%j.log

set -euo pipefail
: "${CB_CHECKPOINT_REVISION:?set the exact immutable checkpoint revision}"
: "${CB_GATE_OUT:?set a new ignored gate result path under results/p0_runs/}"
module purge
module load git/2.47.0-gcc-12.4.0
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CB_GIT_EXECUTABLE="$(command -v git)"
cd "$HOME/crash_bench"
PYTHON="$HOME/crash_bench/envs/openvla/bin/python"
test -x "$PYTHON"
test -x "$CB_GIT_EXECUTABLE"
"$PYTHON" -c 'import sys, numpy; print(f"python={sys.executable}"); print(f"numpy={numpy.__version__}")'
"$CB_GIT_EXECUTABLE" --version
"$PYTHON" scripts/nominal_task_gate.py \
  --suite libero_spatial \
  --task_ids "${CB_TASK_IDS:-1,2,3,4,5,6,7,8,9}" \
  --repeats "${CB_GATE_REPEATS:-5}" \
  --checkpoint-revision "$CB_CHECKPOINT_REVISION" \
  --base-seed "${CB_BASE_SEED:-20260726}" \
  --out "$CB_GATE_OUT"
