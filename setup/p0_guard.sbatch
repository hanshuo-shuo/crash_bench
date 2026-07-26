#!/bin/bash
# P0 stage 3: true held-out online guard/control evaluation and risk--coverage sweep.
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --job-name=crashbench_p0_guard
#SBATCH --output=%x_%j.log

set -euo pipefail
: "${CB_P0_CONFIG:?set CB_P0_CONFIG to the committed P0 config}"
: "${CB_CAPTURE_DIR:?set CB_CAPTURE_DIR to the completed P0 capture directory}"
: "${CB_GUARD_OUT:?set CB_GUARD_OUT to a new result directory}"
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/crash_bench"
PYTHON="$HOME/crash_bench/envs/openvla/bin/python"
test -x "$PYTHON"
"$PYTHON" -c 'import sys, numpy; print(f"python={sys.executable}"); print(f"numpy={numpy.__version__}")'
"$PYTHON" scripts/p0_guard.py \
  --config "$CB_P0_CONFIG" \
  --capture-dir "$CB_CAPTURE_DIR" \
  --out "$CB_GUARD_OUT"
