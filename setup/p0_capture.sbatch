#!/bin/bash
# P0 stage 1: provenance-complete full-arm corridor + raw activation capture.
# Submit through setup/submit_p0.sh so dependencies and paths stay consistent.
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=crashbench_p0_capture
#SBATCH --output=%x_%j.log

set -euo pipefail
: "${CB_P0_CONFIG:?set CB_P0_CONFIG to a committed P0 config}"
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/crash_bench"
python scripts/p0_capture.py --config "$CB_P0_CONFIG"

