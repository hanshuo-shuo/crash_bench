#!/bin/bash
# Phase 3: probe-triggered DetourComplete tool hand-off end-to-end on d62 (expect RECOVERY_SUCCESS).
#   cd ~/crash_bench/setup && sbatch phase3_detour_handoff.sbatch
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=phase3_detour
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
nvidia-smi -L
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/crash_bench"
python scripts/phase3_detour_handoff.py
echo "DONE @ $(date)"
