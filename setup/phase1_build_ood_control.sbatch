#!/bin/bash
# Author the OOD-but-not-crash control scenarios (README §14 obj. #1).
#   cd ~/crash_bench/setup && sbatch phase1_build_ood_control.sbatch
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --job-name=build_ood_ctrl
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
nvidia-smi -L
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/crash_bench"
python scripts/phase1_build_ood_control.py
echo "DONE @ $(date)"
