#!/bin/bash
# Phase 2-④ grasp_instability build: perturb the held bowl + feasibility-filter via HOLD probe
#   cd ~/crash_bench/setup && sbatch phase2_grasp_build.sbatch
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --job-name=grasp_build
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
nvidia-smi -L
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/crash_bench"
python scripts/phase2_grasp_build.py
echo "DONE @ $(date)"
