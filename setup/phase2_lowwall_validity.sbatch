#!/bin/bash
# Phase 0b: does OpenVLA still CRASH into the lowered wall (benchmark stays valid)?
#   cd ~/crash_bench/setup && sbatch phase2_lowwall_validity.sbatch
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --job-name=lowwall_valid
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
nvidia-smi -L
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/crash_bench"
python scripts/phase2_lowwall_validity.py \
  --scenarios 'scenarios/env_collision__T5__libero_spatial_t0_wall_d62' \
  --heights 0.10 0.12 0.14
echo "DONE @ $(date)"
