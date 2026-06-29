#!/bin/bash
# Phase 2-④ no-wall #2 PILOT: fast mechanism check (on-path vs off-path, K=3)
#   cd ~/crash_bench/setup && sbatch phase2_obj_collision_pilot.sbatch
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --job-name=objcol_pilot
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
nvidia-smi -L
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CB_PILOT=1 CB_K=3
cd "$HOME/crash_bench"
python scripts/phase2_obj_collision.py
echo "DONE @ $(date)"
