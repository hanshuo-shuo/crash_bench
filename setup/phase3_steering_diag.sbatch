#!/bin/bash
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --job-name=crashbench_steerdiag
#SBATCH --output=%x_%j.log
set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
cd "$HOME/crash_bench"
python -u scripts/phase3_steering_diag.py
echo "DONE @ $(date)"
