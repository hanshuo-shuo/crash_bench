#!/bin/bash
# Path 1-1b: activation steering sweep (subtract alpha * probe-direction from the final RMSNorm).
#   cd ~/crash_bench/setup && sbatch phase3_steering.sbatch
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=crashbench_steering
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
nvidia-smi -L
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
cd "$HOME/crash_bench"

echo "== running activation-steering sweep =="
python -u scripts/phase3_steering.py

echo "== making figures =="
python scripts/phase3_steering_figures.py

echo "DONE @ $(date)"
