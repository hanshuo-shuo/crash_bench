#!/bin/bash
# Phase 2-④ recon: locate in-distribution (no-wall) crash hazards in libero_10 kitchen scenes
#   cd ~/crash_bench/setup && sbatch phase2_recon_indist.sbatch
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --job-name=recon_indist
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
nvidia-smi -L
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/crash_bench"
python scripts/phase2_recon_indist.py
echo "DONE @ $(date)"
