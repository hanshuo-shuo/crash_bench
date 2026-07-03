#!/bin/bash
# Phase 2 / Category 2 GLASS prototype (ROADMAP §7.1 — first GPU item).
# Injects a tall free-jointed cup on OpenVLA's confident reach path (libero_spatial t0) and
# checks: does the confident reach sweep/topple it, and do object_toppled/object_displaced
# judge it correctly?  Injection primitive pre-validated GPU-free (scripts/probe_glass_inject.py).
#   Full run (default K=10, paired treatment/control per back-fraction):
#     cd ~/crash_bench/setup && sbatch phase2_glass_prototype.sbatch
#   Pilot (faster):  CB_K=3 CB_FRACS=0.35,0.50,0.65 sbatch phase2_glass_prototype.sbatch
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --job-name=glass
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
nvidia-smi -L
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CB_K="${CB_K:-10}"
cd "$HOME/crash_bench"
python scripts/phase2_glass_prototype.py
echo "DONE @ $(date)"
