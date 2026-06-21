#!/bin/bash
# 跑 OpenVLA 闭环 pilot,出 crash-rate (PLAN.md Phase 1 / README §11)
#   cd ~/crash_bench/setup && sbatch run_pilot.sbatch
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=crashbench_pilot
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
nvidia-smi -L
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/crash_bench"
python scripts/run_pilot.py \
  --scenarios scenarios \
  --checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --unnorm_key libero_spatial \
  --out results/pilot.json \
  --video_dir results/pilot_videos
echo "DONE @ $(date)"
