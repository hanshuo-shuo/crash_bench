#!/bin/bash
# 补渲 d70 的 detour(绕开)视频 —— phase2_witness 里唯一缺的那个
#   cd ~/crash_bench/setup && sbatch render_d70_detour.sbatch
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --job-name=d70_detour
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
nvidia-smi -L
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/crash_bench"
python scripts/render_d70_detour.py
echo "DONE @ $(date)"
