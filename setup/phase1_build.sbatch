#!/bin/bash
# 探查 LIBERO 状态布局 + 构造 pre-crash 场景 + 渲染初始帧(无 OpenVLA,快)
#   cd ~/crash_bench/setup && sbatch phase1_build.sbatch
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=phase1_build
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/crash_bench"
python scripts/phase1_build_scenarios.py
echo "DONE @ $(date)"
