#!/bin/bash
# Path 3 对照基线:base OpenVLA 在【当前完全相同】的 scenarios/ + scenarios_control/ 上各跑一遍,
# 与 OFT 做 apples-to-apples 比较(历史 base 控制结果用的是旧场景定义,不可直接比)。
#   cd ~/crash_bench/setup && sbatch run_pilot_base_matched.sbatch
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=crashbench_base_matched
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
nvidia-smi -L
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/crash_bench"

echo "===== BASE walls (on-path) ====="
python scripts/run_pilot.py --policy openvla \
  --scenarios scenarios \
  --checkpoint openvla/openvla-7b-finetuned-libero-spatial --unnorm_key libero_spatial \
  --out results/base_walls_matched.json

echo "===== BASE controls (off-path) ====="
python scripts/run_pilot.py --policy openvla \
  --scenarios scenarios_control \
  --checkpoint openvla/openvla-7b-finetuned-libero-spatial --unnorm_key libero_spatial \
  --out results/base_controls_matched.json
echo "DONE @ $(date)"
