#!/bin/bash
# OpenVLA × LIBERO sanity gate (PLAN.md Phase 0 step 3)
# 跑 stock OpenVLA 在 LIBERO-Spatial 上的 nominal eval,确认 success rate 合理。
# 只有装好环境后再提交: cd ~/crash_bench/setup && sbatch run_libero_sanity.sbatch
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --job-name=openvla_libero_sanity
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
nvidia-smi -L

module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"

export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="/projects/p33100/siosio/pip_cache"
# MuJoCo / robosuite 离屏渲染(计算节点无显示器)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

cd "$HOME/crash_bench/third_party/openvla"
# num_trials_per_task=3 -> libero_spatial 10 tasks = 30 rollouts (快速 sanity;
# 默认 50 会跑 500 episodes 太久)。确认 success rate 合理后再放大。
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 3

echo "DONE @ $(date)"
