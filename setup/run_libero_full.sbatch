#!/bin/bash
# 坐实 nominal baseline: OpenVLA × LIBERO-Spatial 满 50 trials/task = 500 rollouts
#   提交: cd ~/crash_bench/setup && sbatch run_libero_full.sbatch
#   ~6h (sanity 30ep≈24min);与官方报告的 84.7% (500ep) 直接可比
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --job-name=openvla_libero_full
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
nvidia-smi -L

module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

cd "$HOME/crash_bench/third_party/openvla"
# 500 个 MP4 重定向到 p33100,别占 home 配额
if [ ! -L rollouts ]; then
  mkdir -p /projects/p33100/siosio/rollouts
  [ -d rollouts ] && mv rollouts/* /projects/p33100/siosio/rollouts/ 2>/dev/null || true
  rm -rf rollouts; ln -s /projects/p33100/siosio/rollouts rollouts
fi

python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 50

echo "DONE @ $(date)"
