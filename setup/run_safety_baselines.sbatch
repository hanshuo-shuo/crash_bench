#!/bin/bash
# OpenVLA comparison excluding the two-GPU Qwen monitor (see run_vlm_monitor.sbatch).
# Defaults: five frozen on-path walls x K=3.
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=crashbench_safety
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
nvidia-smi -L
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/crash_bench"

python scripts/safety_baseline_suite.py \
  --conditions vanilla prompted_careful safety_filter oracle_stop \
  --repeats "${CB_REPEATS:-3}" \
  --out "${CB_OUT:-results/safety_baselines_nonvlm.json}" \
  --video-dir "${CB_VIDEO_DIR:-results/safety_baseline_videos}"

echo "DONE @ $(date)"
