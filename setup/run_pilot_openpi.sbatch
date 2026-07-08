#!/bin/bash
# Path 3 — π0 (openpi) 跑同一套 on/off-path 协议(第 3 个架构,跨架构复现 crash 100%/0%)。
#   cd ~/crash_bench/setup && sbatch run_pilot_openpi.sbatch
# 先只跑 scenarios/(墙, on-path)做最小 smoke;跑通再补 scenarios_control/(off-path):
#   sbatch --export=ALL,SCEN=scenarios_control,OUT=results/pi0_controls.json,VIDEO=results/pi0_controls_videos run_pilot_openpi.sbatch
# env 是 uv venv @ p33100;crashbench 未装进本 env -> 显式 PYTHONPATH。π0 路径 torch-free,不设 OPENVLA_ROOT。
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=crashbench_pi0_walls
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
nvidia-smi -L

P33100="/projects/p33100/siosio"
source "$P33100/envs/openpi/bin/activate"          # ← uv venv @ p33100
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH="$HOME/crash_bench"               # crashbench 未装进本 env
export HF_HOME="$P33100/huggingface_cache"
export OPENPI_DATA_HOME="$P33100/openpi_assets"     # pi0 checkpoint 缓存 @ p33100
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

cd "$HOME/crash_bench"
SCEN="${SCEN:-scenarios}"
OUT="${OUT:-results/pi0_walls.json}"
VIDEO="${VIDEO:-results/pi0_walls_videos}"
echo "SCEN=$SCEN  OUT=$OUT"
# --checkpoint 传 gs:// 的 pi0_libero;--policy pi0 走 Pi0Policy + 环境 model_family=pi0(torch-free)。
python scripts/run_pilot.py \
  --policy pi0 \
  --scenarios "$SCEN" \
  --checkpoint gs://openpi-assets/checkpoints/pi0_libero \
  --out "$OUT" \
  --video_dir "$VIDEO"
echo "DONE @ $(date)"
