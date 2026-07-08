#!/bin/bash
# Path 3 — OpenVLA-OFT 跑同一套 on/off-path 协议(跨架构复现 crash 100%/0%)。
#   cd ~/crash_bench/setup && sbatch run_pilot_oft.sbatch
# 先只跑 scenarios/(墙,on-path)做最小 smoke;跑通再补 scenarios_control/(off-path)。
# env 在 p33100;crashbench 未装进 OFT env -> 显式 PYTHONPATH。
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=crashbench_oft_walls
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
nvidia-smi -L
module purge
module load mamba/24.3.0
source activate "/projects/p33100/siosio/envs/openvla-oft"      # OFT env @ p33100

export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH="$HOME/crash_bench"                            # crashbench 未装进本 env
export OPENVLA_OFT_ROOT="/projects/p33100/siosio/third_party/openvla-oft"
export CRASHBENCH_OPENVLA_ROOT="$OPENVLA_OFT_ROOT"              # env 与 policy 同一 experiments.robot
export HF_HOME="/projects/p33100/siosio/huggingface_cache"      # checkpoint 在 p33100

cd "$HOME/crash_bench"
# 可用 --export=ALL,SCEN=scenarios_control,OUT=results/oft_controls.json 覆盖
SCEN="${SCEN:-scenarios}"
OUT="${OUT:-results/oft_walls.json}"
VIDEO="${VIDEO:-results/oft_walls_videos}"
echo "SCEN=$SCEN  OUT=$OUT"
python scripts/run_pilot.py \
  --policy openvla-oft \
  --scenarios "$SCEN" \
  --checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --unnorm_key libero_spatial \
  --out "$OUT" \
  --video_dir "$VIDEO"
echo "DONE @ $(date)"
