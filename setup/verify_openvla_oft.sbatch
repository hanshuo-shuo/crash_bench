#!/bin/bash
# GPU 节点离线验证 OFT env(在登录节点 install_openvla_oft_env.sh 装完后提交):
#   cd ~/crash_bench/setup && sbatch verify_openvla_oft.sbatch
# 验证: cuda + flash_attn/robosuite/libero import + OFT 仓库 import + checkpoint 在位。
# 注意 env 在 p33100(不在 home)。
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --job-name=openvla_oft_verify
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
nvidia-smi -L

module purge
module load mamba/24.3.0
source activate "/projects/p33100/siosio/envs/openvla-oft"   # ← env 在 p33100
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OPENVLA_OFT_ROOT="/projects/p33100/siosio/third_party/openvla-oft"

python - <<'PY'
import sys, torch
print("torch       :", torch.__version__, "| cuda?", torch.cuda.is_available(),
      "|", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU"))
assert torch.cuda.is_available(), "CUDA 不可用!"
import flash_attn; print("flash_attn  :", flash_attn.__version__)
import transformers, timm, tokenizers
print("transformers:", transformers.__version__, "| timm:", timm.__version__,
      "| tokenizers:", tokenizers.__version__)
import robosuite; print("robosuite   :", robosuite.__version__)
import libero;    print("libero      : import OK")
# OFT 仓库上 sys.path 后能 import 它的 experiments.robot（后续 wrapper 会用同样路径）
import os
sys.path.insert(0, os.environ["OPENVLA_OFT_ROOT"])
import experiments.robot.openvla_utils as ou
print("OFT repo    : experiments.robot import OK")
print("\nALL GREEN ✓  (下一步：写 OpenVLAOFTPolicy wrapper + 跑 on/off-path 协议)")
PY
echo "DONE @ $(date)"
