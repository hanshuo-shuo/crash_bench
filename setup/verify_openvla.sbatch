#!/bin/bash
# GPU 节点离线验证(环境已在登录节点装好后再提交):
#   cd ~/crash_bench/setup && sbatch verify_openvla.sbatch
# 验证: cuda 可用 + flash_attn / robosuite / libero 能 import + checkpoint 在位
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --job-name=openvla_verify
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

python - <<'PY'
import torch
print("torch       :", torch.__version__, "| cuda?", torch.cuda.is_available(),
      "|", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU"))
assert torch.cuda.is_available(), "CUDA 不可用!"
import flash_attn; print("flash_attn  :", flash_attn.__version__)
import transformers, timm, tokenizers
print("transformers:", transformers.__version__, "| timm:", timm.__version__,
      "| tokenizers:", tokenizers.__version__)
import robosuite; print("robosuite   :", robosuite.__version__)
import libero;    print("libero      : import OK")
# 离屏渲染冒烟测试
import numpy as np, mujoco
print("mujoco      :", mujoco.__version__, "| MUJOCO_GL ok")
print("\nALL GREEN ✓")
PY
echo "DONE @ $(date)"
