#!/bin/bash
# GPU 节点离线验证 π0 (openpi) env(登录节点 install_openpi_env.sh 装完后提交):
#   cd ~/crash_bench/setup && sbatch verify_openpi.sbatch
# 验证:jax 认 GPU + openpi/LIBERO/robosuite/mujoco3.9 import + 加载 pi0_libero + 一次 infer。
# env 是 uv venv,在 p33100(不在 home)。
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=openpi_verify
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
nvidia-smi -L

P33100="/projects/p33100/siosio"
source "$P33100/envs/openpi/bin/activate"          # ← uv venv @ p33100
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH="$HOME/crash_bench"               # crashbench 未装进本 env
export HF_HOME="$P33100/huggingface_cache"
export OPENPI_DATA_HOME="$P33100/openpi_assets"     # checkpoint 缓存
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

python - <<'PY'
import jax
print("jax         :", jax.__version__, "| devices:", jax.devices())
assert any(d.platform == "gpu" for d in jax.devices()), "JAX 没认到 GPU!"
import importlib.metadata as m
for n in ["jaxlib","flax","torch","transformers","mujoco","robosuite","numpy","openpi-client"]:
    try: print(f"{n:12s}:", m.version(n))
    except Exception as e: print(f"{n:12s}: !! {e}")
import robosuite, libero           # LIBERO 仿真栈
print("robosuite   :", robosuite.__version__, "| libero import OK")

# crashbench 侧 π0 路径 torch-free 可 import(不碰 experiments.robot)
from crashbench.policies import canonical_name
print("crashbench  : canonical_name('pi0') =", canonical_name("pi0"))

# 端到端最小验证:加载 pi0_libero + 一次 infer(用 openpi 自带的随机 libero example)
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.policies import libero_policy
cfg = _config.get_config("pi0_libero")
policy = policy_config.create_trained_policy(cfg, "gs://openpi-assets/checkpoints/pi0_libero")
ex = libero_policy.make_libero_example()
out = policy.infer(ex)
import numpy as np
act = np.asarray(out["actions"])
print("pi0 infer   : action chunk shape =", act.shape, "| dtype", act.dtype)
assert act.shape[-1] == 7, f"期望 7-DoF, 得到 {act.shape}"
print("\nALL GREEN ✓  (下一步:sbatch run_pilot_openpi.sbatch 跑 on/off-path 协议)")
PY
echo "DONE @ $(date)"
