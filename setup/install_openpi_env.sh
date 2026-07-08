#!/bin/bash
# =============================================================================
# CrashBench — π0 (openpi) 环境一键安装 (Path 3 跨策略复现,第 3 个架构)
#
#   ★ 与 OFT 脚本一样:一切落 /projects/p33100(home 配额已满),绝不碰其它 env。
#   ★ 与 OFT 的关键不同:π0 是 **JAX 栈**(jax 0.5.3 cuda12 + flax + torch2.7.1),
#     依赖被 openpi 的 uv.lock 锁得极死 —— 手撸 pip 极易崩 JAX/tensorstore/ml-dtypes。
#     所以这里用 openpi **官方支持的 uv** 按 lock 精确安装,再把 LIBERO 仿真栈
#     (robosuite 1.4.1 + mujoco 3.9.0 + bddl)叠进同一个 venv(crashbench 同进程 eval 需要)。
#
#   为什么单 env(而非 openpi 官方的 server/client 双 env):crashbench 的 eval 循环
#   是 **同进程** policy.act();π0 路径已在 crashbench 侧做到 torch-free(见
#   libero_adapter.py 的 model_family=="pi0" 分支),故 openpi(JAX)与 LIBERO(mujoco)
#   可共处一个 venv,`--policy pi0` 即复用 run_pilot.py,零改 eval。
#
#   踩坑:
#   - mujoco 3.10 改了 mj_fullM 签名 -> robosuite 1.4.1 崩;必 pin mujoco==3.9.0(同 base/OFT)。
#   - openpi 的 gym-aloha 会带更新的 mujoco,但 π0 推理(JAX)根本不碰 mujoco,pin 3.9.0 安全。
#   - numpy 两边都要 <2;pin 1.26.4。
#   - 必须在【登录节点】跑(需外网:GCS checkpoint + git 依赖 lerobot/dlimp)。
#   幂等:uv sync / 已装包会跳过。
# =============================================================================
set -euo pipefail

P33100="/projects/p33100/siosio"
ENV_PREFIX="$P33100/envs/openpi"           # ← venv(uv 托管)放这
SRC_DIR="$P33100/third_party"              # ← openpi 仓库已 clone 于此
OPENPI_DIR="$SRC_DIR/openpi"
LIBERO_DIR="$HOME/crash_bench/third_party/LIBERO"   # 复用 base 已 clone 的 LIBERO

# uv / pip / 构建 tmp / HF / checkpoint 全落 p33100(别占 home）
export UV_INSTALL_DIR="$P33100/bin"
export UV_CACHE_DIR="$P33100/uv_cache"
export UV_PYTHON_INSTALL_DIR="$P33100/uv_python"
export UV_PROJECT_ENVIRONMENT="$ENV_PREFIX"     # uv sync -> 这个 venv(而非 repo/.venv)
export TMPDIR="$P33100/tmp"
export PIP_CACHE_DIR="$P33100/pip_cache"
export HF_HOME="${HF_HOME:-$P33100/huggingface_cache}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$P33100/openpi_assets}"   # checkpoint 缓存 -> p33100
export PYTHONNOUSERSITE=1
export GIT_LFS_SKIP_SMUDGE=1
mkdir -p "$UV_INSTALL_DIR" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$TMPDIR" \
         "$PIP_CACHE_DIR" "$OPENPI_DATA_HOME" "$SRC_DIR"

echo "=========================================================="
echo "[$(date)] host=$(hostname)"
echo "  venv    = $ENV_PREFIX"
echo "  openpi  = $OPENPI_DIR"
echo "  assets  = $OPENPI_DATA_HOME  (checkpoint 落这, home 零占用)"
df -h "$P33100" | tail -1
echo "=========================================================="
echo "[net] 联网自检:"; timeout 20 git ls-remote https://github.com/Physical-Intelligence/openpi HEAD >/dev/null 2>&1 \
  && echo "  OK(有外网)" || { echo "  !! 无外网。请在【登录节点】运行本脚本。"; exit 1; }

# ---- 0. clone openpi(若无)-------------------------------------------------
if [ ! -d "$OPENPI_DIR" ]; then
  echo "[0] clone openpi (skip LFS) ..."
  git clone --depth 1 https://github.com/Physical-Intelligence/openpi.git "$OPENPI_DIR"
else
  echo "[0] openpi 仓库已在 $OPENPI_DIR,跳过 clone"
fi

# ---- 1. 安装 uv(standalone,落 p33100)-----------------------------------
if [ ! -x "$UV_INSTALL_DIR/uv" ]; then
  echo "[1] 安装 uv -> $UV_INSTALL_DIR ..."
  curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$UV_INSTALL_DIR" sh
fi
export PATH="$UV_INSTALL_DIR:$PATH"
echo "[1] uv = $(uv --version 2>&1)  ($(which uv))"

# ---- 2. uv sync:按 lock 精确装 openpi 主依赖(jax/flax/torch/lerobot...)---
#   --no-dev 跳过 dev 组;rlds 组(tensorflow-cpu)非默认组,自动跳过(训练才需要)。
cd "$OPENPI_DIR"
#   --no-install-package rerun-sdk:rerun-sdk 0.23.1 只有 glibc>=2.31 wheel,RHEL8(glibc 2.28)
#   装不了;它仅是 lerobot 的可视化传递依赖,π0 **推理**根本不 import,安全跳过。
echo "[2] uv sync --frozen --no-dev (skip rerun-sdk)  (jax cuda12 + torch2.7.1,多 GB,耐心) ..."
uv sync --frozen --no-dev --no-install-package rerun-sdk

# uv 建的 venv 是 **pip-less** 的 —— 后续一律用 `uv pip`(指向该 venv),不走 uv 依赖解析
# 解锁,避免动 openpi 的 lock。
PY="$ENV_PREFIX/bin/python"
export VIRTUAL_ENV="$ENV_PREFIX"      # 让 `uv pip` 作用于本 venv
echo "[2] python = $($PY --version 2>&1)  ($PY)"

# ---- 3. 叠 LIBERO 仿真栈进同一 venv(crashbench 同进程 eval 需要)----------
#   --no-deps 装 LIBERO 本体避免它乱升级 openpi 的锁;--config-settings editable_mode=compat:
#   LIBERO 的 `libero` 是**命名空间包**(无顶层 __init__.py),uv 默认严格 editable 的
#   find_packages() 找不到它 → 装了也 import 不到;compat 模式把 repo root 加进 .pth 才行。
[ -d "$LIBERO_DIR" ] || { cd "$SRC_DIR"; [ -d LIBERO ] || git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git; LIBERO_DIR="$SRC_DIR/LIBERO"; }
echo "[3] uv pip install -e LIBERO (--no-deps, compat) @ $LIBERO_DIR ..."
uv pip install -e "$LIBERO_DIR" --no-deps --config-settings editable_mode=compat
echo "[3] 补 LIBERO 运行依赖(robosuite/bddl/等;numpy/mujoco 稍后 pin)..."
uv pip install \
  "robosuite==1.4.1" "bddl==1.0.1" "future" "easydict" "thop" "hydra-core" \
  "cloudpickle" "gym" "matplotlib" "imageio[ffmpeg]" "opencv-python"
#   注: bddl 1.0.1 import `future` 却没在依赖里声明,须显式补(否则 libero.libero.envs 崩)。

# ---- 4. pin 冲突大坑:mujoco==3.9.0 + numpy<2 --------------------------------
#   robosuite 1.4.1 默认拉 mujoco 2.3.7;必升到 3.9.0(与 base/OFT 同,3.10 崩 mj_fullM)。
echo "[4] pin mujoco==3.9.0 + numpy==1.26.4 ..."
uv pip install "mujoco==3.9.0" "numpy==1.26.4"

# ---- 5. 预下载 π0 LIBERO checkpoint(gs://,匿名公开桶 -> p33100)-----------
#   config=pi0_libero(真 π0);checkpoint dir=gs://openpi-assets/checkpoints/pi0_libero。
#   openpi 的 download.maybe_download 会用 fsspec[gcs] 匿名拉到 OPENPI_DATA_HOME。
echo "[5] 预下载 pi0_libero checkpoint -> $OPENPI_DATA_HOME (可能几 GB) ..."
cd "$OPENPI_DIR"
"$PY" - <<'PY' || echo "  !! 预下载失败;首跑时 create_trained_policy 会重试"
import openpi.shared.download as download
p = download.maybe_download("gs://openpi-assets/checkpoints/pi0_libero")
print("  checkpoint ->", p)
PY

# ---- 6. 登录节点版本验证(不碰 GPU)----------------------------------------
echo "[6] ===== 版本验证(登录节点)====="
"$PY" - <<'PY'
import importlib.metadata as m
def v(n):
    try: return m.version(n)
    except Exception: return "MISSING"
for n in ["jax","jaxlib","flax","torch","transformers","numpy","mujoco","robosuite","openpi-client"]:
    print(f"  {n:14s}: {v(n)}")
import openpi.training.config as c
print("  openpi config get_config('pi0_libero') OK:", c.get_config("pi0_libero").name)
PY
echo "[6] uv pip check:"; uv pip check || true

echo "=========================================================="
echo "[$(date)] 安装完成。venv + 仓库 + checkpoint 全在 p33100,home 未占用。"
echo "  激活: source $ENV_PREFIX/bin/activate"
echo "  GPU 验证:  cd ~/crash_bench/setup && sbatch verify_openpi.sbatch"
echo "  跑 π0:     cd ~/crash_bench/setup && sbatch run_pilot_openpi.sbatch"
echo "=========================================================="
