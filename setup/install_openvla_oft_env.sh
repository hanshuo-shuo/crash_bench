#!/bin/bash
# =============================================================================
# CrashBench — OpenVLA-OFT 环境一键安装 (Path 3 跨策略复现)
#
#   ★ 与 install_openvla_env.sh 的唯一区别 = 全部落在 /projects/p33100 分区,
#     home 一个字节都不占(home 配额已满)。env(~10-15G)+ 仓库 + pip 构建 tmp
#     统统指到 p33100(还剩 ~380G)。绝不碰 envs/openvla —— 独立 env,零污染。
#
#   官方栈(与 base openvla 同代基座,OFT 是其 fork):
#           Python 3.10 / torch 2.2.0+cu121 / flash-attn 2.5.5 + LIBERO
#           + OFT 自带 requirements(pip install -e . 处理)
#
#   踩坑沿用 base 脚本的做法:
#   - torch pip wheel 锁 2.2.0+cu121,让 flash-attn 预编译 wheel 的 ABI 精确匹配
#   - flash-attn 优先预编译 wheel(计算节点无 nvcc,源码编译会失败)
#   - 一个项目一个环境 (-p prefix);绝不 pip --user
#   - 必须在【登录节点】跑(需外网下载);GPU 验证交给 verify_openvla_oft.sbatch
#   幂等:重复跑跳过已建部分。
# =============================================================================
set -euo pipefail

# ---- 路径配置:全部 -> p33100(不占 home)---------------------------------
P33100="/projects/p33100/siosio"
ENV_PREFIX="$P33100/envs/openvla-oft"      # ← 大头 env 放这
SRC_DIR="$P33100/third_party"              # ← OFT 仓库也放这(不占 home)
PY_VER=3.10
OFT_REPO="https://github.com/moojink/openvla-oft.git"

# pip 缓存 + 构建 tmp + HF checkpoint 全落 p33100(别占 home;flash-attn 编译很吃 tmp)
export PIP_CACHE_DIR="$P33100/pip_cache"
export TMPDIR="$P33100/tmp"
export HF_HOME="${HF_HOME:-$P33100/huggingface_cache}"   # checkpoint 下到 p33100,不占 home
export PYTHONNOUSERSITE=1                    # 防 ~/.local 串包
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$SRC_DIR" "$P33100/envs"

echo "=========================================================="
echo "[$(date)] host=$(hostname)"
echo "  env  = $ENV_PREFIX"
echo "  src  = $SRC_DIR/openvla-oft"
echo "  tmp  = $TMPDIR   (home 零占用)"
df -h "$P33100" | tail -1
echo "=========================================================="
# gengpu 计算节点无外网;本脚本须在【登录节点】跑(纯下载+安装,无源码编译)。
echo "[net] 联网自检:"; timeout 20 git ls-remote "$OFT_REPO" HEAD >/dev/null 2>&1 \
  && echo "  OK(有外网)" || { echo "  !! 当前节点无外网。请在【登录节点】运行本脚本。"; exit 1; }

# ---- 1. mamba + 建环境(前缀在 p33100)------------------------------------
module purge
module load mamba/24.3.0

if [ ! -d "$ENV_PREFIX" ]; then
  echo "[1] 创建环境 python=$PY_VER @ $ENV_PREFIX ..."
  mamba create -p "$ENV_PREFIX" python=$PY_VER -y
else
  echo "[1] 环境已存在,跳过创建"
fi
source activate "$ENV_PREFIX"
echo "[1] python = $(python --version 2>&1)  ($(which python))"
python -m pip install --upgrade pip

# ---- 2. torch 2.2.0 + cu121 (版本锁死) ------------------------------------
if python -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('2.2.0') else 1)" 2>/dev/null; then
  echo "[2] torch 2.2.0 已装,跳过"
else
  echo "[2] 安装 torch 2.2.0 / torchvision 0.17.0 / torchaudio 2.2.0 (cu121) ..."
  python -m pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121
fi

# ---- 3. clone + 安装 openvla-oft ------------------------------------------
cd "$SRC_DIR"
[ -d openvla-oft ] || git clone "$OFT_REPO"
cd openvla-oft
echo "[3] pip install -e openvla-oft ..."
python -m pip install -e .

# ---- 4. flash-attn 2.5.5 (优先预编译 wheel) -------------------------------
python -m pip install packaging ninja
if python -m pip show flash-attn >/dev/null 2>&1; then
  echo "[4] flash-attn 已装,跳过"
else
  echo "[4] 安装 flash-attn 2.5.5 ..."
  FA_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
  python -m pip install "$FA_WHL" \
    || { echo "  预编译 wheel 失败,fallback 源码编译(需 nvcc) ..."; \
         python -m pip install "flash-attn==2.5.5" --no-build-isolation; }
fi

# ---- 5. LIBERO(复用已 clone 的仓库,只在本 env 里 pip install -e)----------
#   third_party/LIBERO 已存在(base 装过);OFT env 里也要 editable 装一遍。
LIBERO_DIR="$HOME/crash_bench/third_party/LIBERO"
[ -d "$LIBERO_DIR" ] || { cd "$SRC_DIR"; [ -d LIBERO ] || git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git; LIBERO_DIR="$SRC_DIR/LIBERO"; }
cd "$LIBERO_DIR"
echo "[5] pip install -e LIBERO (editable_mode=compat) @ $LIBERO_DIR ..."
python -m pip install -e . --config-settings editable_mode=compat
printf 'N\n' | python -c "import libero.libero" || true
# OFT 仓库同样带 libero_requirements.txt(路径同 base openvla)
cd "$SRC_DIR/openvla-oft"
if [ -f experiments/robot/libero/libero_requirements.txt ]; then
  echo "[5] 安装 OFT 的 libero_requirements.txt ..."
  python -m pip install -r experiments/robot/libero/libero_requirements.txt
fi

# ---- 5b. numpy/opencv/protobuf/mujoco 生态修正(与 base 同坑)---------------
echo "[5b] pin numpy<2 + 兼容版 opencv ..."
python -m pip install "numpy==1.26.4" "opencv-python==4.9.0.80"
# mujoco 3.10 改了 mj_fullM 签名,robosuite 1.4.1 用旧签名会 TypeError。锁 3.9.0(同 base)。
echo "[5b] pin mujoco==3.9.0(robosuite 1.4.1 兼容;3.10 会崩 mj_fullM)..."
python -m pip install "mujoco==3.9.0"
echo "[5c] pin tensorflow-metadata 1.14.0 + wandb 0.16.6 (配 protobuf 3.20.3) ..."
python -m pip install "tensorflow-metadata==1.14.0" "wandb==0.16.6"

# ---- 6. 预下载 OFT LIBERO-Spatial checkpoint (-> p33100/HF_HOME) -----------
#   HF_HOME 已指向 p33100(env 里 export 过);下载不占 home。
#   ⚠ checkpoint 名以 OFT README 为准 —— 若下面这个 404,去 https://github.com/moojink/openvla-oft
#     的 model zoo 查准确 repo id 再改。
OFT_CKPT="moojink/openvla-7b-oft-finetuned-libero-spatial"
echo "[6] 预下载 $OFT_CKPT (-> p33100) ..."
huggingface-cli download "$OFT_CKPT" >/dev/null \
  || echo "  !! 预下载失败(可能 repo id 需按 OFT model zoo 校正);首跑时会重试"

# ---- 7. 验证(登录节点:只查版本)-----------------------------------------
echo "[7] ===== 版本验证(登录节点)====="
python - <<'PY'
import torch, transformers, timm, tokenizers
print("torch       :", torch.__version__)
print("transformers:", transformers.__version__, "| timm:", timm.__version__,
      "| tokenizers:", tokenizers.__version__)
import importlib.metadata as m
try: print("flash-attn  :", m.version("flash-attn"), "(GPU 验证留给 verify 作业)")
except Exception as e: print("flash-attn  : !! 未装", e)
PY
echo "[7] pip check:"; python -m pip check || true

echo "=========================================================="
echo "[$(date)] 安装完成。env 与仓库均在 p33100,home 未占用。"
echo "  下一步(GPU 节点)验证:  cd ~/crash_bench/setup && sbatch verify_openvla_oft.sbatch"
echo "=========================================================="
