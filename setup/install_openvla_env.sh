#!/bin/bash
# =============================================================================
# CrashBench — OpenVLA 环境一键安装 (Quest / gengpu)
#
#   官方栈: Python 3.10 / torch 2.2.0+cu121 / transformers 4.40.1 /
#           tokenizers 0.19.1 / timm 0.9.10 / flash-attn 2.5.5 + LIBERO
#
#   设计要点(踩过的坑):
#   - torch 用 pip wheel 锁死 2.2.0+cu121,好让 flash-attn 预编译 wheel 的 ABI 精确匹配
#   - flash-attn 优先装预编译 wheel(计算节点不一定有 nvcc,源码编译会失败),失败再 fallback
#   - pip 缓存 + HF checkpoint 全部落 /projects/p33100(不撑爆 home 80G 配额)
#   - 一个项目一个环境 (-p prefix);绝不 pip --user
#
#   既可在交互节点直接 `bash install_openvla_env.sh`,也被 install_openvla.sbatch 调用。
#   幂等:重复跑会跳过已建好的部分。
# =============================================================================
set -euo pipefail

# ---- 路径配置 -------------------------------------------------------------
ENV_PREFIX="$HOME/crash_bench/envs/openvla"
SRC_DIR="$HOME/crash_bench/third_party"
PY_VER=3.10
P33100="/projects/p33100/siosio"

# pip 缓存 -> p33100,省 home 空间
export PIP_CACHE_DIR="$P33100/pip_cache"
export PYTHONNOUSERSITE=1          # 防 ~/.local 串包
mkdir -p "$PIP_CACHE_DIR" "$SRC_DIR"

echo "=========================================================="
echo "[$(date)] host=$(hostname)  env=$ENV_PREFIX"
echo "=========================================================="
# 注意: gengpu 计算节点无外网,本脚本须在【登录节点】跑(纯下载+安装,无源码编译)。
# GPU 相关验证交给 verify_openvla.sbatch 在计算节点离线做。
echo "[net] 联网自检:"; timeout 20 git ls-remote https://github.com/openvla/openvla.git HEAD >/dev/null 2>&1 \
  && echo "  OK(有外网)" || { echo "  !! 当前节点无外网。请在【登录节点】运行本脚本。"; exit 1; }

# ---- 1. 干净初始化 mamba + 建环境 -----------------------------------------
module purge
module load mamba/24.3.0

if [ ! -d "$ENV_PREFIX" ]; then
  echo "[1] 创建环境 python=$PY_VER ..."
  mamba create -p "$ENV_PREFIX" python=$PY_VER -y
else
  echo "[1] 环境已存在,跳过创建"
fi
source activate "$ENV_PREFIX"
echo "[1] python = $(python --version 2>&1)  ($(which python))"
python -m pip install --upgrade pip

# ---- 2. torch 2.2.0 + cu121 (pip wheel, 版本锁死) -------------------------
if python -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('2.2.0') else 1)" 2>/dev/null; then
  echo "[2] torch 2.2.0 已装,跳过"
else
  echo "[2] 安装 torch 2.2.0 / torchvision 0.17.0 / torchaudio 2.2.0 (cu121) ..."
  python -m pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121
fi

# ---- 3. clone + 安装 openvla (pin transformers/timm/tokenizers) -----------
cd "$SRC_DIR"
[ -d openvla ] || git clone https://github.com/openvla/openvla.git
cd openvla
echo "[3] pip install -e openvla ..."
python -m pip install -e .

# ---- 4. flash-attn 2.5.5 (优先预编译 wheel) -------------------------------
# 用 pip show 判断(不 import:登录节点无 GPU,import flash_attn 会找不到 libcuda 而报错)
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

# ---- 5. LIBERO (robosuite + mujoco) ---------------------------------------
cd "$SRC_DIR"
[ -d LIBERO ] || git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
# LIBERO 顶层 libero/ 是 PEP420 命名空间包(无 __init__),新 setuptools 的 PEP660
# editable 用 find_packages 发现不了它 -> import 失败。用 compat 模式退回 legacy 行为。
echo "[5] pip install -e LIBERO (editable_mode=compat) ..."
python -m pip install -e . --config-settings editable_mode=compat
# LIBERO 首次 import 会交互问路径(非交互环境会 EOFError);喂 N 用默认路径生成 config
echo "[5] 非交互生成 ~/.libero/config.yaml ..."
printf 'N\n' | python -c "import libero.libero" || true
cd "$SRC_DIR/openvla"
echo "[5] 安装 libero_requirements.txt ..."
python -m pip install -r experiments/robot/libero/libero_requirements.txt

# ---- 5b. 修正 numpy/opencv (libero_requirements 会把 numpy 拉到 2.x,与
#          torch2.2 / tensorflow2.15 / robosuite 的 numpy<2 ABI 冲突) ----------
echo "[5b] pin numpy<2 + 兼容版 opencv ..."
python -m pip install "numpy==1.26.4" "opencv-python==4.9.0.80"

# ---- 5c. 修正 protobuf 生态 (OpenVLA 没 pin tensorflow-metadata / wandb 版本,
#          pip 抓最新 -> 要 protobuf>=5,但 tensorflow2.15 锁 protobuf<4.24,冲突) -----
#   tensorflow-metadata 1.14.0 + wandb 0.16.6 都吃 protobuf 3.20.3,与 tf2.15 一致。
echo "[5c] pin tensorflow-metadata 1.14.0 + wandb 0.16.6 (配 protobuf 3.20.3) ..."
python -m pip install "tensorflow-metadata==1.14.0" "wandb==0.16.6"

# ---- 6. 预下载 LIBERO-Spatial 微调 checkpoint (-> p33100/HF_HOME) ----------
echo "[6] 预下载 openvla-7b-finetuned-libero-spatial checkpoint (~16GB -> p33100) ..."
huggingface-cli download openvla/openvla-7b-finetuned-libero-spatial >/dev/null \
  || echo "  !! checkpoint 预下载失败,sanity 作业首跑时会自动重试下载"

# ---- 7. 验证(登录节点:只查版本,不 import GPU 库)-------------------------
echo "[7] ===== 版本验证(登录节点)====="
python - <<'PY'
import torch, transformers, timm, tokenizers
print("torch       :", torch.__version__)
print("transformers:", transformers.__version__, "| timm:", timm.__version__,
      "| tokenizers:", tokenizers.__version__)
for p in ("flash-attn",):
    import importlib.metadata as m
    try: print(f"{p:12}:", m.version(p), "(已装,GPU 验证留给 verify 作业)")
    except Exception as e: print(f"{p:12}: !! 未装", e)
PY
echo
echo "[7] pip check (依赖冲突自检,有 warning 正常):"
python -m pip check || true

echo "=========================================================="
echo "[$(date)] 安装完成。下一步在【GPU 节点】跑验证:"
echo "  cd ~/crash_bench/setup && sbatch verify_openvla.sbatch"
echo "=========================================================="
