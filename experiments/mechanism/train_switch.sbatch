#!/usr/bin/env bash
# Post-training of pi0.5 (GPU; no API). Submitted only by launch_train.py. train_switch.py runs openpi's own training
# script from the pi05_libero checkpoint on the datasets and settings of ROOT/TRAIN.json -> ROOT/checkpoints,
# ROOT/TRAINED.json. The peak GPU memory is sampled into ROOT/GPU_MEMORY.txt (MiB).
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=sxm
#SBATCH --exclude=qgpu2014
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --job-name=cb_mech_train
set -euo pipefail
ROOT="${CB_FD_ROOT:?}"
SOURCE="$ROOT/source"
cd "$SOURCE"
sha256sum --quiet -c "$ROOT/source.sha256"
export CB_CODE_COMMIT="$(cat "$ROOT/SOURCE_COMMIT")"
export CB_ASSETS=/projects/p33100/siosio/crashbench_safelibero
export CB_UPSTREAM="$HOME/crash_bench/third_party/vlsa-aegis"
test "$(cat "$CB_UPSTREAM/.git/HEAD")" = 2457feed5968ae803926e178c8ce8243b9ecdcf9
unset OPENROUTER_API_KEY
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HOME=/projects/p33100/siosio/huggingface_cache HF_DATASETS_CACHE=/projects/p33100/siosio/huggingface_cache/datasets
export HF_LEROBOT_HOME=/projects/p33100/siosio/huggingface_cache/lerobot HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export OPENPI_DATA_HOME="$CB_ASSETS/openpi_assets"
export PYTHONPATH="$CB_UPSTREAM/openpi/src:$CB_UPSTREAM/openpi/packages/openpi-client/src"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$ROOT/GPU_train.txt"
( PEAK=0; while true; do
    USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
    if [ "${USED:-0}" -gt "$PEAK" ]; then PEAK="$USED"; echo "$PEAK" > "$ROOT/GPU_MEMORY.txt"; fi
    sleep 15
  done ) &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null || true' EXIT
/projects/p33100/siosio/envs/openpi/bin/python -u "$SOURCE/experiments/mechanism/train_switch.py" "$ROOT"
