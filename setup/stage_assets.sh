#!/usr/bin/env bash
set -euo pipefail
cd "${HOME}/crash_bench"
export CB_ASSETS=/projects/p33100/siosio/crashbench_safelibero
export PYTHONNOUSERSITE=1
export HF_HOME="$CB_ASSETS/huggingface"
export OPENPI_DATA_HOME="$CB_ASSETS/openpi_assets"
export UV_CACHE_DIR="$CB_ASSETS/uv_cache"
export PIP_CACHE_DIR="$CB_ASSETS/pip_cache"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$CB_ASSETS/logs"
exec /projects/p33100/siosio/envs/openpi/bin/python -u scripts/stage_assets.py
