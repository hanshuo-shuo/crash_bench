#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:2
#SBATCH --exclude=qgpu2014
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=cb_v5s3_32b
# Launched only by verifier_training.launch_v5 with CB_V5_ROOT and CB_V5_ARGS; qwen-vlm env (transformers 5.x), no 3B overlay.
set -euo pipefail
cd "$CB_V5_ROOT/source"
sha256sum --quiet -c "$CB_V5_ROOT/source.sha256"
unset OPENROUTER_API_KEY
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CB_CODE_COMMIT="$(cat "$CB_V5_ROOT/SOURCE_COMMIT")"
export PYTHONPATH="$CB_V5_ROOT/source"
# The system libstdc++ is older than the one Pillow in qwen-vlm needs.
export LD_LIBRARY_PATH="/projects/p33100/siosio/envs/qwen-vlm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# CB_V5_ARGS is a module name followed by plain path/token arguments (no spaces).
/projects/p33100/siosio/envs/qwen-vlm/bin/python -u -m $CB_V5_ARGS
