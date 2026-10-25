#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=qgpu2014
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=cb_v5s3_gpu
# Launched only by verifier_training.launch_v5 with CB_V5_ROOT and CB_V5_ARGS.
set -euo pipefail
cd "$CB_V5_ROOT/source"
sha256sum --quiet -c "$CB_V5_ROOT/source.sha256"
unset OPENROUTER_API_KEY
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CB_CODE_COMMIT="$(cat "$CB_V5_ROOT/SOURCE_COMMIT")"
export PYTHONPATH="$CB_VERIFIER_ASSETS/packages:$CB_V5_ROOT/source"
# CB_V5_ARGS is a module name followed by plain path/token arguments (no spaces).
/projects/p33100/siosio/envs/openpi/bin/python -u -m $CB_V5_ARGS
