#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=cb_v4_cpu
# Launched only by verifier_training.launch_fd1000 with CB_V4_ROOT and CB_V4_ARGS.
set -euo pipefail
cd "$CB_V4_ROOT/source"
sha256sum --quiet -c "$CB_V4_ROOT/source.sha256"
unset OPENROUTER_API_KEY
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CB_CODE_COMMIT="$(cat "$CB_V4_ROOT/SOURCE_COMMIT")"
export PYTHONPATH="$CB_VERIFIER_ASSETS/packages:$CB_V4_ROOT/source"
if [[ "${CB_V4_STATISTICS:-0}" == 1 ]]; then
  /projects/p33100/siosio/envs/openpi/bin/python -m verifier_dataset.statistics_assets prepare "$CB_V4_ROOT"
  export PYTHONPATH="$CB_V4_ROOT/statistics_packages:$PYTHONPATH"
fi
# CB_V4_ARGS is a module name followed by plain path/token arguments (no spaces).
/projects/p33100/siosio/envs/openpi/bin/python -u -m $CB_V4_ARGS
