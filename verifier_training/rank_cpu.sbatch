#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=cb_rank64_cpu
set -euo pipefail
cd "$CB_VERIFIER_RUN/source"
sha256sum --quiet -c "$CB_VERIFIER_RUN/source.sha256"
unset OPENROUTER_API_KEY
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CB_CODE_COMMIT="$(cat "$CB_VERIFIER_RUN/SOURCE_COMMIT")"
export PYTHONPATH="$CB_VERIFIER_ASSETS/packages:$CB_VERIFIER_RUN/source"
if [[ "$CB_BINARY_STAGE" == report ]]; then
  /projects/p33100/siosio/envs/openpi/bin/python -u -m verifier_training.rank_report "$CB_VERIFIER_RUN"
else
  /projects/p33100/siosio/envs/openpi/bin/python -u -m verifier_training.binary_run preflight "$CB_VERIFIER_RUN"
fi
