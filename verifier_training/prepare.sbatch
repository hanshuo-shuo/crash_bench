#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=cb_verifier_prepare
set -euo pipefail
cd "$CB_VERIFIER_RUN/source"
sha256sum --quiet -c "$CB_VERIFIER_RUN/source.sha256"
unset OPENROUTER_API_KEY
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CB_CODE_COMMIT="$(cat "$CB_VERIFIER_RUN/SOURCE_COMMIT")"
/projects/p33100/siosio/envs/openpi/bin/python -u -m verifier_training.prepare "$CB_VERIFIER_RUN" "$CB_VERIFIER_ASSETS"
