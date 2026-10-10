#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=cb_verifier_report
set -euo pipefail
cd "$CB_VERIFIER_RUN/source"
sha256sum --quiet -c "$CB_VERIFIER_RUN/source.sha256"
unset OPENROUTER_API_KEY
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CB_CODE_COMMIT="$(cat "$CB_VERIFIER_RUN/SOURCE_COMMIT")"
/projects/p33100/siosio/envs/openpi/bin/python -m verifier_training.report "$CB_VERIFIER_SMOKE" --out "$CB_VERIFIER_RUN"
