#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=cb_rank64_review
set -euo pipefail
cd "$CB_VERIFIER_RUN/source"
sha256sum --quiet -c "$CB_VERIFIER_RUN/source.sha256"
unset OPENROUTER_API_KEY
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CB_CODE_COMMIT="$(cat "$CB_VERIFIER_RUN/SOURCE_COMMIT")"
export PYTHONPATH="$CB_VERIFIER_ASSETS/packages:$CB_VERIFIER_RUN/source"
/projects/p33100/siosio/envs/openpi/bin/python -m unittest verifier_training.tests.test_rank_contract verifier_training.tests.test_rank_smoke_report -v > "$CB_VERIFIER_RUN/report_tests.txt" 2>&1
/projects/p33100/siosio/envs/openpi/bin/python -u -m verifier_training.rank_smoke_report "$CB_RANK_PARENT" "$CB_VERIFIER_RUN"
