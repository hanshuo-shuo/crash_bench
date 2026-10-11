#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --job-name=cb_verifier_prefix
set -euo pipefail
cd "$CB_VERIFIER_RUN/source"
sha256sum --quiet -c "$CB_VERIFIER_RUN/source.sha256"
unset OPENROUTER_API_KEY
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CB_CODE_COMMIT="$(cat "$CB_VERIFIER_RUN/SOURCE_COMMIT")"
export PYTHONPATH="$CB_VERIFIER_ASSETS/packages:$CB_VERIFIER_RUN/source"
/projects/p33100/siosio/envs/openpi/bin/python -m unittest discover -s verifier_training/tests -v > "$CB_VERIFIER_RUN/unit_tests.txt" 2>&1
/projects/p33100/siosio/envs/openpi/bin/python -u -m verifier_training.prefix_diagnostic "$CB_VERIFIER_RUN" "$CB_VERIFIER_CAMPAIGN" "$CB_VERIFIER_ASSETS" "$CB_DIAGNOSTIC_PARENT"
