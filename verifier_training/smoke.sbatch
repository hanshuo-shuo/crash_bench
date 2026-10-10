#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=cb_verifier_smoke
set -euo pipefail
cd "$CB_VERIFIER_RUN/source"
sha256sum --quiet -c "$CB_VERIFIER_RUN/source.sha256"
unset OPENROUTER_API_KEY
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CB_CODE_COMMIT="$(cat "$CB_VERIFIER_RUN/SOURCE_COMMIT")"
export PYTHONPATH="$CB_VERIFIER_ASSETS/packages:$CB_VERIFIER_RUN/source"
for VARIANT in A B; do
 /projects/p33100/siosio/envs/openpi/bin/python -u -m verifier_training.train \
  --variant "$VARIANT" --dataset "$CB_VERIFIER_DATASET" --assets "$CB_VERIFIER_ASSETS" \
  --config verifier_training/configs/smoke.json --out "$CB_VERIFIER_RUN/$VARIANT"
done
/projects/p33100/siosio/envs/openpi/bin/python -m verifier_training.report "$CB_VERIFIER_RUN"
