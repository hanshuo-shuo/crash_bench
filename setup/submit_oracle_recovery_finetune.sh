#!/bin/bash
# Submit a provenance-linked train -> held-out evaluation pipeline.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
COMMIT="$(git rev-parse HEAD)"
MODEL_DIR="${CB_RECOVERY_MODEL_DIR:-/projects/p33100/siosio/openvla_checkpoints/oracle_stop_recovery_v1}"

train_job="$(sbatch --parsable \
  --export=ALL,CB_CODE_COMMIT="$COMMIT",CB_RECOVERY_MODEL_DIR="$MODEL_DIR" \
  setup/run_oracle_recovery_finetune.sbatch)"
eval_job="$(sbatch --parsable --dependency="afterok:$train_job" \
  --export=ALL,CB_CODE_COMMIT="$COMMIT",CB_RECOVERY_MODEL_DIR="$MODEL_DIR" \
  setup/eval_oracle_recovery_finetune.sbatch)"

printf 'train_job=%s\neval_job=%s\ncommit=%s\ncheckpoint=%s/merged\n' \
  "$train_job" "$eval_job" "$COMMIT" "$MODEL_DIR"
