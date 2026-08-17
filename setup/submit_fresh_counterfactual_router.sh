#!/usr/bin/env bash
# Run on Quest after the exact commit has been synced.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
git diff --quiet
git diff --cached --quiet
COMMIT="$(git rev-parse HEAD)"
SHORT_COMMIT="${COMMIT:0:12}"
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="${CB_CF_FRESH_ROOT:-results/counterfactual_router/fresh_online_${SHORT_COMMIT}_${RUN_TAG}}"
[[ ! -e "$RUN_ROOT" ]]

PREPARE_JOB="$(sbatch --parsable \
  --export="ALL,CB_CF_FRESH_ROOT=$RUN_ROOT,CB_CODE_COMMIT=$COMMIT" \
  setup/prepare_fresh_counterfactual_router.sbatch)"
SMOKE_JOB="$(sbatch --parsable --dependency="afterok:$PREPARE_JOB" \
  --export="ALL,CB_CF_FRESH_MODE=smoke,CB_CF_FRESH_ROOT=$RUN_ROOT,CB_CODE_COMMIT=$COMMIT" \
  setup/fresh_counterfactual_router.sbatch)"
FULL_JOB="$(sbatch --parsable --dependency="afterok:$SMOKE_JOB" \
  --export="ALL,CB_CF_FRESH_MODE=full,CB_CF_FRESH_ROOT=$RUN_ROOT,CB_CODE_COMMIT=$COMMIT" \
  setup/fresh_counterfactual_router.sbatch)"

echo "prepare_job=$PREPARE_JOB"
echo "smoke_job=$SMOKE_JOB"
echo "full_job=$FULL_JOB"
echo "run_root=$RUN_ROOT"
