#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
git diff --quiet
git diff --cached --quiet
COMMIT="$(git rev-parse HEAD)"
SHORT_COMMIT="${COMMIT:0:12}"
RUN_TAG="${CB_CF_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${CB_CF_SUPPLEMENT_ROOT:-results/counterfactual_router/fresh_random_${SHORT_COMMIT}_${RUN_TAG}}"
FROZEN_ROOT="${CB_CF_FROZEN_ROOT:-results/counterfactual_router/fresh_online_2eab4a4a53dc_20260817T080708Z}"

PREPARE_JOB="$(sbatch --parsable \
  --export="ALL,CB_CF_SUPPLEMENT_ROOT=$RUN_ROOT,CB_CF_FROZEN_ROOT=$FROZEN_ROOT,CB_CODE_COMMIT=$COMMIT" \
  setup/prepare_fresh_counterfactual_router_supplement.sbatch)"
FULL_JOB="$(sbatch --parsable --dependency="afterok:$PREPARE_JOB" \
  --export="ALL,CB_CF_SUPPLEMENT_ROOT=$RUN_ROOT,CB_CF_FROZEN_ROOT=$FROZEN_ROOT,CB_CODE_COMMIT=$COMMIT" \
  setup/fresh_counterfactual_router_supplement.sbatch)"

printf 'prepare_job=%s\nfull_job=%s\ncommit=%s\nrun_root=%s\nfrozen_root=%s\n' \
  "$PREPARE_JOB" "$FULL_JOB" "$COMMIT" "$RUN_ROOT" "$FROZEN_ROOT"
