#!/usr/bin/env bash
# Submit exactly one P3.2 calibration + fresh cohort + frozen closeout chain.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
git diff --quiet
git diff --cached --quiet
COMMIT="$(git rev-parse HEAD)"
SHORT_COMMIT="${COMMIT:0:12}"
RUN_TAG="${CB_P3_2_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${CB_P3_2_RUN_ROOT:-results/counterfactual_router/p3_2_closeout_${SHORT_COMMIT}_${RUN_TAG}}"
[[ ! -e "$RUN_ROOT" ]]

CALIBRATION_JOB="$(sbatch --parsable \
  --export="ALL,CB_P3_2_RUN_ROOT=$RUN_ROOT,CB_CODE_COMMIT=$COMMIT" \
  setup/p3_2_direct_sequential_calibration.sbatch)"
PREPARE_JOB="$(sbatch --parsable \
  --export="ALL,CB_P3_2_RUN_ROOT=$RUN_ROOT,CB_CODE_COMMIT=$COMMIT" \
  setup/p3_2_prepare_fresh_cohort.sbatch)"
CLOSEOUT_JOB="$(sbatch --parsable \
  --dependency="afterok:$CALIBRATION_JOB:$PREPARE_JOB" \
  --export="ALL,CB_P3_2_RUN_ROOT=$RUN_ROOT,CB_CODE_COMMIT=$COMMIT" \
  setup/p3_2_frozen_dynamic_closeout.sbatch)"

printf 'calibration_job=%s\nprepare_job=%s\ncloseout_job=%s\ncommit=%s\nrun_root=%s\n' \
  "$CALIBRATION_JOB" "$PREPARE_JOB" "$CLOSEOUT_JOB" "$COMMIT" "$RUN_ROOT"
