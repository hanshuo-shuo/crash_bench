#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
git diff --quiet
git diff --cached --quiet
COMMIT="$(git rev-parse HEAD)"
SHORT_COMMIT="${COMMIT:0:12}"
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT="${CB_P2_CAL_OUTPUT:-results/counterfactual_router/p2_sequential_calibration_${SHORT_COMMIT}_${RUN_TAG}}"
[[ ! -e "$OUTPUT" ]]

JOB="$(sbatch --parsable \
  --export="ALL,CB_P2_CAL_OUTPUT=$OUTPUT,CB_CODE_COMMIT=$COMMIT" \
  setup/dynamic_first_crossing_calibration.sbatch)"

echo "job=$JOB"
echo "output=$OUTPUT"
