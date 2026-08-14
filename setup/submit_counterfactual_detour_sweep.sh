#!/usr/bin/env bash
# Run on a Quest login node after syncing a clean commit.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
git diff --quiet
git diff --cached --quiet
COMMIT="$(git rev-parse HEAD)"
SHORT_COMMIT="${COMMIT:0:12}"
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT="${CB_CF_SWEEP_OUTPUT:-results/counterfactual_router/detour_sweep_${SHORT_COMMIT}_${RUN_TAG}}"
[[ ! -e "$OUTPUT" ]] || { echo "refusing existing output: $OUTPUT" >&2; exit 2; }

sbatch --parsable \
  --export="ALL,CB_CF_SWEEP_OUTPUT=$OUTPUT,CB_CODE_COMMIT=$COMMIT" \
  setup/counterfactual_detour_sweep.sbatch
echo "output=$OUTPUT"
