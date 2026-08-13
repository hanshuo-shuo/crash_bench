#!/usr/bin/env bash
# Run on a Quest login node: setup/submit_glass_detector_d0.sh smoke|full

set -euo pipefail

MODE="${1:-}"
case "$MODE" in
  smoke|full) ;;
  *) echo "usage: $0 smoke|full" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
git diff --quiet
git diff --cached --quiet
COMMIT="$(git rev-parse HEAD)"
SHORT_COMMIT="${COMMIT:0:12}"

if [[ "$MODE" == "smoke" ]]; then
  RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
  OUTPUT="${CB_D0_OUTPUT:-results/glass_recovery_v2/d0_capture_smoke_${SHORT_COMMIT}_${RUN_TAG}}"
else
  OUTPUT="${CB_D0_OUTPUT:-results/glass_recovery_v2/d0_capture_full_20260813_r2}"
fi
[[ ! -e "$OUTPUT" ]] || { echo "refusing existing output: $OUTPUT" >&2; exit 2; }

sbatch --parsable \
  --export="ALL,CB_D0_MODE=$MODE,CB_D0_OUTPUT=$OUTPUT,CB_CODE_COMMIT=$COMMIT" \
  setup/glass_detector_d0_capture.sbatch
echo "output=$OUTPUT"
