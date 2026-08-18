#!/usr/bin/env bash
# Submit one immutable P2 run. Defaults to a one-source execution smoke.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
git diff --quiet
git diff --cached --quiet
COMMIT="$(git rev-parse HEAD)"
SHORT_COMMIT="${COMMIT:0:12}"
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
MODE="${CB_P2_MODE:-smoke}"
OUTPUT="${CB_P2_OUTPUT:-results/counterfactual_router/p2_dynamic_${MODE}_${SHORT_COMMIT}_${RUN_TAG}}"
[[ ! -e "$OUTPUT" ]]

case "$MODE" in
  smoke)
    PLACEMENT_IDS="${CB_P2_PLACEMENT_IDS:-glass_recovery_heldout_0008}"
    TARGET_VALID="${CB_P2_TARGET_VALID:-1}"
    ;;
  full)
    PLACEMENT_IDS="${CB_P2_PLACEMENT_IDS:-}"
    TARGET_VALID="${CB_P2_TARGET_VALID:-}"
    ;;
  *)
    echo "CB_P2_MODE must be smoke or full" >&2
    exit 2
    ;;
esac

JOB="$(sbatch --parsable \
  --export="ALL,CB_P2_OUTPUT=$OUTPUT,CB_P2_PLACEMENT_IDS=$PLACEMENT_IDS,CB_P2_TARGET_VALID=$TARGET_VALID,CB_CODE_COMMIT=$COMMIT" \
  setup/dynamic_first_crossing_router.sbatch)"

echo "job=$JOB"
echo "mode=$MODE"
echo "output=$OUTPUT"
