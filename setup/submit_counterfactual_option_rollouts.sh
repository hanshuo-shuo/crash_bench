#!/usr/bin/env bash
# Run on a Quest login node: setup/submit_counterfactual_option_rollouts.sh smoke|full

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
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT="${CB_CF_OUTPUT:-results/counterfactual_router/${MODE}_${SHORT_COMMIT}_${RUN_TAG}}"
[[ ! -e "$OUTPUT" ]] || { echo "refusing existing output: $OUTPUT" >&2; exit 2; }

if [[ "$MODE" == "full" ]]; then
  : "${CB_CF_DETOUR_CONFIG:?full requires the frozen detour config from the development sweep}"
  : "${CB_CF_SMOKE_DIR:?full requires the protocol-correct multi-H smoke directory}"
  [[ -f "$CB_CF_DETOUR_CONFIG" ]] || { echo "missing $CB_CF_DETOUR_CONFIG" >&2; exit 2; }
  [[ -d "$CB_CF_SMOKE_DIR" ]] || { echo "missing $CB_CF_SMOKE_DIR" >&2; exit 2; }
  python scripts/audit_counterfactual_smoke.py \
    --smoke-dir "$CB_CF_SMOKE_DIR" \
    --detour-config "$CB_CF_DETOUR_CONFIG" \
    --expected-commit "$COMMIT"
fi

sbatch --parsable \
  --export="ALL,CB_CF_MODE=$MODE,CB_CF_OUTPUT=$OUTPUT,CB_CODE_COMMIT=$COMMIT" \
  setup/counterfactual_option_rollouts.sbatch
echo "output=$OUTPUT"
