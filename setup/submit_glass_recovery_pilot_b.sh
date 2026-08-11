#!/bin/bash
set -euo pipefail

STAGE="${1:-}"
case "$STAGE" in
  source_traces|diagnose|frontier|collect) ;;
  *) echo "usage: $0 source_traces|diagnose|frontier|collect" >&2; exit 2 ;;
esac

: "${CB_PILOT_B_ROOT:?set CB_PILOT_B_ROOT}"
if [[ "$STAGE" == collect ]]; then
  : "${CB_PILOT_B_H:?set CB_PILOT_B_H from frontier_summary.json}"
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
git diff --quiet
git diff --cached --quiet
COMMIT="$(git rev-parse HEAD)"
JOB_ID="$(sbatch --parsable \
  --export="ALL,CB_PILOT_B_STAGE=$STAGE,CB_CODE_COMMIT=$COMMIT" \
  setup/glass_recovery_pilot_b.sbatch)"

printf 'job_id=%s\nstage=%s\ncommit=%s\nrun_root=%s\n' \
  "$JOB_ID" "$STAGE" "$COMMIT" "$CB_PILOT_B_ROOT"
