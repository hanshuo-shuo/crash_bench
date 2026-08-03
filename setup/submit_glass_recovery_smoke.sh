#!/bin/bash
# Submit one provenance-linked end-to-end glass recovery smoke job.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
COMMIT="$(git rev-parse HEAD)"
EXPORTS="ALL,CB_CODE_COMMIT=$COMMIT"
if [[ -n "${CB_GLASS_RECOVERY_RUN_ROOT:-}" ]]; then
  EXPORTS="$EXPORTS,CB_GLASS_RECOVERY_RUN_ROOT=$CB_GLASS_RECOVERY_RUN_ROOT"
fi

job_id="$(sbatch --parsable \
  --export="$EXPORTS" \
  setup/glass_recovery_smoke.sbatch)"
RUN_ROOT="${CB_GLASS_RECOVERY_RUN_ROOT:-results/glass_recovery_v1/smoke_${job_id}}"

printf 'job_id=%s\ncommit=%s\nrun_root=%s\n' "$job_id" "$COMMIT" "$RUN_ROOT"
