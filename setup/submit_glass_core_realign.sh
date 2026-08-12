#!/bin/bash
# Submit the provenance-linked Pilot A realignment diagnostic only.
set -euo pipefail

if [[ "${CB_ENABLE_LEGACY_GLASS_RECOVERY:-0}" != "1" ]]; then
  echo "legacy Pilot A realignment is disabled: the frozen audit is complete" >&2
  echo "set CB_ENABLE_LEGACY_GLASS_RECOVERY=1 only for an explicit provenance diagnostic" >&2
  exit 2
fi

require_env() {
  local name="$1"
  [[ -n "${!name:-}" ]] || { echo "required environment variable is empty: $name" >&2; exit 2; }
}

for name in \
  CB_PILOT_A_SOURCE_ROOT \
  CB_PILOT_A_AUDIT \
  CB_PILOT_A_INVENTORY_SUMMARY \
  CB_PILOT_A_PLACEMENTS \
  CB_PILOT_A_OUTPUT_ROOT \
  CB_PILOT_A_SUMMARY_OUT \
  CB_PILOT_A_H
do
  require_env "$name"
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
git diff --quiet && git diff --cached --quiet \
  || { echo "tracked source differs from HEAD" >&2; exit 1; }
COMMIT="$(git rev-parse HEAD)"

job_id="$(sbatch --parsable \
  --export="ALL,CB_CODE_COMMIT=$COMMIT" \
  setup/glass_core_realign.sbatch)"

printf 'job_id=%s\nstage=pilot_a_realign\ncommit=%s\noutput_root=%s\nsummary=%s\n' \
  "$job_id" "$COMMIT" "$CB_PILOT_A_OUTPUT_ROOT" "$CB_PILOT_A_SUMMARY_OUT"
