#!/bin/bash
# Submit one provenance-linked E15/v2 train or accepted-only evaluation stage.
set -euo pipefail

usage() {
  echo "usage: $0 train|evaluate" >&2
  exit 2
}

require_env() {
  local name="$1"
  [[ -n "${!name:-}" ]] || { echo "required environment variable is empty: $name" >&2; exit 2; }
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || { echo "required input file does not exist: $path" >&2; exit 2; }
}

STAGE="${1:-}"
[[ "$STAGE" == "train" || "$STAGE" == "evaluate" ]] || usage

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
git diff --quiet && git diff --cached --quiet \
  || { echo "tracked source differs from HEAD" >&2; exit 1; }
COMMIT="$(git rev-parse HEAD)"

require_env CB_GLASS_RECOVERY_RUN_ROOT
require_env CB_GLASS_RECOVERY_PRIMARY_PROTOCOL_SHA256
require_env CB_BASE_CHECKPOINT_REVISION
require_env CB_BASE_UNNORM_KEY
require_env CB_GLASS_RECOVERY_H

if [[ "$STAGE" == "train" ]]; then
  require_env CB_GLASS_RECOVERY_TRAIN_MANIFEST
  require_env CB_GLASS_RECOVERY_VALIDATION_MANIFEST
  require_file "$CB_GLASS_RECOVERY_TRAIN_MANIFEST"
  require_file "$CB_GLASS_RECOVERY_VALIDATION_MANIFEST"
else
  require_env CB_GLASS_RECOVERY_PLACEMENT_MANIFEST
  require_env CB_GLASS_RECOVERY_TRAJECTORY_MANIFEST
  require_env CB_GLASS_RECOVERY_EVALUATION_COHORT
  require_env CB_GLASS_RECOVERY_PROTOCOL
  require_env CB_GLASS_RECOVERY_EVALUATION_PROTOCOL_SHA256
  require_env CB_GLASS_RECOVERY_CHECKPOINT
  require_env CB_BASE_CHECKPOINT
  require_file "$CB_GLASS_RECOVERY_PLACEMENT_MANIFEST"
  require_file "$CB_GLASS_RECOVERY_TRAJECTORY_MANIFEST"
  require_file "$CB_GLASS_RECOVERY_EVALUATION_COHORT"
  require_file "$CB_GLASS_RECOVERY_PROTOCOL"
  require_file "$CB_GLASS_RECOVERY_CHECKPOINT"
fi

job_id="$(sbatch --parsable \
  --export="ALL,CB_GLASS_RECOVERY_STAGE=$STAGE,CB_CODE_COMMIT=$COMMIT" \
  setup/glass_recovery_smoke.sbatch)"

printf 'job_id=%s\nstage=%s\ncommit=%s\nrun_root=%s\n' \
  "$job_id" "$STAGE" "$COMMIT" "$CB_GLASS_RECOVERY_RUN_ROOT"
