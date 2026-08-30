#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
test -z "$(git status --porcelain=v1 --untracked-files=all)"
commit="$(python scripts/expansion/hash_tree_manifest.py --print-git-head .)"
protocol="$(python - <<'PY'
import json
print(json.load(open('results/expansion/governance/protocol_v1_1_freeze.json'))['protocol_sha256'])
PY
)"
utc="$(date -u +%Y%m%dT%H%M%SZ)"
run_id="${protocol:0:12}_${commit:0:12}_${utc}"
job_id="$(sbatch --parsable --array=0-47 --export="ALL,CB_EXPANSION_RUN_ID=$run_id" setup/expansion_collect_statewise.sbatch)"
printf 'run_id=%s\ncommit=%s\njob=%s\n' "$run_id" "$commit" "$job_id"
