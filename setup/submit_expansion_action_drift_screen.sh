#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
test -z "$(git status --porcelain=v1 --untracked-files=all)"
commit="$(python scripts/expansion/hash_tree_manifest.py --print-git-head .)"
protocol="$(sha256sum configs/expansion/benchmark_v1.yaml | awk '{print $1}')"
utc="$(date -u +%Y%m%dT%H%M%SZ)"
run_id="${protocol:0:12}_${commit:0:12}_${utc}"
job_id="$(sbatch --parsable --array=0-15 --export="ALL,CB_EXPANSION_RUN_ID=$run_id" setup/expansion_action_drift_screen.sbatch)"
printf 'run_id=%s\ncommit=%s\njob=%s\n' "$run_id" "$commit" "$job_id"
