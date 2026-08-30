#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${CB_D8_RUN_ROOT:?set exact frozen D8 run root below results/expansion/d8_benchmark}"
[[ "$CB_D8_RUN_ROOT" =~ ^results/expansion/d8_benchmark/[A-Za-z0-9._-]+$ ]] \
  || { printf 'invalid CB_D8_RUN_ROOT\n' >&2; exit 2; }
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test -f "$CB_D8_RUN_ROOT/execution_freeze.json"
test -f "$CB_D8_RUN_ROOT/authorization/test_authorization.json"
test -f "$CB_D8_RUN_ROOT/authorization/test_open.lock"
test ! -e "$CB_D8_RUN_ROOT/authorization/test_complete.seal"
run_id="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$CB_D8_RUN_ROOT/execution_freeze.json")"
commit="$(python scripts/expansion/hash_tree_manifest.py --print-git-head .)"
frozen_commit="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["git_commit"])' "$CB_D8_RUN_ROOT/execution_freeze.json")"
test "$commit" = "$frozen_commit"
job_id="$(sbatch --parsable --array=0-31 --export="ALL,CB_D8_RUN_ROOT=$CB_D8_RUN_ROOT,CB_D8_RUN_ID=$run_id" setup/expansion_benchmark_test.sbatch)"
printf 'run_root=%s\nrun_id=%s\ncommit=%s\njob=%s\n' "$CB_D8_RUN_ROOT" "$run_id" "$commit" "$job_id"
