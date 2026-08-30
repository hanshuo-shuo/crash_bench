#!/usr/bin/env bash
# Submit one D1 backend through the maintained expansion verification entry.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

policy="${1:-}"
task_id="${2:-0}"
repeats="${3:-3}"
case "$policy" in
  openvla|pi0) ;;
  *) printf 'usage: %s {openvla|pi0}\n' "$0" >&2; exit 2 ;;
esac
case "$task_id" in
  0|2) ;;
  *) printf 'usage: %s {openvla|pi0} {0|2}\n' "$0" >&2; exit 2 ;;
esac
case "$repeats" in
  3|5) ;;
  *) printf 'usage: %s {openvla|pi0} {0|2} {3|5}\n' "$0" >&2; exit 2 ;;
esac

test -z "$(git status --porcelain=v1 --untracked-files=all)"
commit="$(python scripts/expansion/hash_tree_manifest.py --print-git-head .)"
job_id="$(sbatch --parsable --export="ALL,CB_POLICY=$policy,CB_TASK_ID=$task_id,CB_REPEATS=$repeats" setup/expansion_verify.sbatch)"
printf 'policy=%s\ntask_id=%s\nrepeats=%s\ncommit=%s\njob=%s\n' "$policy" "$task_id" "$repeats" "$commit" "$job_id"
