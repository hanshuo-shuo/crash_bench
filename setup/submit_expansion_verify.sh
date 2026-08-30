#!/usr/bin/env bash
# Submit one D1 backend through the maintained expansion verification entry.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

policy="${1:-}"
case "$policy" in
  openvla|pi0) ;;
  *) printf 'usage: %s {openvla|pi0}\n' "$0" >&2; exit 2 ;;
esac

test -z "$(git status --porcelain=v1 --untracked-files=all)"
commit="$(python scripts/expansion/hash_tree_manifest.py --print-git-head .)"
job_id="$(sbatch --parsable --export="ALL,CB_POLICY=$policy" setup/expansion_verify.sbatch)"
printf 'policy=%s\ncommit=%s\njob=%s\n' "$policy" "$commit" "$job_id"
