#!/bin/bash
# Verify, commit, and push exactly the P0 implementation files from this change.
# This refuses unrelated worktree changes so a convenience command cannot sweep user work
# into the commit. Usage: bash setup/commit_p0_core.sh [commit-message]

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

allowed=(
  .gitignore
  configs/p0_core.example.json
  crashbench/corridor.py
  crashbench/envs/libero_adapter.py
  crashbench/p0.py
  crashbench/policies/openvla_policy.py
  crashbench/provenance.py
  docs/P0_EXPERIMENT.md
  scripts/p0_capture.py
  scripts/p0_guard.py
  scripts/p0_probe_analysis.py
  setup/commit_p0_core.sh
  setup/p0_analyze.sbatch
  setup/p0_capture.sbatch
  setup/p0_guard.sbatch
  setup/submit_p0.sh
  tests/test_core.py
)

unexpected=0
while IFS= read -r line; do
  path="${line:3}"
  is_allowed=0
  for candidate in "${allowed[@]}"; do
    if [[ "$path" == "$candidate" ]]; then
      is_allowed=1
      break
    fi
  done
  if [[ "$is_allowed" -eq 0 ]]; then
    echo "refusing to commit unrelated path: $path" >&2
    unexpected=1
  fi
done < <(git status --porcelain=v1 --untracked-files=all)
[[ "$unexpected" -eq 0 ]] || exit 1

git diff --check
python -m pytest tests -q
python scripts/audit_repo.py
git add -- "${allowed[@]}"
git diff --cached --stat
git commit -m "${1:-Add provenance-complete P0 experiment pipeline}"
git push -u origin HEAD
