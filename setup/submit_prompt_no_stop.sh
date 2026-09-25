#!/bin/bash
# Run two fixed E13 prompt ablations on Quest after quest_sync.sh push.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "refusing to submit from a dirty checkout" >&2
  git status --short >&2
  exit 1
fi
commit="$(git rev-parse HEAD)"
wall_job="$(sbatch --parsable --export=ALL,CB_CODE_COMMIT="$commit" setup/run_prompt_no_stop_wall.sbatch)"
glass_job="$(sbatch --parsable --export=ALL,CB_CODE_COMMIT="$commit" setup/run_prompt_no_stop_glass.sbatch)"
printf 'commit=%s\nwall_job=%s\nglass_job=%s\n' "$commit" "$wall_job" "$glass_job"
printf 'wall_result=results/prompt_no_stop/%s/wall_%s.json\n' "$commit" "$wall_job"
printf 'glass_result=results/prompt_no_stop/%s/glass_%s.json\n' "$commit" "$glass_job"
