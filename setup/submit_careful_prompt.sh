#!/bin/bash
# Submit the wall and glass GPU matrices plus a CPU analysis job that runs only
# after both evaluations complete successfully.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "refusing to submit from a dirty checkout" >&2
  git status --short >&2
  exit 1
fi

COMMIT="$(git rev-parse HEAD)"
WALL_JOB="$(sbatch --parsable \
  --export=ALL,CB_CODE_COMMIT="$COMMIT" \
  setup/run_careful_prompt_wall.sbatch)"
GLASS_JOB="$(sbatch --parsable \
  --export=ALL,CB_CODE_COMMIT="$COMMIT" \
  setup/run_careful_prompt_glass.sbatch)"
ANALYSIS_JOB="$(sbatch --parsable \
  --dependency="afterok:${WALL_JOB}:${GLASS_JOB}" \
  --export=ALL,CB_CODE_COMMIT="$COMMIT" \
  setup/analyze_careful_prompt.sbatch)"

echo "commit=$COMMIT"
echo "wall_job=$WALL_JOB"
echo "glass_job=$GLASS_JOB"
echo "analysis_job=$ANALYSIS_JOB dependency=afterok:${WALL_JOB}:${GLASS_JOB}"
echo "wall_result=results/careful_prompt/wall_prompt_matrix.json"
echo "glass_result=results/careful_prompt/glass_prompt_matrix.json"
echo "combined_result=results/careful_prompt/combined_summary.json"
echo "analysis_report=results/ANALYSIS_careful_prompt.md"
