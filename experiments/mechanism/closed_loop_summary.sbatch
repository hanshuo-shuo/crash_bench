#!/usr/bin/env bash
# Mechanism pilot, closed-loop summary (CPU): tool check, readings and tables -> ROOT/SUMMARY.json. Submitted by launch_closed.py.
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=cb_mech_closed_summary
set -euo pipefail
ROOT="${CB_FD_ROOT:?}"
cd "$ROOT/source"
sha256sum --quiet -c "$ROOT/source.sha256"
export CB_CODE_COMMIT="$(cat "$ROOT/SOURCE_COMMIT")"
unset OPENROUTER_API_KEY
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
/projects/p33100/siosio/envs/openpi/bin/python -u "$ROOT/source/experiments/mechanism/closed_loop.py" summarize "$ROOT"
