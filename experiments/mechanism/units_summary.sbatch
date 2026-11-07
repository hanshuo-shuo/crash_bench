#!/usr/bin/env bash
# Mechanism pilot, unit study summary (CPU, numpy only): C1, R1, R2 and the tables -> ROOT/SUMMARY.json.
# Submitted by launch_units.py. With CB_UNITS_GATE=1 it is the smoke's gate: it fails unless the smoke reads tool_ok,
# which cancels the formal batch through its afterok dependency.
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=cb_mech_units_summary
set -euo pipefail
ROOT="${CB_FD_ROOT:?}"
cd "$ROOT/source"
sha256sum --quiet -c "$ROOT/source.sha256"
export CB_CODE_COMMIT="$(cat "$ROOT/SOURCE_COMMIT")"
unset OPENROUTER_API_KEY
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
if [ "${CB_UNITS_GATE:-0}" = 1 ]; then
  /projects/p33100/siosio/envs/openpi/bin/python -u "$ROOT/source/experiments/mechanism/run_units.py" summarize "$ROOT" --require-tool-ok
else
  /projects/p33100/siosio/envs/openpi/bin/python -u "$ROOT/source/experiments/mechanism/run_units.py" summarize "$ROOT"
fi
