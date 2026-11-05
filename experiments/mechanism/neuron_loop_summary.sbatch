#!/usr/bin/env bash
# P4 closed-loop summary (CPU): tool check, N3-N5 and tables -> ROOT/SUMMARY.json. Submitted by launch_neuron_loop.py. With
# CB_NRNLOOP_GATE=1 it is the smoke's gate: it fails unless the smoke reads tool_ok, which cancels the formal batch.
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=cb_mech_nrnloop_summary
set -euo pipefail
ROOT="${CB_FD_ROOT:?}"
cd "$ROOT/source"
sha256sum --quiet -c "$ROOT/source.sha256"
export CB_CODE_COMMIT="$(cat "$ROOT/SOURCE_COMMIT")"
unset OPENROUTER_API_KEY
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
if [ "${CB_NRNLOOP_GATE:-0}" = 1 ]; then
  /projects/p33100/siosio/envs/openpi/bin/python -u "$ROOT/source/experiments/mechanism/neuron_loop.py" summarize "$ROOT" --require-tool-ok
else
  /projects/p33100/siosio/envs/openpi/bin/python -u "$ROOT/source/experiments/mechanism/neuron_loop.py" summarize "$ROOT"
fi
