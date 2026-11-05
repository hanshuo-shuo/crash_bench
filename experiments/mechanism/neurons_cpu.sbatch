#!/usr/bin/env bash
# P4 neuron study, open-loop CPU steps (select or summarize; numpy only). Submitted only by launch_neurons.py.
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=cb_mech_neurons_cpu
set -euo pipefail
ROOT="${CB_FD_ROOT:?}"
cd "$ROOT/source"
sha256sum --quiet -c "$ROOT/source.sha256"
export CB_CODE_COMMIT="$(cat "$ROOT/SOURCE_COMMIT")"
unset OPENROUTER_API_KEY
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
ACTION="${CB_NEURON_ACTION:?}"
case "$ACTION" in
  select|summarize) /projects/p33100/siosio/envs/openpi/bin/python -u "$ROOT/source/experiments/mechanism/run_neurons.py" "$ACTION" "$ROOT";;
  *) echo "Unknown action $ACTION" >&2; exit 2;;
esac
