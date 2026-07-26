#!/bin/bash
# Login-node submission wrapper for P0 task selection and scenario authoring.
#
# CB_CHECKPOINT_REVISION=<40hex> bash setup/submit_p0_authoring.sh all

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${CB_CHECKPOINT_REVISION:?export the exact immutable checkpoint revision}"
export CB_GATE_OUT="${CB_GATE_OUT:-results/p0_runs/p0_nominal_gate_${CB_CHECKPOINT_REVISION:0:12}.json}"

case "${1:-}" in
  gate)
    sbatch --export=ALL,CB_CHECKPOINT_REVISION="$CB_CHECKPOINT_REVISION",CB_GATE_OUT="$CB_GATE_OUT" \
      setup/p0_nominal_gate.sbatch
    ;;
  author)
    sbatch --export=ALL,CB_CHECKPOINT_REVISION="$CB_CHECKPOINT_REVISION",CB_GATE_OUT="$CB_GATE_OUT" \
      setup/p0_author_scenarios.sbatch
    ;;
  all)
    gate_job="$(sbatch --parsable --export=ALL,CB_CHECKPOINT_REVISION="$CB_CHECKPOINT_REVISION",CB_GATE_OUT="$CB_GATE_OUT" setup/p0_nominal_gate.sbatch)"
    author_job="$(sbatch --parsable --dependency="afterok:$gate_job" --export=ALL,CB_CHECKPOINT_REVISION="$CB_CHECKPOINT_REVISION",CB_GATE_OUT="$CB_GATE_OUT" setup/p0_author_scenarios.sbatch)"
    echo "gate=$gate_job author=$author_job"
    echo "gate result: $CB_GATE_OUT"
    ;;
  *)
    echo "usage: CB_CHECKPOINT_REVISION=<40hex> bash setup/submit_p0_authoring.sh {gate|author|all}" >&2
    exit 2
    ;;
esac

