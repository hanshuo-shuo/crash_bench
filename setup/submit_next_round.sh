#!/bin/bash
# Login-node submitter for the current paper round.
# This script only calls sbatch; all GPU work happens on the compute nodes.
#
# Usage:
#   bash setup/submit_next_round.sh m0
#   bash setup/submit_next_round.sh gate
#   bash setup/submit_next_round.sh baseline
#   bash setup/submit_next_round.sh shield
#   bash setup/submit_next_round.sh all

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

submit() {
  local script="$1"
  local job_id
  job_id="$(sbatch --parsable "$ROOT/setup/$script")"
  echo "$script -> job $job_id"
}

case "${1:-}" in
  m0)
    python3 scripts/prepare_m0_geometry.py
    ;;
  gate)
    submit m1_nominal_gate.sbatch
    ;;
  baseline)
    submit run_prompted_careful.sbatch
    ;;
  shield)
    submit phase3_intervention_expanded.sbatch
    ;;
  all)
    submit m1_nominal_gate.sbatch
    submit run_prompted_careful.sbatch
    submit phase3_intervention_expanded.sbatch
    ;;
  *)
    echo "usage: bash setup/submit_next_round.sh {gate|baseline|shield|all}" >&2
    exit 2
    ;;
esac
