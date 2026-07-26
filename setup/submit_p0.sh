#!/bin/bash
# Login-node orchestrator for the new P0 pipeline.
#
#   CB_P0_CONFIG=configs/p0_core.json bash setup/submit_p0.sh preflight
#   CB_P0_CONFIG=configs/p0_core.json bash setup/submit_p0.sh all
#
# capture/analyze/guard may also be submitted separately.  For analyze/guard, set
# CB_CAPTURE_DIR; for guard, optionally set CB_GUARD_OUT.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CB_P0_CONFIG="${CB_P0_CONFIG:-configs/p0_core.json}"
P0_PYTHON="${CB_PYTHON:-$HOME/crash_bench/envs/openvla/bin/python}"
if [[ ! -x "$P0_PYTHON" ]]; then
  echo "missing P0 interpreter: $P0_PYTHON" >&2
  exit 2
fi

if [[ ! -f "$CB_P0_CONFIG" ]]; then
  echo "missing $CB_P0_CONFIG; copy configs/p0_core.example.json and fill every REPLACE_* value" >&2
  exit 2
fi

capture_dir="${CB_CAPTURE_DIR:-$("$P0_PYTHON" -c 'import json,os; print(json.load(open(os.environ["CB_P0_CONFIG"]))["output_dir"])')}"
guard_out="${CB_GUARD_OUT:-${capture_dir}_online_guard}"

case "${1:-}" in
  preflight)
    "$P0_PYTHON" scripts/p0_capture.py --config "$CB_P0_CONFIG" --preflight-only
    ;;
  capture)
    sbatch --export=ALL,CB_P0_CONFIG="$CB_P0_CONFIG" setup/p0_capture.sbatch
    ;;
  analyze)
    sbatch --export=ALL,CB_CAPTURE_DIR="$capture_dir" setup/p0_analyze.sbatch
    ;;
  guard)
    sbatch --export=ALL,CB_P0_CONFIG="$CB_P0_CONFIG",CB_CAPTURE_DIR="$capture_dir",CB_GUARD_OUT="$guard_out" setup/p0_guard.sbatch
    ;;
  all)
    capture_job="$(sbatch --parsable --export=ALL,CB_P0_CONFIG="$CB_P0_CONFIG" setup/p0_capture.sbatch)"
    analyze_job="$(sbatch --parsable --dependency="afterok:$capture_job" --export=ALL,CB_CAPTURE_DIR="$capture_dir" setup/p0_analyze.sbatch)"
    guard_job="$(sbatch --parsable --dependency="afterok:$analyze_job" --export=ALL,CB_P0_CONFIG="$CB_P0_CONFIG",CB_CAPTURE_DIR="$capture_dir",CB_GUARD_OUT="$guard_out" setup/p0_guard.sbatch)"
    echo "capture=$capture_job analyze=$analyze_job guard=$guard_job"
    ;;
  *)
    echo "usage: CB_P0_CONFIG=configs/p0_core.json bash setup/submit_p0.sh {preflight|capture|analyze|guard|all}" >&2
    exit 2
    ;;
esac
