#!/bin/bash
# Submit the wall-directed braking diagnostic from a login node.  This wrapper performs no
# GPU work itself; it only submits setup/wall_directed_braking.sbatch.
#
# Usage:
#   cd ~/crash_bench
#   bash setup/submit_wall_directed_braking.sh
#   CB_REPEATS=5 bash setup/submit_wall_directed_braking.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
job_id="$(sbatch --parsable "$ROOT/setup/wall_directed_braking.sbatch")"
echo "Submitted wall-directed braking diagnostic: job $job_id"
echo "Log: $ROOT/crashbench_wallbrake_${job_id}.log"
