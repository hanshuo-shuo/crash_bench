#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/crash_bench"
module load git/2.37.2
/projects/p33100/siosio/envs/openpi/bin/python scripts/launch_safelibero_batch.py
