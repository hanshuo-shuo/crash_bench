#!/bin/bash
# P0 stage 2: CPU-only grouped probe/baseline analysis.
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=crashbench_p0_analyze
#SBATCH --output=%x_%j.log

set -euo pipefail
: "${CB_CAPTURE_DIR:?set CB_CAPTURE_DIR to the completed P0 capture directory}"
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1
cd "$HOME/crash_bench"
PYTHON="$HOME/crash_bench/envs/openvla/bin/python"
test -x "$PYTHON"
"$PYTHON" -c 'import sys, numpy; print(f"python={sys.executable}"); print(f"numpy={numpy.__version__}")'
"$PYTHON" scripts/p0_probe_analysis.py "$CB_CAPTURE_DIR"
