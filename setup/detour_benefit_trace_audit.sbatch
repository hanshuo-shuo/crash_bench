#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=detour_trace_audit
#SBATCH --output=%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/crash_bench}"
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
run="results/detour_benefit/a89f755da200_job5791132"
[[ -f "$run/complete.json" ]]
python scripts/expansion/audit_detour_repeat_traces.py --run "$run"
python scripts/expansion/render_detour_benefit.py --analysis "$run/analysis"
