#!/usr/bin/env bash
# Existing captured features only: no simulator, VLA inference, or new rollout.
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=cb_recoverability_probe
#SBATCH --output=%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/crash_bench}"
module purge
module load git/2.47.0-gcc-12.4.0 mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
recoverability_probe_commit=$(git rev-parse HEAD)
test -z "$(git status --porcelain=v1 --untracked-files=all)"
export CB_CODE_COMMIT="$recoverability_probe_commit"
recoverability_probe_output="results/recoverability_hidden_probe/${recoverability_probe_commit:0:12}_${SLURM_JOB_ID}"
python -m unittest discover -s tests -p test_recoverability_hidden_probe.py
python scripts/analysis/recoverability_hidden_probe.py \
  --config configs/recoverability/hidden_probe_v1.json \
  --output-dir "$recoverability_probe_output"
