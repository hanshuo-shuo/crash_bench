#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=cb_gain_r2
#SBATCH --output=%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/crash_bench}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
# Quest CPU nodes do not provide the login node's /usr/bin/git.
module purge
module load mamba/24.3.0 git/2.37.2
source activate "$HOME/crash_bench/envs/openvla"
repair_commit=$(git rev-parse HEAD)
test -z "$(git status --porcelain=v1 --untracked-files=all)"
export CB_REPAIR_GIT_COMMIT="$repair_commit"
repair_data=results/expansion/d5_statewise/e2a5264a802a_fe660c168e72_20260830T105726Z/postprocess_a4b1714
repair_output="results/repair_round2/${repair_commit:0:12}_${SLURM_JOB_ID}"
python scripts/expansion/run_round2_gain.py \
  --merged-dir "$repair_data/merged" \
  --feature-cache "$repair_data/feature_cache/features.npz" \
  --artifact-store /projects/p33100/siosio/crash_bench_provenance \
  --output-dir "$repair_output"
