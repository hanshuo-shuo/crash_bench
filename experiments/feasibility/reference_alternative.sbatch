#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=cb_reference_south25
set -euo pipefail
ROOT="${CB_REFERENCE_ROOT:?}"
TARGET="${CB_REFERENCE_TARGET:?}"
cd "$ROOT/source"
sha256sum --quiet -c "$ROOT/source.sha256"
module purge
module load singularityce/4.3.1-gcc-8.5.0
export SINGULARITYENV_CB_UPSTREAM="$HOME/crash_bench/third_party/vlsa-aegis"
export SINGULARITYENV_SLURM_JOB_ID="$SLURM_JOB_ID"
export SINGULARITYENV_PYTHONNOUSERSITE=1 SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
singularity exec --bind "/projects,/home,$ROOT:$ROOT:rw,$ROOT/source:$ROOT/source:ro,$TARGET:$TARGET:ro" /projects/p33100/siosio/crashbench_safelibero/containers/aegis-py38.sif /projects/p33100/siosio/crashbench_safelibero/envs/aegis_sim/bin/python -u "$ROOT/source/experiments/feasibility/reference_alternative.py" "$ROOT" "$TARGET"
