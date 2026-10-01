#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=cb_feasibility_report
set -euo pipefail
ROOT="${CB_ANALYSIS_ROOT:?}"
TARGET="${CB_ANALYSIS_TARGET:?}"
SOURCE="$ROOT/source"
cd "$SOURCE"
sha256sum --quiet -c "$ROOT/source.sha256"
export CB_ANALYSIS_COMMIT="$(cat "$ROOT/SOURCE_COMMIT")"
module purge
module load singularityce/4.3.1-gcc-8.5.0
export SINGULARITYENV_CB_ANALYSIS_COMMIT="$CB_ANALYSIS_COMMIT"
export SINGULARITYENV_SLURM_JOB_ID="$SLURM_JOB_ID"
export SINGULARITYENV_PYTHONNOUSERSITE=1 SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
singularity exec --bind "/projects,/home,$ROOT:$ROOT:rw,$SOURCE:$SOURCE:ro,$TARGET:$TARGET:ro" /projects/p33100/siosio/crashbench_safelibero/containers/aegis-py38.sif /projects/p33100/siosio/crashbench_safelibero/envs/aegis_sim/bin/python -u "$SOURCE/experiments/feasibility/interpret.py" "$TARGET" "$ROOT/report"
