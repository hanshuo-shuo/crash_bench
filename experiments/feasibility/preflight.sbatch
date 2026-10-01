#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --job-name=cb_feasibility_preflight
set -euo pipefail
ROOT="${CB_DIAGNOSTIC_ROOT:?}"
SOURCE="$ROOT/source"
cd "$SOURCE"
sha256sum --quiet -c "$ROOT/source.sha256"
module purge
module load singularityce/4.3.1-gcc-8.5.0
export SINGULARITYENV_CB_UPSTREAM="$HOME/crash_bench/third_party/vlsa-aegis"
export SINGULARITYENV_CB_ASSETS=/projects/p33100/siosio/crashbench_safelibero
export SINGULARITYENV_CB_CODE_COMMIT="$(cat "$ROOT/SOURCE_COMMIT")"
export SINGULARITYENV_PYTHONNOUSERSITE=1 SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
export SINGULARITYENV_PYTHONPATH="$HOME/crash_bench/third_party/vlsa-aegis/safelibero"
singularity exec --bind "/projects,/home,$ROOT:$ROOT:rw,$SOURCE:$SOURCE:ro" /projects/p33100/siosio/crashbench_safelibero/containers/aegis-py38.sif /projects/p33100/siosio/crashbench_safelibero/envs/aegis_sim/bin/python -u "$SOURCE/experiments/feasibility/preflight.py" "$ROOT"
