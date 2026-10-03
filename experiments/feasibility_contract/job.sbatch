#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=cb_contract_pilot
set -euo pipefail
ROOT="${CB_CONTRACT_ROOT:?}"
SOURCE="$ROOT/source"
cd "$SOURCE"
sha256sum --quiet -c "$ROOT/source.sha256"
module purge
module load singularityce/4.3.1-gcc-8.5.0
unset OPENROUTER_API_KEY
export SINGULARITYENV_CB_UPSTREAM="$HOME/crash_bench/third_party/vlsa-aegis"
export SINGULARITYENV_CB_CODE_COMMIT="$(cat "$ROOT/SOURCE_COMMIT")"
export SINGULARITYENV_PYTHONNOUSERSITE=1 SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
export SINGULARITYENV_OMP_NUM_THREADS=1 SINGULARITYENV_OPENBLAS_NUM_THREADS=1
export SINGULARITYENV_MUJOCO_GL=glx
export SINGULARITYENV_PYTHONPATH="$HOME/crash_bench/third_party/vlsa-aegis/safelibero"
singularity exec --bind "/projects,/home,$ROOT:$ROOT:rw,$SOURCE:$SOURCE:ro" \
 /projects/p33100/siosio/crashbench_safelibero/containers/aegis-py38.sif \
 /projects/p33100/siosio/crashbench_safelibero/envs/aegis_sim/bin/python -u \
 "$SOURCE/experiments/feasibility_contract/run.py" "$ROOT" "${CB_CONTRACT_STAGE:?}"
