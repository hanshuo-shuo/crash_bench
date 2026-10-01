#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --job-name=cb_observation_probe
set -euo pipefail
ROOT="${CB_DIAGNOSTIC_ROOT:?}"
SOURCE="$ROOT/source"
cd "$SOURCE"
sha256sum --quiet -c "$ROOT/source.sha256"
module purge
module load singularityce/4.3.1-gcc-8.5.0
export SINGULARITYENV_CB_UPSTREAM="$HOME/crash_bench/third_party/vlsa-aegis"
export SINGULARITYENV_PYTHONNOUSERSITE=1 SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
singularity exec --bind "/projects,/home,$ROOT:$ROOT:rw,$SOURCE:$SOURCE:ro" /projects/p33100/siosio/crashbench_safelibero/containers/aegis-py38.sif /projects/p33100/siosio/crashbench_safelibero/envs/aegis_sim/bin/python -u "$SOURCE/experiments/feasibility/observation_probe.py" /projects/p33100/siosio/crashbench_safelibero/feasibility/20261001T181952Z_smoke_f8d89948b49a "$ROOT/observation_probe"
