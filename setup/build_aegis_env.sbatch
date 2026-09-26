#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=cb_aegis_env
#SBATCH --output=cb_aegis_env_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/crash_bench}"
A=/projects/p33100/siosio/crashbench_safelibero
module purge
module load singularityce/4.3.1-gcc-8.5.0
export SINGULARITYENV_CB_ASSETS="$A"
export SINGULARITYENV_TORCH_CUDA_ARCH_LIST=8.0
export SINGULARITYENV_MAX_JOBS=4
export SINGULARITYENV_PYTHONNOUSERSITE=1
export SINGULARITYENV_PIP_NO_INDEX=1
export SINGULARITYENV_PIP_CACHE_DIR="$A/pip_cache"
singularity exec --nv --bind "/projects,/home,$PWD:$PWD" "$A/containers/aegis-py38.sif" bash setup/build_sim_env.sh
