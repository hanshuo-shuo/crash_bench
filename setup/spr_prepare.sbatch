#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --job-name=cb_spr_prepare
#SBATCH --output=%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/crash_bench}"
module purge
module load mamba/24.3.0 git/2.47.0-gcc-12.4.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=0 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=/projects/p33100/siosio/crashbench_external_recovery/sprvla/hf_cache
export PIP_CACHE_DIR=/projects/p33100/siosio/crashbench_external_recovery/sprvla/pip_cache
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test "$(git rev-parse HEAD)" = "$CB_SPR_COMMIT"
python scripts/expansion/spr_preflight.py --stage prepare --run-dir "$CB_SPR_OUTPUT"
