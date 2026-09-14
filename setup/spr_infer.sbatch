#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:35:00
#SBATCH --job-name=cb_spr_interface
#SBATCH --output=%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/crash_bench}"
module purge
module load git/2.47.0-gcc-12.4.0
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/projects/p33100/siosio/crashbench_external_recovery/sprvla/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V1=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test "$(git rev-parse HEAD)" = "$CB_SPR_COMMIT"
"$CB_SPR_OUTPUT/venv/bin/python" scripts/expansion/spr_preflight.py --stage infer --run-dir "$CB_SPR_OUTPUT"
