#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=cb_safelibero
#SBATCH --output=cb_safelibero_%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/crash_bench}"
/projects/p33100/siosio/envs/openpi/bin/python scripts/safelibero_batch.py run "${CB_BATCH_ROOT:?}" --index "${SLURM_ARRAY_TASK_ID:?}"
