#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=cb_fd1000_export
set -euo pipefail
ROOT="${CB_FD_ROOT:?}"
cd "$ROOT/source"
sha256sum --quiet -c "$ROOT/source.sha256"
unset OPENROUTER_API_KEY
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
/projects/p33100/siosio/envs/openpi/bin/python -u -m verifier_dataset.export_candidates "$ROOT" --replay-root "${CB_FD_REPLAY:?}"
