#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:03:00
#SBATCH --job-name=cb_feasibility_next
set -euo pipefail
CONTROLLER="${CB_FEASIBILITY_CONTROLLER:?}"
TARGET="${CB_FEASIBILITY_TARGET:?}"
PREDECESSOR="${CB_FEASIBILITY_PREDECESSOR:?}"
cd "$CONTROLLER/source"
sha256sum --quiet -c "$CONTROLLER/source.sha256"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
/projects/p33100/siosio/envs/openpi/bin/python "$CONTROLLER/source/experiments/feasibility/orchestrate.py" "$TARGET" "$CONTROLLER" "$PREDECESSOR"
