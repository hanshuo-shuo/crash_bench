#!/usr/bin/env bash
# Mechanism pilot, task-pair census: branch plans of the finished census prefix run (CPU, read-only on it). Submitted by
# launch_dose_census.py after the prefix summary; writes PLAN.json into the prepared smoke and formal roots. Fails (and so
# cancels every census job after it) unless the prefix run is complete, unchanged and env_ok.
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=cb_mech_census_plan
set -euo pipefail
ROOT="${CB_FD_ROOT:?}"
cd "$ROOT/source"
sha256sum --quiet -c "$ROOT/source.sha256"
unset OPENROUTER_API_KEY
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
/projects/p33100/siosio/envs/openpi/bin/python -u "$ROOT/source/experiments/mechanism/census.py" plan \
  "${CB_CENSUS_PREFIX_ROOT:?}" "${CB_CENSUS_SMOKE_ROOT:?}" "${CB_CENSUS_FORMAL_ROOT:?}" "${CB_STAGE1_ROOT:?}"
