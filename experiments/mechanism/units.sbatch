#!/usr/bin/env bash
# Mechanism pilot, unit study, open loop (GPU; openpi environment; no simulator, no API). Submitted only by
# launch_units.py.
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=sxm  # 80 GB: units.unit_scores needs ~27 GiB scratch on top of the model (a 40 GB card ran out; HIL_MECHANISM_UNITS_REPORT.md)
#SBATCH --exclude=qgpu2014
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --job-name=cb_mech_units
set -euo pipefail
ROOT="${CB_FD_ROOT:?}"
SOURCE="$ROOT/source"
cd "$SOURCE"
sha256sum --quiet -c "$ROOT/source.sha256"
export CB_CODE_COMMIT="$(cat "$ROOT/SOURCE_COMMIT")"
export CB_ASSETS=/projects/p33100/siosio/crashbench_safelibero
export CB_UPSTREAM="$HOME/crash_bench/third_party/vlsa-aegis"
test "$(cat "$CB_UPSTREAM/.git/HEAD")" = 2457feed5968ae803926e178c8ce8243b9ecdcf9
unset OPENROUTER_API_KEY
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HOME="$CB_ASSETS/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENPI_DATA_HOME="$CB_ASSETS/openpi_assets"
export PYTHONPATH="$CB_UPSTREAM/openpi/src:$CB_UPSTREAM/openpi/packages/openpi-client/src"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$ROOT/GPU_${SHARD}.txt"
/projects/p33100/siosio/envs/openpi/bin/python -u "$SOURCE/experiments/mechanism/run_units.py" run "$ROOT" \
 --shard "$SHARD" --shards "${CB_MECH_SHARDS:?}" --checkpoint "${CB_POLICY_CHECKPOINT:?}"
