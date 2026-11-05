#!/usr/bin/env bash
# P4 neuron study, open-loop GPU steps (screen or measure; openpi environment; no simulator, no API). Submitted only by
# launch_neurons.py with CB_NEURON_ACTION set.
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=qgpu2014
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=cb_mech_neurons
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
ACTION="${CB_NEURON_ACTION:?}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$ROOT/GPU_${ACTION}_${SHARD}.txt"
case "$ACTION" in
  screen) /projects/p33100/siosio/envs/openpi/bin/python -u "$SOURCE/experiments/mechanism/run_neurons.py" screen "$ROOT" \
            --checkpoint "${CB_POLICY_CHECKPOINT:?}";;
  measure) /projects/p33100/siosio/envs/openpi/bin/python -u "$SOURCE/experiments/mechanism/run_neurons.py" measure "$ROOT" \
            --shard "$SHARD" --shards "${CB_MECH_SHARDS:?}" --checkpoint "${CB_POLICY_CHECKPOINT:?}";;
  *) echo "Unknown action $ACTION" >&2; exit 2;;
esac
