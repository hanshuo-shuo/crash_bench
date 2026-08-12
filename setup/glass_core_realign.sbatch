#!/bin/bash
# Pilot A only: exact-H realignment of provenance-reconstructed E14 pairs.
# This job never writes into the historical E14 root and never starts Pilot B.
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=glass_core_realign
#SBATCH --output=%x_%j.log

set -euo pipefail

if [[ "${CB_ENABLE_LEGACY_GLASS_RECOVERY:-0}" != "1" ]]; then
  echo "legacy Pilot A realignment is disabled: the provenance audit is complete" >&2
  echo "set CB_ENABLE_LEGACY_GLASS_RECOVERY=1 only for an explicit provenance diagnostic" >&2
  exit 2
fi

require_env() {
  local name="$1"
  [[ -n "${!name:-}" ]] || { echo "required environment variable is empty: $name" >&2; exit 2; }
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || { echo "required input file does not exist: $path" >&2; exit 2; }
}

for name in \
  CB_PILOT_A_SOURCE_ROOT \
  CB_PILOT_A_AUDIT \
  CB_PILOT_A_INVENTORY_SUMMARY \
  CB_PILOT_A_PLACEMENTS \
  CB_PILOT_A_OUTPUT_ROOT \
  CB_PILOT_A_SUMMARY_OUT \
  CB_PILOT_A_H
do
  require_env "$name"
done
for path in \
  "$CB_PILOT_A_AUDIT" \
  "$CB_PILOT_A_INVENTORY_SUMMARY" \
  "$CB_PILOT_A_PLACEMENTS"
do
  require_file "$path"
done
[[ -d "$CB_PILOT_A_SOURCE_ROOT" ]] \
  || { echo "historical source root does not exist: $CB_PILOT_A_SOURCE_ROOT" >&2; exit 2; }

echo "JOB $SLURM_JOB_ID stage=pilot_a_realign on $(hostname) @ $(date)"
nvidia-smi -L
module purge
module load mamba/24.3.0 git/2.47.0-gcc-12.4.0
GIT_BIN="${CB_GIT_BIN:-$(command -v git)}"
source activate "$HOME/crash_bench/envs/openvla"

export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/projects/p33100/siosio/huggingface_cache
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export TORCH_HOME=/projects/p33100/siosio/torch_cache

REPO_ROOT="${CB_REPO_ROOT:-$HOME/crash_bench}"
export PYTHONPATH="$REPO_ROOT/third_party/openvla:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
"$GIT_BIN" diff --quiet && "$GIT_BIN" diff --cached --quiet \
  || { echo "tracked source differs from HEAD" >&2; exit 1; }
export CB_CODE_COMMIT="${CB_CODE_COMMIT:-$("$GIT_BIN" rev-parse HEAD)}"
[[ "$CB_CODE_COMMIT" == "$("$GIT_BIN" rev-parse HEAD)" ]] \
  || { echo "CB_CODE_COMMIT does not match checked-out HEAD" >&2; exit 1; }

python scripts/realign_glass_core_artifacts.py \
  --source-root "$CB_PILOT_A_SOURCE_ROOT" \
  --audit "$CB_PILOT_A_AUDIT" \
  --inventory-summary "$CB_PILOT_A_INVENTORY_SUMMARY" \
  --placements "$CB_PILOT_A_PLACEMENTS" \
  --output-root "$CB_PILOT_A_OUTPUT_ROOT" \
  --summary-out "$CB_PILOT_A_SUMMARY_OUT" \
  --target-h "$CB_PILOT_A_H" \
  --oracle-steps "${CB_PILOT_A_ORACLE_STEPS:-900}" \
  --expected-candidates "${CB_PILOT_A_EXPECTED_CANDIDATES:-3}"

require_file "$CB_PILOT_A_SUMMARY_OUT"
echo "SUMMARY=$CB_PILOT_A_SUMMARY_OUT"
echo "DONE @ $(date)"
