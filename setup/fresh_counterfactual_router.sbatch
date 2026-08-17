#!/bin/bash
# Fresh matched online branch evaluation; submit through the wrapper below.
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --job-name=cf_fresh_online
#SBATCH --output=%x_%j.log

set -euo pipefail

MODE="${CB_CF_FRESH_MODE:?set CB_CF_FRESH_MODE to smoke or full}"
RUN_ROOT="${CB_CF_FRESH_ROOT:?set CB_CF_FRESH_ROOT}"
CHECKPOINT="${CB_BASE_CHECKPOINT:-openvla/openvla-7b-finetuned-libero-spatial}"
REVISION="${CB_BASE_CHECKPOINT_REVISION:-962318cec55ac10993ff0f5f43eda9a270b4c873}"
DETOUR_CONFIG="${CB_CF_DETOUR_CONFIG:-results/counterfactual_router/detour_freeze_partial_d7bbf9f_20260814T1205Z/frozen_detour_config.json}"
case "$MODE" in
  smoke|full) ;;
  *) echo "CB_CF_FRESH_MODE must be smoke or full" >&2; exit 2 ;;
esac

module purge
module load mamba/24.3.0 git/2.47.0-gcc-12.4.0
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
git diff --quiet
git diff --cached --quiet
COMMIT="$(git rev-parse HEAD)"
[[ "${CB_CODE_COMMIT:-$COMMIT}" == "$COMMIT" ]]
[[ -f "$RUN_ROOT/router.json" ]]
[[ -f "$RUN_ROOT/placements/placements.json" ]]
[[ -f "$DETOUR_CONFIG" ]]
OUTPUT="$RUN_ROOT/$MODE"
[[ ! -e "$OUTPUT" ]]

SELECTION_ARGS=()
if [[ "$MODE" == smoke ]]; then
  # These first two source states are mechanical smoke only and are excluded
  # from the official full cohort even when the first one succeeds.
  SELECTION_ARGS=(--max-placements 2 --target-valid-placements 1)
else
  SELECTION_ARGS=(--skip-placements 2)
fi

python scripts/collect_fresh_counterfactual_router.py \
  --placements "$RUN_ROOT/placements/placements.json" \
  --router-model "$RUN_ROOT/router.json" \
  --detour-config "$DETOUR_CONFIG" \
  --output "$OUTPUT" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-revision "$REVISION" \
  --unnorm-key libero_spatial \
  --rollout-seed 0 \
  --horizon 20 \
  "${SELECTION_ARGS[@]}"

if [[ "$MODE" == full ]]; then
  python scripts/analyze_fresh_counterfactual_router.py \
    --capture "$OUTPUT" \
    --bootstrap-replicates 5000 \
    --bootstrap-seed 20260817
fi

echo "DONE mode=$MODE commit=$COMMIT output=$OUTPUT @ $(date)"
