#!/bin/bash
# P2: from-reset dynamic first-crossing Router plus matched T-20 timing bound.
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --job-name=p2_dynamic_router
#SBATCH --output=%x_%j.log

set -euo pipefail

OUTPUT="${CB_P2_OUTPUT:?set CB_P2_OUTPUT to a new output directory}"
REPO_ROOT="${CB_REPO_ROOT:-$HOME/crash_bench}"
INPUT_ROOT="${CB_P2_INPUT_ROOT:-results/counterfactual_router/fresh_online_2eab4a4a53dc_20260817T080708Z}"
PLACEMENTS="${CB_P2_PLACEMENTS:-$INPUT_ROOT/placements/placements.json}"
ROUTER="${CB_P2_ROUTER_MODEL:-$INPUT_ROOT/router.json}"
DETOUR_CONFIG="${CB_CF_DETOUR_CONFIG:-results/counterfactual_router/detour_freeze_partial_d7bbf9f_20260814T1205Z/frozen_detour_config.json}"
CHECKPOINT="${CB_BASE_CHECKPOINT:-openvla/openvla-7b-finetuned-libero-spatial}"
REVISION="${CB_BASE_CHECKPOINT_REVISION:-962318cec55ac10993ff0f5f43eda9a270b4c873}"
TARGET_VALID="${CB_P2_TARGET_VALID:-}"

module purge
module load mamba/24.3.0 git/2.47.0-gcc-12.4.0
source activate "$REPO_ROOT/envs/openvla"
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/projects/p33100/siosio/huggingface_cache
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export TORCH_HOME=/projects/p33100/siosio/torch_cache
export PYTHONPATH="$REPO_ROOT/third_party/openvla:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT"
git diff --quiet
git diff --cached --quiet
COMMIT="$(git rev-parse HEAD)"
[[ "${CB_CODE_COMMIT:-$COMMIT}" == "$COMMIT" ]]
[[ -f "$PLACEMENTS" ]]
[[ -f "$ROUTER" ]]
[[ -f "$DETOUR_CONFIG" ]]
[[ ! -e "$OUTPUT" ]]

SELECTION_ARGS=()
IFS='+' read -r -a P2_PLACEMENT_IDS <<< "${CB_P2_PLACEMENT_IDS:-}"
for placement_id in "${P2_PLACEMENT_IDS[@]}"; do
  [[ -z "$placement_id" ]] || SELECTION_ARGS+=(--placement-id "$placement_id")
done
if [[ -n "$TARGET_VALID" ]]; then
  SELECTION_ARGS+=(--target-valid-placements "$TARGET_VALID")
fi

python scripts/collect_dynamic_first_crossing_router.py \
  --placements "$PLACEMENTS" \
  --router-model "$ROUTER" \
  --detour-config "$DETOUR_CONFIG" \
  --output "$OUTPUT" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-revision "$REVISION" \
  --unnorm-key libero_spatial \
  --rollout-seed 0 \
  --catastrophe-cost 1 \
  --target-intervention-rate 0.6 \
  --oracle-horizon 20 \
  --retreat-mode hold \
  "${SELECTION_ARGS[@]}"

python scripts/analyze_dynamic_first_crossing_router.py --capture "$OUTPUT"

echo "DONE commit=$COMMIT output=$OUTPUT @ $(date)"
