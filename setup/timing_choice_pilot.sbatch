#!/bin/bash
# P1 smoke: one fresh trajectory, four matched timing anchors, all three options.
# This reuses the frozen exact-state collector and retains raw fresh features.
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=p1_timing_choice
#SBATCH --output=%x_%j.log

set -euo pipefail

OUTPUT="${CB_P1_OUTPUT:?set CB_P1_OUTPUT to a new output directory}"
REPO_ROOT="${CB_REPO_ROOT:-$HOME/crash_bench}"
CHECKPOINT="${CB_BASE_CHECKPOINT:-openvla/openvla-7b-finetuned-libero-spatial}"
REVISION="${CB_BASE_CHECKPOINT_REVISION:-962318cec55ac10993ff0f5f43eda9a270b4c873}"
EXPOSED_PLACEMENTS="${CB_D0_EXPOSED_PLACEMENTS:-results/glass_recovery_v2/pilot_b_task0_20260811_r5/frontier_placements/placements.json}"
P1_PLACEMENTS="${CB_P1_PLACEMENTS:-results/counterfactual_router/fresh_random_9ed5ad656a08_20260817T0845Z/placements/placements.json}"
DETOUR_CONFIG="${CB_CF_DETOUR_CONFIG:-results/counterfactual_router/detour_freeze_partial_d7bbf9f_20260814T1205Z/frozen_detour_config.json}"
IFS='+' read -r -a P1_PLACEMENT_IDS <<< "${CB_P1_PLACEMENT_IDS:-glass_recovery_heldout_0000}"
TARGET_VALID="${CB_P1_TARGET_VALID:-1}"

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
COMMIT="$(git rev-parse HEAD)"
[[ "${CB_CODE_COMMIT:-$COMMIT}" == "$COMMIT" ]]
[[ ! -e "$OUTPUT" ]]

SELECTION_ARGS=()
for placement_id in "${P1_PLACEMENT_IDS[@]}"; do
  SELECTION_ARGS+=(--placement-key "fresh:$placement_id")
done

python scripts/collect_counterfactual_option_rollouts.py \
  --exposed-placements "$EXPOSED_PLACEMENTS" \
  --development-placements "$P1_PLACEMENTS" \
  --output "$OUTPUT" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-revision "$REVISION" \
  --unnorm-key libero_spatial \
  --rollout-seed 0 \
  --horizons 30 20 10 5 \
  --conditions glass \
  --history-length 8 \
  --detector-splits development \
  --target-valid-placements "$TARGET_VALID" \
  "${SELECTION_ARGS[@]}" \
  --retreat-mode hold \
  --detour-config "$DETOUR_CONFIG"
