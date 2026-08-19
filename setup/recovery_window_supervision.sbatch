#!/bin/bash
# P3.0: dense exact-state Base/Detour/FailSafeHold recovery-window labels.
# Development authoring only: no training, Router retuning, or dynamic evaluation.
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=p3_recovery_window
#SBATCH --output=%x_%j.log

set -euo pipefail

OUTPUT="${CB_P3_CAPTURE_OUTPUT:?set CB_P3_CAPTURE_OUTPUT to a new output directory}"
REPO_ROOT="${CB_REPO_ROOT:-$HOME/crash_bench}"
INPUT_ROOT="${CB_P3_INPUT_ROOT:-results/counterfactual_router/fresh_online_2eab4a4a53dc_20260817T080708Z}"
PLACEMENTS="${CB_P3_PLACEMENTS:-results/counterfactual_router/fresh_random_9ed5ad656a08_20260817T0845Z/placements/placements.json}"
EXPOSED_PLACEMENTS="${CB_D0_EXPOSED_PLACEMENTS:-results/glass_recovery_v2/pilot_b_task0_20260811_r5/frontier_placements/placements.json}"
ROUTER="${CB_P3_ROUTER_MODEL:-$INPUT_ROOT/router.json}"
HARD_CONTROLS="${CB_P3_HARD_CONTROLS:-results/counterfactual_router/p2_dynamic_exact_seq_0f92a1b420dd_20260818T162140Z}"
DETOUR_CONFIG="${CB_CF_DETOUR_CONFIG:-results/counterfactual_router/detour_freeze_partial_d7bbf9f_20260814T1205Z/frozen_detour_config.json}"
CHECKPOINT="${CB_BASE_CHECKPOINT:-openvla/openvla-7b-finetuned-libero-spatial}"
REVISION="${CB_BASE_CHECKPOINT_REVISION:-962318cec55ac10993ff0f5f43eda9a270b4c873}"
P3_PLACEMENT_IDS="${CB_P3_PLACEMENT_IDS:-glass_recovery_heldout_0010+glass_recovery_heldout_0019+glass_recovery_heldout_0009+glass_recovery_heldout_0004+glass_recovery_heldout_0003}"

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
[[ -f "$EXPOSED_PLACEMENTS" ]]
[[ -f "$ROUTER" ]]
[[ -f "$HARD_CONTROLS/capture_manifest.json" ]]
[[ -f "$HARD_CONTROLS/dynamic_episodes.jsonl" ]]
[[ -f "$HARD_CONTROLS/router_trace.jsonl" ]]
[[ -f "$DETOUR_CONFIG" ]]
[[ ! -e "$OUTPUT" ]]

SELECTION_ARGS=()
IFS='+' read -r -a SELECTED_PLACEMENTS <<< "$P3_PLACEMENT_IDS"
for placement_id in "${SELECTED_PLACEMENTS[@]}"; do
  SELECTION_ARGS+=(--placement-key "fresh:$placement_id")
done

python scripts/collect_counterfactual_option_rollouts.py \
  --exposed-placements "$EXPOSED_PLACEMENTS" \
  --development-placements "$PLACEMENTS" \
  --output "$OUTPUT" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-revision "$REVISION" \
  --unnorm-key libero_spatial \
  --rollout-seed 0 \
  --horizons 30 28 26 24 22 20 18 16 14 12 10 8 6 4 2 1 \
  --conditions glass \
  --history-length 8 \
  --detector-splits development \
  --target-valid-placements "${#SELECTED_PLACEMENTS[@]}" \
  "${SELECTION_ARGS[@]}" \
  --retreat-mode hold \
  --detour-config "$DETOUR_CONFIG"

python scripts/author_recovery_window_supervision.py \
  --capture "$OUTPUT" \
  --router-model "$ROUTER" \
  --hard-control-capture "$HARD_CONTROLS" \
  --quest-job-id "$SLURM_JOB_ID"

echo "DONE commit=$COMMIT job=$SLURM_JOB_ID output=$OUTPUT @ $(date)"
