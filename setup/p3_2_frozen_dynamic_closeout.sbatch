#!/bin/bash
# P3.2: the one immutable 8-source, three-method dynamic closeout.
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --job-name=p3_2_closeout
#SBATCH --output=%x_%j.log

set -euo pipefail

RUN_ROOT="${CB_P3_2_RUN_ROOT:?set CB_P3_2_RUN_ROOT}"
REPO_ROOT="${CB_REPO_ROOT:-$HOME/crash_bench}"
OUTPUT="$RUN_ROOT/closeout"
PLACEMENTS="$RUN_ROOT/fresh_cohort/placements/placements.json"
FEATURE_ROUTER="${CB_P3_2_FEATURE_ROUTER:-results/counterfactual_router/fresh_online_2eab4a4a53dc_20260817T080708Z/router.json}"
OLD_BOUNDARY="${CB_P3_2_OLD_BOUNDARY:-results/counterfactual_router/p2_sequential_calibration_4cbe033e7850_20260818T152624Z/sequential_boundary.json}"
DIRECT_HEAD="${CB_P3_2_DIRECT_HEAD:-results/p3_1_direct_recovery_head_model.npz}"
DIRECT_BOUNDARY="$RUN_ROOT/calibration/sequential_boundary.json"
SUPERVISION="${CB_P3_2_SUPERVISION:-results/p3_recovery_window_records_20260819.jsonl}"
DETOUR_CONFIG="${CB_CF_DETOUR_CONFIG:-results/counterfactual_router/detour_freeze_partial_d7bbf9f_20260814T1205Z/frozen_detour_config.json}"
CHECKPOINT="${CB_BASE_CHECKPOINT:-openvla/openvla-7b-finetuned-libero-spatial}"
REVISION="${CB_BASE_CHECKPOINT_REVISION:-962318cec55ac10993ff0f5f43eda9a270b4c873}"

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
[[ -f "$FEATURE_ROUTER" ]]
[[ -f "$OLD_BOUNDARY" ]]
[[ -f "$DIRECT_HEAD" ]]
[[ -f "$DIRECT_BOUNDARY" ]]
[[ -f "$SUPERVISION" ]]
[[ -f "$DETOUR_CONFIG" ]]
[[ ! -e "$OUTPUT" ]]

python scripts/collect_direct_recovery_head_dynamic_closeout.py \
  --placements "$PLACEMENTS" \
  --feature-router-model "$FEATURE_ROUTER" \
  --old-sequential-boundary "$OLD_BOUNDARY" \
  --direct-head-model "$DIRECT_HEAD" \
  --direct-sequential-boundary "$DIRECT_BOUNDARY" \
  --supervision-records "$SUPERVISION" \
  --detour-config "$DETOUR_CONFIG" \
  --output "$OUTPUT" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-revision "$REVISION" \
  --unnorm-key libero_spatial \
  --rollout-seed 0 \
  --retreat-mode hold

python scripts/analyze_direct_recovery_head_dynamic_closeout.py \
  --capture "$OUTPUT"

echo "DONE P3.2 closeout commit=$COMMIT output=$OUTPUT @ $(date)"
