#!/bin/bash
# Evaluate the already-frozen router on held-out random reset initializations.
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --job-name=cf_fresh_random
#SBATCH --output=%x_%j.log

set -euo pipefail

RUN_ROOT="${CB_CF_SUPPLEMENT_ROOT:?set CB_CF_SUPPLEMENT_ROOT}"
CHECKPOINT="${CB_BASE_CHECKPOINT:-openvla/openvla-7b-finetuned-libero-spatial}"
REVISION="${CB_BASE_CHECKPOINT_REVISION:-962318cec55ac10993ff0f5f43eda9a270b4c873}"
DETOUR_CONFIG="${CB_CF_DETOUR_CONFIG:-results/counterfactual_router/detour_freeze_partial_d7bbf9f_20260814T1205Z/frozen_detour_config.json}"

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
OUTPUT="$RUN_ROOT/full"
[[ ! -e "$OUTPUT" ]]

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
  --target-valid-placements 8

python scripts/analyze_fresh_counterfactual_router.py \
  --capture "$OUTPUT" \
  --bootstrap-replicates 5000 \
  --bootstrap-seed 20260817

echo "DONE fresh_random commit=$COMMIT output=$OUTPUT @ $(date)"
