#!/bin/bash
# P2 sequential calibration: complete Base-success control trajectories.
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=p2_seq_cal
#SBATCH --output=%x_%j.log

set -euo pipefail

OUTPUT="${CB_P2_CAL_OUTPUT:?set CB_P2_CAL_OUTPUT to a new output directory}"
REPO_ROOT="${CB_REPO_ROOT:-$HOME/crash_bench}"
INPUT_ROOT="${CB_P2_INPUT_ROOT:-results/counterfactual_router/fresh_online_2eab4a4a53dc_20260817T080708Z}"
PLACEMENTS="${CB_P2_CAL_PLACEMENTS:-results/glass_recovery_v2/pilot_b_task0_20260811_r5/frontier_placements/placements.json}"
ROUTER="${CB_P2_ROUTER_MODEL:-$INPUT_ROOT/router.json}"
ROUTER_CAPTURE="${CB_P2_ROUTER_CAPTURE:-results/counterfactual_router/full_d4751330395e_20260814T152231Z/capture_manifest.json}"
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
[[ -f "$ROUTER" ]]
[[ -f "$ROUTER_CAPTURE" ]]
[[ ! -e "$OUTPUT" ]]

python scripts/calibrate_dynamic_first_crossing_router.py \
  --placements "$PLACEMENTS" \
  --router-model "$ROUTER" \
  --router-capture-manifest "$ROUTER_CAPTURE" \
  --output "$OUTPUT" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-revision "$REVISION" \
  --unnorm-key libero_spatial \
  --rollout-seed 0 \
  --splits validation heldout \
  --conditions offpath noglass \
  --catastrophe-cost 1 \
  --target-intervention-rate 0.4 \
  --alphas 0.1 0.2 0.3 0.4 \
  --primary-alpha 0.1

echo "DONE commit=$COMMIT output=$OUTPUT @ $(date)"
