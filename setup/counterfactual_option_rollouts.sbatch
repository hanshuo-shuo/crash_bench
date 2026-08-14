#!/bin/bash
# Exact-state multi-H option-outcome capture for the counterfactual router.
# Submit through setup/submit_counterfactual_option_rollouts.sh.
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --job-name=cf_option_rollouts
#SBATCH --output=%x_%j.log

set -euo pipefail

MODE="${CB_CF_MODE:?set CB_CF_MODE to smoke or full}"
OUTPUT="${CB_CF_OUTPUT:?set CB_CF_OUTPUT to a new ignored result directory}"
case "$MODE" in
  smoke|full) ;;
  *) echo "CB_CF_MODE must be smoke or full" >&2; exit 2 ;;
esac

CHECKPOINT="${CB_BASE_CHECKPOINT:-openvla/openvla-7b-finetuned-libero-spatial}"
REVISION="${CB_BASE_CHECKPOINT_REVISION:-962318cec55ac10993ff0f5f43eda9a270b4c873}"
UNNORM_KEY="${CB_BASE_UNNORM_KEY:-libero_spatial}"
EXPOSED_PLACEMENTS="${CB_D0_EXPOSED_PLACEMENTS:-results/glass_recovery_v2/pilot_b_task0_20260811_r5/frontier_placements/placements.json}"
DEVELOPMENT_PLACEMENTS="${CB_D0_DEVELOPMENT_PLACEMENTS:-results/glass_recovery_v2/pilot_b_fresh_h20_20260812_r2/frontier_placements/placements.json}"

echo "JOB $SLURM_JOB_ID mode=$MODE on $(hostname) @ $(date)"
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
"$GIT_BIN" diff --quiet
"$GIT_BIN" diff --cached --quiet
COMMIT="$($GIT_BIN rev-parse HEAD)"
[[ "${CB_CODE_COMMIT:-$COMMIT}" == "$COMMIT" ]]
[[ ! -e "$OUTPUT" ]] || { echo "output already exists: $OUTPUT" >&2; exit 2; }
[[ -f "$EXPOSED_PLACEMENTS" ]] || { echo "missing $EXPOSED_PLACEMENTS" >&2; exit 2; }
[[ -f "$DEVELOPMENT_PLACEMENTS" ]] || { echo "missing $DEVELOPMENT_PLACEMENTS" >&2; exit 2; }

SELECTION_ARGS=()
if [[ "$MODE" == "smoke" ]]; then
  # Try predeclared development candidates until one produces a usable Base
  # catastrophe.  A single stochastic no-crash rollout is an audited exclusion,
  # not grounds for an empty smoke dataset.
  SELECTION_ARGS=(
    --detector-splits development
    --placement-key fresh:glass_recovery_heldout_0000
    --placement-key fresh:glass_recovery_heldout_0004
    --placement-key fresh:glass_recovery_train_0003
    --target-valid-placements 1
  )
fi

python scripts/collect_counterfactual_option_rollouts.py \
  --exposed-placements "$EXPOSED_PLACEMENTS" \
  --development-placements "$DEVELOPMENT_PLACEMENTS" \
  --output "$OUTPUT" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-revision "$REVISION" \
  --unnorm-key "$UNNORM_KEY" \
  --rollout-seed 0 \
  --horizons 40 30 20 10 5 \
  --history-length 8 \
  "${SELECTION_ARGS[@]}"

echo "DONE mode=$MODE output=$OUTPUT @ $(date)"
