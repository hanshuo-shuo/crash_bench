#!/bin/bash
# D0 glass detector capture over frozen exposed/fresh E15 placement designs.
# Submit through setup/submit_glass_detector_d0.sh.
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --job-name=glass_detector_d0
#SBATCH --output=%x_%j.log

set -euo pipefail

MODE="${CB_D0_MODE:?set CB_D0_MODE to smoke or full}"
OUTPUT="${CB_D0_OUTPUT:?set CB_D0_OUTPUT to a new ignored result directory}"
case "$MODE" in
  smoke|full) ;;
  *) echo "CB_D0_MODE must be smoke or full" >&2; exit 2 ;;
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

PLACEMENT_ARGS=()
if [[ "$MODE" == "smoke" ]]; then
  PLACEMENT_ARGS=(--max-placements 1)
fi

python scripts/capture_glass_detector_placements.py \
  --exposed-placements "$EXPOSED_PLACEMENTS" \
  --development-placements "$DEVELOPMENT_PLACEMENTS" \
  --output "$OUTPUT" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-revision "$REVISION" \
  --unnorm-key "$UNNORM_KEY" \
  --rollout-seeds 0 \
  --settle-steps 10 \
  --max-steps 220 \
  "${PLACEMENT_ARGS[@]}"

if [[ "$MODE" == "full" ]]; then
  python scripts/fit_glass_detector.py \
    --capture "$OUTPUT/hidden.npz" \
    --metadata "$OUTPUT/meta.json" \
    --split-manifest "$OUTPUT/source_split.json" \
    --output "$OUTPUT/detector"
fi

echo "DONE mode=$MODE output=$OUTPUT @ $(date)"
