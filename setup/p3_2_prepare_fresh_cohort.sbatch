#!/bin/bash
# P3.2: author a new random-reset candidate pool; no Router is run here.
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=p3_2_fresh_prepare
#SBATCH --output=%x_%j.log

set -euo pipefail

RUN_ROOT="${CB_P3_2_RUN_ROOT:?set CB_P3_2_RUN_ROOT}"
REPO_ROOT="${CB_REPO_ROOT:-$HOME/crash_bench}"
COHORT_ROOT="$RUN_ROOT/fresh_cohort"
CHECKPOINT="${CB_BASE_CHECKPOINT:-openvla/openvla-7b-finetuned-libero-spatial}"
REVISION="${CB_BASE_CHECKPOINT_REVISION:-962318cec55ac10993ff0f5f43eda9a270b4c873}"
N_SOURCES="${CB_P3_2_RANDOM_SOURCES:-30}"
SOURCE_SEED="${CB_P3_2_SOURCE_SEED:-2026081901}"

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
[[ ! -e "$COHORT_ROOT" ]]
mkdir -p "$COHORT_ROOT"

SOURCE_ROOT="$COHORT_ROOT/source_traces"
python scripts/capture_glass_nominal_source_traces.py \
  --output "$SOURCE_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-revision "$REVISION" \
  --unnorm-key libero_spatial \
  --random-initializations "$N_SOURCES" \
  --random-initialization-seed "$SOURCE_SEED" \
  --rollout-seed 0 \
  --settle-steps 10 \
  --max-steps 220

SUCCESSFUL="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["successful_source_states"])' "$SOURCE_ROOT/source_traces.json")"
[[ "$SUCCESSFUL" -ge 8 ]]

python scripts/prepare_glass_recovery_placements.py \
  --output "$COHORT_ROOT/placements" \
  --source-trace-manifest "$SOURCE_ROOT/source_traces.json" \
  --checkpoint-revision "$REVISION" \
  --unnorm-key libero_spatial \
  --fresh-evaluation \
  --skip-fixed-action-screen \
  --geometry-profile controller_compatible_narrow \
  --min-target-clearance 0.12 \
  --control-offset 0.30 \
  --screen-horizon 20 \
  --screen-force-margin 10.0 \
  --screen-target-rest-threshold 0.005 \
  --max-proposals-per-candidate 12 \
  --train-placements 0 --validation-placements 0 \
  --heldout-placements "$SUCCESSFUL" \
  --train-accepted-targets 0 --validation-accepted-targets 0 \
  --heldout-accepted-targets 1 \
  --train-states 0 --validation-states 0 \
  --heldout-states "$SUCCESSFUL" \
  --source-state-reserve 0

echo "DONE P3.2 fresh prepare commit=$COMMIT successful=$SUCCESSFUL root=$COHORT_ROOT @ $(date)"
