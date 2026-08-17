#!/bin/bash
# Freeze the probabilistic router and author a source-disjoint fresh cohort.
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=cf_fresh_prepare
#SBATCH --output=%x_%j.log

set -euo pipefail

RUN_ROOT="${CB_CF_FRESH_ROOT:?set CB_CF_FRESH_ROOT}"
CAPTURE="${CB_CF_CAPTURE:-results/counterfactual_router/full_d4751330395e_20260814T152231Z}"
SOURCE_TRACES="${CB_CF_SOURCE_TRACES:-results/glass_recovery_v2/pilot_b_task0_20260811_r2/source_traces_task0/source_traces.json}"
REVISION="${CB_BASE_CHECKPOINT_REVISION:-962318cec55ac10993ff0f5f43eda9a270b4c873}"

module purge
module load mamba/24.3.0 git/2.47.0-gcc-12.4.0
source activate "$HOME/crash_bench/envs/openvla"

REPO_ROOT="${CB_REPO_ROOT:-$HOME/crash_bench}"
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
git diff --quiet
git diff --cached --quiet
COMMIT="$(git rev-parse HEAD)"
[[ "${CB_CODE_COMMIT:-$COMMIT}" == "$COMMIT" ]]
[[ ! -e "$RUN_ROOT" ]]
[[ -f "$CAPTURE/decision_features.npz" ]]
[[ -f "$SOURCE_TRACES" ]]
mkdir -p "$RUN_ROOT"

python scripts/train_counterfactual_outcome_router.py \
  --capture "$CAPTURE" \
  --output "$RUN_ROOT/router.json" \
  --pca-components 16 \
  --l2 0.01

# The 27 excluded indices are exactly those exposed by the two earlier
# placement manifests.  All remaining 20 successful traces are authored without
# replay screening; the online collector performs the only Base catastrophe/T-20
# eligibility pass and excludes an ineligible source as a whole matched group.
python scripts/prepare_glass_recovery_placements.py \
  --output "$RUN_ROOT/placements" \
  --source-trace-manifest "$SOURCE_TRACES" \
  --checkpoint-revision "$REVISION" \
  --unnorm-key libero_spatial \
  --fresh-evaluation \
  --skip-fixed-action-screen \
  --geometry-profile controller_compatible_narrow \
  --min-target-clearance 0.17 \
  --control-offset 0.30 \
  --screen-horizon 20 \
  --screen-force-margin 10.0 \
  --screen-target-rest-threshold 0.005 \
  --max-proposals-per-candidate 12 \
  --train-placements 0 --validation-placements 0 --heldout-placements 20 \
  --train-accepted-targets 0 --validation-accepted-targets 0 \
  --heldout-accepted-targets 1 \
  --train-states 0 --validation-states 0 --heldout-states 20 \
  --source-state-reserve 0 \
  --exclude-source-state-indices 1,2,3,5,6,9,11,12,14,16,17,18,19,20,21,22,23,24,26,28,29,34,35,38,40,43,47

echo "DONE prepare commit=$COMMIT root=$RUN_ROOT @ $(date)"
