#!/bin/bash
# Pilot B task-0 feasibility: source traces, H frontier, then 10/5/5 collection.
# Submit one stage at a time with setup/submit_glass_recovery_pilot_b.sh.
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --job-name=glass_pilot_b
#SBATCH --output=%x_%j.log

set -euo pipefail

STAGE="${CB_PILOT_B_STAGE:?set CB_PILOT_B_STAGE}"
RUN_ROOT="${CB_PILOT_B_ROOT:?set CB_PILOT_B_ROOT}"
CHECKPOINT="${CB_BASE_CHECKPOINT:-openvla/openvla-7b-finetuned-libero-spatial}"
REVISION="${CB_BASE_CHECKPOINT_REVISION:-962318cec55ac10993ff0f5f43eda9a270b4c873}"
UNNORM_KEY="${CB_BASE_UNNORM_KEY:-libero_spatial}"

echo "JOB $SLURM_JOB_ID stage=$STAGE on $(hostname) @ $(date)"
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

SOURCE_ROOT="$RUN_ROOT/source_traces_task0"
SOURCE_MANIFEST="$SOURCE_ROOT/source_traces.json"

case "$STAGE" in
  source_traces)
    python scripts/capture_glass_nominal_source_traces.py \
      --output "$SOURCE_ROOT" \
      --checkpoint "$CHECKPOINT" \
      --checkpoint-revision "$REVISION" \
      --unnorm-key "$UNNORM_KEY" \
      --rollout-seed 0 \
      --settle-steps 10 \
      --max-steps 220
    ;;

  frontier)
    test -f "$SOURCE_MANIFEST"
    FRONTIER_PLACEMENTS="$RUN_ROOT/frontier_placements"
    FRONTIER_ROOT="$RUN_ROOT/frontier_task0"
    python scripts/prepare_glass_recovery_placements.py \
      --output "$FRONTIER_PLACEMENTS" \
      --source-trace-manifest "$SOURCE_MANIFEST" \
      --checkpoint-revision "$REVISION" \
      --unnorm-key "$UNNORM_KEY" \
      --train-placements 10 --validation-placements 5 --heldout-placements 5 \
      --train-accepted-targets 2 --validation-accepted-targets 1 \
      --heldout-accepted-targets 1 \
      --train-states 4 --validation-states 2 --heldout-states 2
    python scripts/run_glass_avoidability_frontier.py \
      --placements "$FRONTIER_PLACEMENTS/placements.json" \
      --output-root "$FRONTIER_ROOT" \
      --checkpoint "$CHECKPOINT" \
      --checkpoint-revision "$REVISION" \
      --unnorm-key "$UNNORM_KEY" \
      --rollout-seed 0 \
      --execute
    ;;

  collect)
    test -f "$SOURCE_MANIFEST"
    HORIZON="${CB_PILOT_B_H:?set CB_PILOT_B_H after the frontier}"
    DATA_PLACEMENTS="$RUN_ROOT/placements_h${HORIZON}"
    DATA_ROOT="$RUN_ROOT/dataset_h${HORIZON}"
    python scripts/prepare_glass_recovery_placements.py \
      --output "$DATA_PLACEMENTS" \
      --source-trace-manifest "$SOURCE_MANIFEST" \
      --checkpoint-revision "$REVISION" \
      --unnorm-key "$UNNORM_KEY" \
      --train-placements 50 --validation-placements 25 --heldout-placements 25 \
      --train-accepted-targets 10 --validation-accepted-targets 5 \
      --heldout-accepted-targets 5 \
      --train-states "${CB_PILOT_B_TRAIN_STATES:-15}" \
      --validation-states "${CB_PILOT_B_VALIDATION_STATES:-8}" \
      --heldout-states "${CB_PILOT_B_HELDOUT_STATES:-8}"
    COLLECTION_STATUS=0
    python scripts/collect_glass_recovery_pairs.py \
      --placements "$DATA_PLACEMENTS/placements.json" \
      --output "$DATA_ROOT" \
      --checkpoint "$CHECKPOINT" \
      --checkpoint-revision "$REVISION" \
      --unnorm-key "$UNNORM_KEY" \
      --rollout-seed 0 \
      --max-train 10 --max-validation 5 --max-heldout 5 \
      --precrash-horizon "$HORIZON" || COLLECTION_STATUS=$?
    python scripts/summarize_glass_pilot_b.py \
      --frontier-summary "$RUN_ROOT/frontier_task0/frontier_summary.json" \
      --placements "$DATA_PLACEMENTS/placements.json" \
      --dataset "$DATA_ROOT" \
      --out "$RUN_ROOT/pilot_b_summary.json"
    if (( COLLECTION_STATUS != 0 )); then
      exit "$COLLECTION_STATUS"
    fi
    ;;

  *)
    echo "unknown Pilot B stage: $STAGE" >&2
    exit 2
    ;;
esac

echo "DONE stage=$STAGE root=$RUN_ROOT @ $(date)"
