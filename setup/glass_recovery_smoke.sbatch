#!/bin/bash
# E15/v2 learned-recovery smoke.  Submit only through the companion wrapper.
#
# train: consume accepted schema-v2 train/validation manifests and a frozen
# primary-protocol SHA.  evaluate: consume a frozen accepted cohort and the
# complete protocol whose checkpoint identity was sealed after training.
# Historical E14 artifacts are never inputs or output targets of this job.
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --job-name=glass_recovery_e15
#SBATCH --output=%x_%j.log

set -euo pipefail

if [[ "${CB_ENABLE_LEGACY_GLASS_RECOVERY:-0}" != "1" ]]; then
  echo "legacy E15 train/evaluate is disabled: current checkpoint does not justify Pilots D/F" >&2
  echo "set CB_ENABLE_LEGACY_GLASS_RECOVERY=1 only for an explicit provenance diagnostic" >&2
  exit 2
fi

require_env() {
  local name="$1"
  [[ -n "${!name:-}" ]] || { echo "required environment variable is empty: $name" >&2; exit 2; }
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || { echo "required input file does not exist: $path" >&2; exit 2; }
}

require_sha256() {
  local name="$1"
  local value="${!name:-}"
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || { echo "$name must be a lowercase SHA-256 digest" >&2; exit 2; }
}

require_env CB_GLASS_RECOVERY_STAGE
case "$CB_GLASS_RECOVERY_STAGE" in
  train|evaluate) ;;
  *) echo "CB_GLASS_RECOVERY_STAGE must be train or evaluate" >&2; exit 2 ;;
esac

echo "JOB $SLURM_JOB_ID stage=$CB_GLASS_RECOVERY_STAGE on $(hostname) @ $(date)"
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
BASE_REPO_ROOT="${CB_BASE_REPO_ROOT:-$HOME/crash_bench}"
export PYTHONPATH="$BASE_REPO_ROOT/third_party/openvla:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
"$GIT_BIN" diff --quiet && "$GIT_BIN" diff --cached --quiet \
  || { echo "tracked source differs from HEAD" >&2; exit 1; }
export CB_CODE_COMMIT="${CB_CODE_COMMIT:-$("$GIT_BIN" rev-parse HEAD)}"
[[ "$CB_CODE_COMMIT" == "$("$GIT_BIN" rev-parse HEAD)" ]] \
  || { echo "CB_CODE_COMMIT does not match checked-out HEAD" >&2; exit 1; }

require_env CB_GLASS_RECOVERY_RUN_ROOT
require_env CB_GLASS_RECOVERY_PRIMARY_PROTOCOL_SHA256
require_env CB_BASE_CHECKPOINT_REVISION
require_env CB_BASE_UNNORM_KEY
require_env CB_GLASS_RECOVERY_H
require_sha256 CB_GLASS_RECOVERY_PRIMARY_PROTOCOL_SHA256

RUN_ROOT="$CB_GLASS_RECOVERY_RUN_ROOT"
TRAIN_SEED="${CB_GLASS_RECOVERY_TRAIN_SEED:-17}"
CHECKPOINT_DIR="$RUN_ROOT/train_seed_${TRAIN_SEED}"

if [[ "$CB_GLASS_RECOVERY_STAGE" == "train" ]]; then
  require_env CB_GLASS_RECOVERY_TRAIN_MANIFEST
  require_env CB_GLASS_RECOVERY_VALIDATION_MANIFEST
  require_file "$CB_GLASS_RECOVERY_TRAIN_MANIFEST"
  require_file "$CB_GLASS_RECOVERY_VALIDATION_MANIFEST"
  mkdir -p "$RUN_ROOT"
  python scripts/train_glass_recovery.py \
    --train-manifest "$CB_GLASS_RECOVERY_TRAIN_MANIFEST" \
    --validation-manifest "$CB_GLASS_RECOVERY_VALIDATION_MANIFEST" \
    --output "$CHECKPOINT_DIR" \
    --base-resolved-revision "$CB_BASE_CHECKPOINT_REVISION" \
    --unnorm-key "$CB_BASE_UNNORM_KEY" \
    --protocol-sha256 "$CB_GLASS_RECOVERY_PRIMARY_PROTOCOL_SHA256" \
    --trigger-horizon "$CB_GLASS_RECOVERY_H" \
    --gating-horizon "$CB_GLASS_RECOVERY_H" \
    --max-steps "${CB_GLASS_RECOVERY_TRAIN_STEPS:-40}" \
    --batch-size "${CB_GLASS_RECOVERY_BATCH_SIZE:-32}" \
    --width "${CB_GLASS_RECOVERY_WIDTH:-128}" \
    --depth "${CB_GLASS_RECOVERY_DEPTH:-2}" \
    --eval-every "${CB_GLASS_RECOVERY_EVAL_EVERY:-20}" \
    --log-every "${CB_GLASS_RECOVERY_LOG_EVERY:-5}" \
    --seed "$TRAIN_SEED"
  test -f "$CHECKPOINT_DIR/glass_recovery.pt"
  test -f "$CHECKPOINT_DIR/training_summary.json"
  echo "CHECKPOINT=$CHECKPOINT_DIR/glass_recovery.pt"
  echo "NEXT=freeze checkpoint SHA, accepted evaluation cohort, and evaluation protocol before evaluate"
  echo "DONE @ $(date)"
  exit 0
fi

require_env CB_GLASS_RECOVERY_PLACEMENT_MANIFEST
require_env CB_GLASS_RECOVERY_TRAJECTORY_MANIFEST
require_env CB_GLASS_RECOVERY_EVALUATION_COHORT
require_env CB_GLASS_RECOVERY_PROTOCOL
require_env CB_GLASS_RECOVERY_EVALUATION_PROTOCOL_SHA256
require_env CB_GLASS_RECOVERY_CHECKPOINT
require_env CB_BASE_CHECKPOINT
require_sha256 CB_GLASS_RECOVERY_EVALUATION_PROTOCOL_SHA256
for input in \
  "$CB_GLASS_RECOVERY_PLACEMENT_MANIFEST" \
  "$CB_GLASS_RECOVERY_TRAJECTORY_MANIFEST" \
  "$CB_GLASS_RECOVERY_EVALUATION_COHORT" \
  "$CB_GLASS_RECOVERY_PROTOCOL" \
  "$CB_GLASS_RECOVERY_CHECKPOINT"
do
  require_file "$input"
done

# Fail before OpenVLA/LIBERO initialization if the exported protocol identities
# do not equal the canonical hashes embedded in the supplied protocol file.
python -c 'import json, os; from crashbench.glass_recovery_data import canonical_sha256; p=json.load(open(os.environ["CB_GLASS_RECOVERY_PROTOCOL"])); primary=canonical_sha256(p["primary_protocol"]); evaluation=canonical_sha256(p["evaluation_protocol"]); assert p["primary_protocol_sha256"] == primary == os.environ["CB_GLASS_RECOVERY_PRIMARY_PROTOCOL_SHA256"], "primary protocol SHA mismatch"; assert p["evaluation_protocol_sha256"] == evaluation == os.environ["CB_GLASS_RECOVERY_EVALUATION_PROTOCOL_SHA256"], "evaluation protocol SHA mismatch"'

EVAL_LABEL="${CB_GLASS_RECOVERY_EVAL_LABEL:-full}"
[[ "$EVAL_LABEL" =~ ^[a-z0-9_]+$ ]] \
  || { echo "CB_GLASS_RECOVERY_EVAL_LABEL must use lowercase letters, digits, or underscore" >&2; exit 2; }
EVAL_OUT="$RUN_ROOT/eval_${EVAL_LABEL}_seed_${TRAIN_SEED}.json"
SOURCE_ANALYSIS="$RUN_ROOT/analysis_${EVAL_LABEL}_seed_${TRAIN_SEED}_source_to_task.json"
ANCHOR_ANALYSIS="$RUN_ROOT/analysis_${EVAL_LABEL}_seed_${TRAIN_SEED}_exact_anchor.json"
EVAL_MODE_ARGS=()
EVAL_REGIME_ARGS=()
EVAL_CONDITION_ARGS=()
if [[ -n "${CB_GLASS_RECOVERY_EVAL_MODES:-}" ]]; then
  read -r -a EVAL_MODES <<< "$CB_GLASS_RECOVERY_EVAL_MODES"
  EVAL_MODE_ARGS=(--mode "${EVAL_MODES[@]}")
fi
if [[ -n "${CB_GLASS_RECOVERY_EVAL_REGIMES:-}" ]]; then
  read -r -a EVAL_REGIMES <<< "$CB_GLASS_RECOVERY_EVAL_REGIMES"
  EVAL_REGIME_ARGS=(--regime "${EVAL_REGIMES[@]}")
fi
if [[ -n "${CB_GLASS_RECOVERY_EVAL_CONDITIONS:-}" ]]; then
  read -r -a EVAL_CONDITIONS <<< "$CB_GLASS_RECOVERY_EVAL_CONDITIONS"
  EVAL_CONDITION_ARGS=(--conditions "${EVAL_CONDITIONS[@]}")
fi

python scripts/eval_glass_recovery.py \
  --placement-manifest "$CB_GLASS_RECOVERY_PLACEMENT_MANIFEST" \
  --trajectory-manifest "$CB_GLASS_RECOVERY_TRAJECTORY_MANIFEST" \
  --evaluation-cohort "$CB_GLASS_RECOVERY_EVALUATION_COHORT" \
  --protocol "$CB_GLASS_RECOVERY_PROTOCOL" \
  --checkpoint "$CB_GLASS_RECOVERY_CHECKPOINT" \
  --base-checkpoint "$CB_BASE_CHECKPOINT" \
  --base-checkpoint-revision "$CB_BASE_CHECKPOINT_REVISION" \
  --unnorm-key "$CB_BASE_UNNORM_KEY" \
  "${EVAL_MODE_ARGS[@]}" \
  "${EVAL_REGIME_ARGS[@]}" \
  "${EVAL_CONDITION_ARGS[@]}" \
  --out "$EVAL_OUT"

if [[ "${CB_GLASS_RECOVERY_SKIP_ANALYSIS:-0}" != 1 ]]; then
  python scripts/analyze_glass_recovery_eval.py \
    --evaluation "$EVAL_OUT" \
    --mode source_to_task \
    --out "$SOURCE_ANALYSIS"
  python scripts/analyze_glass_recovery_eval.py \
    --evaluation "$EVAL_OUT" \
    --mode exact_anchor \
    --out "$ANCHOR_ANALYSIS"
fi

test -f "$EVAL_OUT"
if [[ "${CB_GLASS_RECOVERY_SKIP_ANALYSIS:-0}" != 1 ]]; then
  test -f "$SOURCE_ANALYSIS"
  test -f "$ANCHOR_ANALYSIS"
fi
echo "RUN_ROOT=$RUN_ROOT"
echo "RESULT=$EVAL_OUT"
echo "DONE @ $(date)"
