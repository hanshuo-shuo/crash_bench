#!/bin/bash
# End-to-end first-stage smoke: author 100/20/40 placement specs, collect one
# accepted four-way pair per split, train, replay, and evaluate one held-out scene.
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --job-name=glass_recovery_smoke
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
nvidia-smi -L
module purge
module load mamba/24.3.0
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

RUN_ROOT="${CB_GLASS_RECOVERY_RUN_ROOT:-results/glass_recovery_v1/smoke_${SLURM_JOB_ID}}"
PLACEMENTS="$RUN_ROOT/placements"
DATASET="$RUN_ROOT/dataset"
CHECKPOINT="$RUN_ROOT/checkpoint"
EVAL_OUT="$RUN_ROOT/eval_heldout.json"
BASE_CHECKPOINT="${CB_BASE_CHECKPOINT:-/projects/p33100/siosio/huggingface_cache/hub/models--openvla--openvla-7b-finetuned-libero-spatial/snapshots/962318cec55ac10993ff0f5f43eda9a270b4c873}"

python scripts/prepare_glass_recovery_placements.py \
  --output "$PLACEMENTS" \
  --train-placements 100 \
  --validation-placements 20 \
  --heldout-placements 40 \
  --overwrite

python scripts/collect_glass_recovery_pairs.py \
  --placements "$PLACEMENTS/placements.json" \
  --output "$DATASET" \
  --checkpoint "$BASE_CHECKPOINT" \
  --max-train "${CB_SMOKE_TRAIN_PAIRS:-1}" \
  --max-validation "${CB_SMOKE_VALIDATION_PAIRS:-1}" \
  --max-heldout "${CB_SMOKE_HELDOUT_PAIRS:-1}"

python scripts/replay_glass_recovery_pair.py \
  --placements "$PLACEMENTS/placements.json" \
  --manifest "$DATASET/train.jsonl" \
  --branch nominal_catastrophe \
  --out "$RUN_ROOT/replay_nominal.json"

python scripts/train_glass_recovery.py \
  --train-manifest "$DATASET/train.jsonl" \
  --validation-manifest "$DATASET/validation.jsonl" \
  --output "$CHECKPOINT" \
  --max-steps "${CB_SMOKE_TRAIN_STEPS:-40}" \
  --batch-size 32 \
  --width 128 \
  --depth 2 \
  --eval-every 20 \
  --log-every 5 \
  --overwrite

python scripts/eval_glass_recovery.py \
  --placements "$PLACEMENTS/placements.json" \
  --checkpoint "$CHECKPOINT/glass_recovery.pt" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --split heldout \
  --max-placements 1 \
  --repeats 1 \
  --out "$EVAL_OUT" \
  --overwrite

test -f "$DATASET/collection_summary.json"
test -f "$CHECKPOINT/training_summary.json"
test -f "$EVAL_OUT"
echo "RUN_ROOT=$RUN_ROOT"
echo "RESULT=$EVAL_OUT"
echo "DONE @ $(date)"
