#!/bin/bash
# Held-out evaluation for the oracle-stop recovery-finetuned OpenVLA checkpoint.
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --job-name=oracle_recovery_eval
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
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$HOME/crash_bench/third_party/openvla:$HOME/crash_bench${PYTHONPATH:+:$PYTHONPATH}"
export CB_OVERWRITE=1

cd "$HOME/crash_bench"
MODEL_DIR="${CB_RECOVERY_MODEL_DIR:-/projects/p33100/siosio/openvla_checkpoints/oracle_stop_recovery_v1}"
OUT_DIR="${CB_RECOVERY_EVAL_DIR:-results/oracle_recovery/eval_v1}"
HELDOUT=(
  env_collision__T1__libero_spatial_t0_wall_wide
  env_collision__T5__libero_spatial_t0_wall_d85
)
TRAIN=(
  env_collision__T5__libero_spatial_t0_wall_d62
  env_collision__T5__libero_spatial_t0_wall_d70
  env_collision__T5__libero_spatial_t0_wall_d78
)
test -f "$MODEL_DIR/merged/config.json"
mkdir -p "$OUT_DIR"

python scripts/safety_baseline_suite.py \
  --conditions vanilla \
  --scenario-ids "${HELDOUT[@]}" \
  --repeats 3 \
  --checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --video-dir "$OUT_DIR/videos_base_heldout" --video-reps 1 \
  --out "$OUT_DIR/base_heldout.json"

python scripts/safety_baseline_suite.py \
  --conditions recovery_finetuned \
  --scenario-ids "${HELDOUT[@]}" \
  --repeats 3 \
  --checkpoint "$MODEL_DIR/merged" \
  --video-dir "$OUT_DIR/videos_finetuned_heldout" --video-reps 1 \
  --out "$OUT_DIR/finetuned_heldout.json"

python scripts/safety_baseline_suite.py \
  --conditions recovery_finetuned \
  --scenario-ids "${TRAIN[@]}" \
  --repeats 3 \
  --checkpoint "$MODEL_DIR/merged" \
  --out "$OUT_DIR/finetuned_train.json"

echo "RESULT_DIR=$OUT_DIR"
echo "DONE @ $(date)"
