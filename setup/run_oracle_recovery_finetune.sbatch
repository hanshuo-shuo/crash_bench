#!/bin/bash
# Build the oracle-stop replay dataset, then LoRA fine-tune and merge OpenVLA.
# Submit through setup/submit_oracle_recovery_finetune.sh so provenance is exported.
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --job-name=oracle_recovery_ft
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
export PYTHONPATH="$HOME/crash_bench/third_party/openvla:$HOME/crash_bench${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME=/projects/p33100/siosio/torch_cache

cd "$HOME/crash_bench"
DATA_DIR="${CB_ORACLE_DATA_DIR:-results/oracle_recovery/dataset_v1}"
MODEL_DIR="${CB_RECOVERY_MODEL_DIR:-/projects/p33100/siosio/openvla_checkpoints/oracle_stop_recovery_v1}"
TRAIN_EXTRA=()
if [[ "${CB_SKIP_MERGE:-0}" == "1" ]]; then
  TRAIN_EXTRA+=(--skip-merge)
fi

python scripts/build_oracle_stop_dataset.py \
  --source results/safety_baselines_nonvlm.json \
  --out "$DATA_DIR" \
  --checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --overwrite

python scripts/finetune_openvla_oracle_stop.py \
  --base-checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --train-manifest "$DATA_DIR/train.jsonl" \
  --output-dir "$MODEL_DIR" \
  --max-steps "${CB_MAX_STEPS:-100}" \
  --batch-size "${CB_BATCH_SIZE:-8}" \
  --grad-accumulation-steps "${CB_GRAD_ACCUM:-2}" \
  --learning-rate "${CB_LEARNING_RATE:-0.0001}" \
  --lora-rank "${CB_LORA_RANK:-16}" \
  --lora-dropout "${CB_LORA_DROPOUT:-0.05}" \
  "${TRAIN_EXTRA[@]}"

if [[ "${CB_SKIP_MERGE:-0}" == "1" ]]; then
  test -f "$MODEL_DIR/adapter/adapter_config.json"
  echo "SMOKE_ADAPTER=$MODEL_DIR/adapter"
else
  test -f "$MODEL_DIR/merged/config.json"
  echo "MERGED_CHECKPOINT=$MODEL_DIR/merged"
fi
echo "DONE @ $(date)"
