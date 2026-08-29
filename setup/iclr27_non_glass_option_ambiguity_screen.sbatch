#!/usr/bin/env bash
# One bounded OpenVLA exact-state screen for unstable final placement.
# Submit only through scripts/quest_sync.sh submit.
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --job-name=cb_nonglass_place
#SBATCH --output=%x_%j.log

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$HOME/crash_bench}"
module purge
module load git/2.47.0-gcc-12.4.0
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HOME=/projects/p33100/siosio/huggingface_cache
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export TORCH_HOME=/projects/p33100/siosio/torch_cache
export PYTHONPATH="$PWD/third_party/openvla:$PWD${PYTHONPATH:+:$PYTHONPATH}"

git_bin="$(command -v git)"
commit="$($git_bin rev-parse HEAD)"
[[ "${CB_CODE_COMMIT:-$commit}" == "$commit" ]]
"$git_bin" diff --quiet
"$git_bin" diff --cached --quiet

checkpoint_revision="${CB_BASE_CHECKPOINT_REVISION:-962318cec55ac10993ff0f5f43eda9a270b4c873}"
screen_config="${CB_NG_SCREEN_CONFIG:-configs/iclr27/non_glass_unstable_placement_screen_v1.json}"
output_parent="${CB_NG_SCREEN_OUTPUT_PARENT:-results/iclr27}"

printf 'commit=%s\njob=%s\naccount=p33100\npartition=gengpu\ncheckpoint_revision=%s\nconfig=%s\noutput_parent=%s\nstarted_utc=%s\n' \
  "$commit" "$SLURM_JOB_ID" "$checkpoint_revision" "$screen_config" \
  "$output_parent" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi -L

python scripts/iclr27/run_non_glass_option_ambiguity_screen.py \
  --freeze-only \
  --execute \
  --output "$output_parent" \
  --checkpoint-revision "$checkpoint_revision" \
  --screen-config "$screen_config"

printf 'DONE job=%s commit=%s finished_utc=%s\n' \
  "$SLURM_JOB_ID" "$commit" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
