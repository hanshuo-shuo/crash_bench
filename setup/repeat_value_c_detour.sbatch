#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --job-name=cb_value_c_detour
#SBATCH --output=%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/crash_bench}"
module purge
module load mamba/24.3.0 git/2.47.0-gcc-12.4.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/projects/p33100/siosio/huggingface_cache
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD/third_party/openvla:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export CB_CHECKPOINT_PATH="$HF_HOME/hub/models--openvla--openvla-7b-finetuned-libero-spatial/snapshots/962318cec55ac10993ff0f5f43eda9a270b4c873"
repeat_c_commit=$(git rev-parse HEAD)
test -z "$(git status --porcelain=v1 --untracked-files=all)"
repeat_c_name="${repeat_c_commit:0:12}_detour_${SLURM_JOB_ID}"
repeat_c_output="/projects/p33100/siosio/crashbench_repeat_value/$repeat_c_name"
mkdir -p results/repeat_value /projects/p33100/siosio/crashbench_repeat_value
test ! -e "$repeat_c_output"
ln -s "$repeat_c_output" "results/repeat_value/$repeat_c_name"
python scripts/expansion/run_repeat_value_c.py --panel detour --checkpoint-dir "$CB_CHECKPOINT_PATH" --output "$repeat_c_output"
python scripts/expansion/analyze_repeat_value_c.py --run-dir "$repeat_c_output"
