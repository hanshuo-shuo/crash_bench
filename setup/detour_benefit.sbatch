#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --job-name=detour_benefit
#SBATCH --output=%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/crash_bench}"
module purge
module load mamba/24.3.0 git/2.47.0-gcc-12.4.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/projects/p33100/siosio/huggingface_cache
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD/third_party/openvla:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export CB_CHECKPOINT_PATH="$HF_HOME/hub/models--openvla--openvla-7b-finetuned-libero-spatial/snapshots/962318cec55ac10993ff0f5f43eda9a270b4c873"
commit="$(git rev-parse HEAD)"
git diff --quiet
git diff --cached --quiet
run_name="${commit:0:12}_job${SLURM_JOB_ID}"
output="/projects/p33100/siosio/crashbench_detour_benefit/$run_name"
mkdir -p results/detour_benefit /projects/p33100/siosio/crashbench_detour_benefit
[[ ! -e "$output" ]]
ln -s "$output" "results/detour_benefit/$run_name"
python scripts/expansion/run_detour_benefit.py --output "$output"
python scripts/expansion/analyze_detour_benefit.py --run "$output"
