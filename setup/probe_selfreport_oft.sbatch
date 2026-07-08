#!/bin/bash
# Path 3 probe fit — OpenVLA-OFT: capture the LM hidden state (wall/no-wall/off-path) + train
# the linear crash-imminence probe. Mirrors probe_selfreport.sbatch but on the OFT env @ p33100.
#   cd ~/crash_bench/setup && sbatch probe_selfreport_oft.sbatch
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --job-name=crashbench_selfreport_oft
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
nvidia-smi -L
module purge
module load mamba/24.3.0
source activate "/projects/p33100/siosio/envs/openvla-oft"      # OFT env @ p33100

export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH="$HOME/crash_bench"
export OPENVLA_OFT_ROOT="/projects/p33100/siosio/third_party/openvla-oft"
export CRASHBENCH_OPENVLA_ROOT="$OPENVLA_OFT_ROOT"
export HF_HOME="/projects/p33100/siosio/huggingface_cache"
cd "$HOME/crash_bench"

echo "== [1/2] capturing OFT hidden states (wall vs no-wall vs off-path) =="
python scripts/probe_selfreport_capture.py \
  --policy openvla-oft \
  --out results/selfreport_oft \
  --checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --unnorm_key libero_spatial

echo "== [2/2] training the linear probe + figures =="
python scripts/probe_selfreport_analysis.py results/selfreport_oft

echo "DONE @ $(date)"
