#!/bin/bash
# Path 3 probe fit — π0 (openpi): capture the VLM (PaliGemma) final-layer prefix hidden state
# (wall/no-wall/off-path) + train the linear crash-imminence probe. openpi uv venv @ p33100,
# torch-free π0 path (no OPENVLA_ROOT). Mirrors run_pilot_openpi.sbatch's env setup.
#   cd ~/crash_bench/setup && sbatch probe_selfreport_pi0.sbatch
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --job-name=crashbench_selfreport_pi0
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
nvidia-smi -L

P33100="/projects/p33100/siosio"
source "$P33100/envs/openpi/bin/activate"          # uv venv @ p33100
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH="$HOME/crash_bench"
export HF_HOME="$P33100/huggingface_cache"
export OPENPI_DATA_HOME="$P33100/openpi_assets"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
cd "$HOME/crash_bench"

echo "== [1/2] capturing π0 VLM hidden states (wall vs no-wall vs off-path) =="
python scripts/probe_selfreport_capture.py \
  --policy pi0 \
  --out results/selfreport_pi0 \
  --checkpoint gs://openpi-assets/checkpoints/pi0_libero

echo "== [2/2] training the linear probe + figures =="
python scripts/probe_selfreport_analysis.py results/selfreport_pi0

echo "DONE @ $(date)"
