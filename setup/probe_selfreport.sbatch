#!/bin/bash
# Self-report probe: capture OpenVLA hidden states (wall vs no-wall) + train the linear probe.
#   cd ~/crash_bench/setup && sbatch probe_selfreport.sbatch
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --job-name=crashbench_selfreport
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
nvidia-smi -L
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/crash_bench"

echo "== [1/2] capturing hidden states (wall vs no-wall vs off-path) =="
python scripts/probe_selfreport.py

echo "== [2/2] training the linear probe + figures =="
python scripts/probe_selfreport_analysis.py

echo "DONE @ $(date)"
