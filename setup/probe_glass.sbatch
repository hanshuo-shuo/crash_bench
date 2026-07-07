#!/bin/bash
# Self-report probe on Category 2 (glass): capture OpenVLA hidden states (glass / off-path /
# no-glass) + within-glass fit + cross-hazard transfer of the frozen wall probe.
#   cd ~/crash_bench/setup && sbatch probe_glass.sbatch
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --job-name=crashbench_probe_glass
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
nvidia-smi -L
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/crash_bench"

echo "== [1/2] capturing glass hidden states (glass / off-path / no-glass) =="
python scripts/probe_glass_capture.py

echo "== [2/2] within-glass probe + cross-hazard transfer + figure (CPU) =="
python scripts/probe_glass_analysis.py

echo "DONE @ $(date)"
