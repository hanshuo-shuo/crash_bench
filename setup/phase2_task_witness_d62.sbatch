#!/bin/bash
# Historical low-wall d62 task-completion witness sweep, NO save. Never use it to
# lower or overwrite the tall-wall treatment in scenarios/.
# First pass: read per-body wall force to see if raising transit height + widening the detour
# lane clears the elbow, or whether we need the wrist pose term / a wall-height reduction.
#   cd ~/crash_bench/setup && sbatch phase2_task_witness_d62.sbatch
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --job-name=task_witness_d62
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
nvidia-smi -L
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/crash_bench"
python scripts/phase2_task_witness.py --scenarios 'scenarios_detour_lowwall/env_collision__T5__libero_spatial_t0_wall_d62__lowwall_detour_v1' --sweep
echo "DONE @ $(date)"
