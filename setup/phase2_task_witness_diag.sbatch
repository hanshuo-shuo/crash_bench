#!/bin/bash
# Historical low-wall d62 diagnostic. Never mutate the tall-wall treatment in scenarios/.
#   cd ~/crash_bench/setup && sbatch phase2_task_witness_diag.sbatch
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --job-name=task_witness_diag
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
nvidia-smi -L
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/crash_bench"
python scripts/phase2_task_witness.py --scenarios 'scenarios_detour_lowwall/env_collision__T5__libero_spatial_t0_wall_d62__lowwall_detour_v1' --diag
echo "DONE @ $(date)"
