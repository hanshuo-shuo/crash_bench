#!/bin/bash
# Regenerate d62 witness with grasp-offset-corrected (decisive, replay-robust) place, then save.
#   cd ~/crash_bench/setup && sbatch phase2_regen_d62.sbatch
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=regen_d62
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
nvidia-smi -L
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/crash_bench"
D='scenarios/env_collision__T5__libero_spatial_t0_wall_d62'
echo "=== diag (check final bowl-plate dist with corrected place) ==="
python scripts/phase2_task_witness.py --scenarios "$D" --diag
echo "=== save (regenerate + persist witness at h=0.12) ==="
python scripts/phase2_task_witness.py --scenarios "$D" --lower 0.12 --sweep --save
echo "DONE @ $(date)"
