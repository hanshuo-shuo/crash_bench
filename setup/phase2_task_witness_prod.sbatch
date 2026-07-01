#!/bin/bash
# Lock in d62 (validated h=0.12) + discover witness heights for the other T5 walls.
#   cd ~/crash_bench/setup && sbatch phase2_task_witness_prod.sbatch
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --time=01:15:00
#SBATCH --job-name=task_witness_prod
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
nvidia-smi -L
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/crash_bench"

echo "=========== (1) LOCK d62 @ h=0.12 (save witness + lowered wall) ==========="
python scripts/phase2_task_witness.py \
  --scenarios 'scenarios/env_collision__T5__libero_spatial_t0_wall_d62' \
  --lower 0.12 --sweep --save

echo "=========== (2) DISCOVER heights for d70/d78/d85 (auto-lower, NO save) ==========="
for w in d70 d78 d85; do
  echo "----- $w -----"
  python scripts/phase2_task_witness.py \
    --scenarios "scenarios/env_collision__T5__libero_spatial_t0_wall_${w}" \
    --auto-lower --sweep
done
echo "DONE @ $(date)"
