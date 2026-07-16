#!/bin/bash
# M1 prerequisite: clean nominal gate for LIBERO-Spatial tasks t1--t9.
# Only tasks above the nominal gate should be used for cross-task corridor authoring.
#   cd ~/crash_bench && sbatch setup/m1_nominal_gate.sbatch
# Override the candidate set, for example:
#   sbatch --export=ALL,TASK_IDS=1,2,3 setup/m1_nominal_gate.sbatch
#
# This job does not create hazard scenarios yet; it only decides which tasks are worth
# spending the later builder/evaluation budget on.
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=crashbench_m1_gate
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
nvidia-smi -L

module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/crash_bench"

TASK_IDS="${TASK_IDS:-1,2,3,4,5,6,7,8,9}"
REPEATS="${REPEATS:-5}"
python scripts/nominal_task_gate.py \
  --suite libero_spatial \
  --task_ids "$TASK_IDS" \
  --repeats "$REPEATS" \
  --fail_below 0.60 \
  --out results/m1_nominal_gate.json

echo "DONE @ $(date)"
