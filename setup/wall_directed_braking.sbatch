#!/bin/bash
# Geometry-aligned braking diagnostic: action projection, clearance delta, EEF velocity, TTC.
# Submit from a Quest login node with:
#   cd ~/crash_bench && bash setup/submit_wall_directed_braking.sh
# Optional overrides are forwarded by the submit wrapper, e.g.:
#   CB_REPEATS=5 bash setup/submit_wall_directed_braking.sh
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=crashbench_wallbrake
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
nvidia-smi -L
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/crash_bench"

echo "== wall-directed braking diagnostic =="
python scripts/wall_directed_braking.py \
  --policy "${CB_POLICY:-openvla}" \
  --repeats "${CB_REPEATS:-3}" \
  --near-steps "${CB_NEAR_STEPS:-2}" \
  --out "${CB_OUT:-results/wall_directed_braking}"

echo "DONE @ $(date)"
