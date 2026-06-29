#!/bin/bash
# Path 1-1a: probe-triggered safe-abort (baseline vs guarded on treatment + false-trigger checks).
#   cd ~/crash_bench/setup && sbatch phase3_intervention.sbatch
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=crashbench_intervention
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
nvidia-smi -L
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/crash_bench"

# Ensure the T=5 probe is dumped (CPU/numpy; idempotent). Needs results/selfreport/hidden.npz.
if [ ! -f results/selfreport/probe_T5.npz ]; then
  echo "== dumping probe_T5.npz =="
  python scripts/probe_selfreport_analysis.py
fi

echo "== running intervention experiment =="
python scripts/phase3_intervention.py

echo "== making figures =="
python scripts/phase3_make_figures.py

echo "DONE @ $(date)"
