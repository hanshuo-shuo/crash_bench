#!/bin/bash
# OOD-but-not-crash control v3: grow n + matched pairs + corridor characterization, one job.
# Needs results/nominal_traj.npy (scripts/probe_nominal_traj.py) to exist.
#   cd ~/crash_bench/setup && sbatch run_ood_control_v3.sbatch
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --job-name=ood_ctrl_v3
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
nvidia-smi -L
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/crash_bench"
python scripts/phase1_ood_control_v3.py
python scripts/phase1_ood_control_analysis.py \
  --treatment results/pilot.json --control results/pilot_control.json \
  --scn scenarios --ctrl_scn scenarios_control --out results/ood_control.json
echo "DONE @ $(date)"
