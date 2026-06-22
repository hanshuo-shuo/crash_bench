#!/bin/bash
# Run OpenVLA closed-loop on the OOD-but-not-crash control scenarios (README §14 obj. #1).
# Reuses run_pilot.py but on scenarios_control/ -> results/pilot_control.json, so the
# headline pilot crash-rate (results/pilot.json) is untouched.
#   cd ~/crash_bench/setup && sbatch run_ood_control.sbatch
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=crashbench_ood_ctrl
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
nvidia-smi -L
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/crash_bench"
python scripts/run_pilot.py \
  --scenarios scenarios_control \
  --checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --unnorm_key libero_spatial \
  --out results/pilot_control.json \
  --video_dir results/ood_control_videos
# analysis runs on CPU; do it here too so the contrast lands in the same job log
python scripts/phase1_ood_control_analysis.py \
  --treatment results/pilot.json --control results/pilot_control.json \
  --scn scenarios --ctrl_scn scenarios_control --out results/ood_control.json
echo "DONE @ $(date)"
