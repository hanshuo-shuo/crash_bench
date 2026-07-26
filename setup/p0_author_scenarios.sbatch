#!/bin/bash
# P0 authoring stage B: build two-task train/calibration/heldout corridor scenarios.
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=crashbench_p0_author
#SBATCH --output=%x_%j.log

set -euo pipefail
: "${CB_CHECKPOINT_REVISION:?set the exact immutable checkpoint revision}"
: "${CB_GATE_OUT:?set the completed P0 gate result path}"
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/crash_bench"
python scripts/p0_author_scenarios.py \
  --gate "$CB_GATE_OUT" \
  --checkpoint-revision "$CB_CHECKPOINT_REVISION" \
  --base-seed "${CB_BASE_SEED:-20260726}" \
  --geometry-seed "${CB_GEOMETRY_SEED:-20261726}" \
  --scenario-root "${CB_SCENARIO_ROOT:-scenarios_p0}" \
  --config-out "${CB_CONFIG_OUT:-configs/p0_core.json}" \
  --report-out "${CB_AUTHOR_REPORT:-results/p0_runs/p0_authoring}"

