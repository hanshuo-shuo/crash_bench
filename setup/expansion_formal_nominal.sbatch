#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --job-name=cb_exp_d3_formal
#SBATCH --output=%x_%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/crash_bench}"
: "${CB_EXPANSION_RUN_ID:?wrapper must export CB_EXPANSION_RUN_ID}"
: "${SLURM_ARRAY_TASK_ID:?must submit as array 0-5}"
module purge
module load mamba/24.3.0
source /projects/p33100/siosio/envs/openpi/bin/activate
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH="$HOME/crash_bench"
export OPENPI_DATA_HOME=/projects/p33100/siosio/openpi_assets
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
output="results/expansion/d3_formal_nominal/${CB_EXPANSION_RUN_ID}/shard_${SLURM_ARRAY_TASK_ID}.json"
python scripts/expansion/run_formal_nominal.py \
  --source-plan results/expansion/d3_sources/formal_nominal_plan_a42e34a64bf2e0f8200e3171f80bd646e59ffc442edbcd09fa62fe376029e667.json \
  --exposure-registry results/expansion/governance/exposure_registry.json \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --output "$output"
