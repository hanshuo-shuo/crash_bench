#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=cb_exp_d5_collect
#SBATCH --output=%x_%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/crash_bench}"
: "${CB_EXPANSION_RUN_ID:?wrapper must export CB_EXPANSION_RUN_ID}"
: "${SLURM_ARRAY_TASK_ID:?must submit as array 0-47}"
module purge
module load mamba/24.3.0
source /projects/p33100/siosio/envs/openpi/bin/activate
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH="$HOME/crash_bench"
export OPENPI_DATA_HOME=/projects/p33100/siosio/openpi_assets
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

root="results/expansion/d5_statewise/${CB_EXPANSION_RUN_ID}"
python scripts/expansion/collect_staleness_statewise.py \
  --formal-analysis results/expansion/d3_formal_nominal/05ddc6a3a7c3_acb3d2d28dc2_20260830T103120Z/formal_nominal_analysis.json \
  --split-manifest results/expansion/governance/split_manifest_v1_1.json \
  --assignment-index "$SLURM_ARRAY_TASK_ID" \
  --artifact-store "$root/artifact_store" \
  --output "$root/shards/source_${SLURM_ARRAY_TASK_ID}.json"
