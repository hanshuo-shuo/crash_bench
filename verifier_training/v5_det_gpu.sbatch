#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=qgpu2014
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=cb_v5det_gpu
# Launched only by verifier_training.launch_v5 det with CB_V5_ROOT and CB_V5_ARGS; the AEGIS container and environment (Python 3.8).
set -euo pipefail
cd "$CB_V5_ROOT/source"
sha256sum --quiet -c "$CB_V5_ROOT/source.sha256"
A=/projects/p33100/siosio/crashbench_safelibero
sha256sum --quiet -c "$A/containers/aegis-py38.sif.sha256"
test -f "$A/envs/aegis_sim/READY.json"
unset OPENROUTER_API_KEY
module purge
module load singularityce/4.3.1-gcc-8.5.0
export SINGULARITYENV_CB_CODE_COMMIT="$(cat "$CB_V5_ROOT/SOURCE_COMMIT")"
export SINGULARITYENV_PYTHONNOUSERSITE=1 SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1 SINGULARITYENV_OMP_NUM_THREADS=4
export SINGULARITYENV_HF_HUB_OFFLINE=1 SINGULARITYENV_TRANSFORMERS_OFFLINE=1 SINGULARITYENV_TOKENIZERS_PARALLELISM=false
# CB_V5_ARGS is the script path relative to the source followed by plain path arguments (no spaces).
singularity exec --nv --bind "/projects,/home,/gpfs" "$A/containers/aegis-py38.sif" "$A/envs/aegis_sim/bin/python" -I -u $CB_V5_ARGS
