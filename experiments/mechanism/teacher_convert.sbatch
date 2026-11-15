#!/usr/bin/env bash
# Teacher data conversion (CPU; no API). Submitted only by launch_train.py after the export. teacher.py convert checks the
# export (every episode replayed with equal physics and re-rendered the stored policy input) and writes the episodes as a
# LeRobot dataset with the features of physical-intelligence/libero -> ROOT/lerobot, ROOT/DATASET.json.
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=cb_mech_teacher_convert
set -euo pipefail
ROOT="${CB_FD_ROOT:?}"
cd "$ROOT/source"
sha256sum --quiet -c "$ROOT/source.sha256"
export CB_CODE_COMMIT="$(cat "$ROOT/SOURCE_COMMIT")"
unset OPENROUTER_API_KEY
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/projects/p33100/siosio/huggingface_cache HF_DATASETS_CACHE=/projects/p33100/siosio/huggingface_cache/datasets
export HF_LEROBOT_HOME=/projects/p33100/siosio/huggingface_cache/lerobot HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
/projects/p33100/siosio/envs/openpi/bin/python -u "$ROOT/source/experiments/mechanism/teacher.py" convert "$ROOT"
