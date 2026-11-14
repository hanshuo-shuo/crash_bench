#!/usr/bin/env bash
# Retreat and redo, dry run of the scripted segment (CPU; no policy, no rendering, no API). Submitted only by
# launch_retreat.py dry. In the LIBERO container: every unit of ROOT/PLAN.json under every named setting replays its census
# prefix to the switch step in a headless environment (physics hash checked) and runs the script -> ROOT/DRY.json.
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --job-name=cb_mech_retreat_dry
set -euo pipefail
head_commit() { local h; h="$(cat "$1/.git/HEAD")"; case "$h" in "ref: "*) cat "$1/.git/${h#ref: }";; *) echo "$h";; esac; }
ROOT="${CB_FD_ROOT:?}"
cd "$ROOT/source"
sha256sum --quiet -c "$ROOT/source.sha256"
export CB_CODE_COMMIT="$(cat "$ROOT/SOURCE_COMMIT")"
unset OPENROUTER_API_KEY
module purge
module load singularityce/4.3.1-gcc-8.5.0
test "$(head_commit "$HOME/crash_bench/third_party/LIBERO")" = 8f1084e3132a39270c3a13ebe37270a43ece2a01
export SINGULARITYENV_CB_CODE_COMMIT="$CB_CODE_COMMIT" SINGULARITYENV_CB_POLICY_CONFIG=pi05_libero
export SINGULARITYENV_SLURM_JOB_ID="$SLURM_JOB_ID"
export SINGULARITYENV_PYTHONNOUSERSITE=1 SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
export SINGULARITYENV_CB_LIBERO="$HOME/crash_bench/third_party/LIBERO"
export SINGULARITYENV_MUJOCO_GL=glx SINGULARITYENV_PYOPENGL_PLATFORM=glx
export SINGULARITYENV_PYTHONPATH="$HOME/crash_bench/third_party/vlsa-aegis/openpi/packages/openpi-client/src"
singularity exec --bind "/projects,/home,$ROOT:$ROOT:rw,$ROOT/source:$ROOT/source:ro" \
 /projects/p33100/siosio/crashbench_safelibero/containers/aegis-py38.sif \
 /projects/p33100/siosio/crashbench_safelibero/envs/aegis_sim/bin/python -u -c \
 'import sys; from pathlib import Path; from verifier_dataset.hil_suite_run import retreat_dry; retreat_dry(Path(sys.argv[1]))' "$ROOT"
