#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=cb_hil_suite_summary
set -euo pipefail
head_commit() { local h; h="$(cat "$1/.git/HEAD")"; case "$h" in "ref: "*) cat "$1/.git/${h#ref: }";; *) echo "$h";; esac; }
ROOT="${CB_FD_ROOT:?}"
cd "$ROOT/source"
sha256sum --quiet -c "$ROOT/source.sha256"
module purge
module load singularityce/4.3.1-gcc-8.5.0
unset OPENROUTER_API_KEY
export SINGULARITYENV_CB_CODE_COMMIT="$(cat "$ROOT/SOURCE_COMMIT")"
export SINGULARITYENV_CB_POLICY_CONFIG="${CB_POLICY_CONFIG:?}"
export SINGULARITYENV_SLURM_JOB_ID="$SLURM_JOB_ID"
export SINGULARITYENV_PYTHONNOUSERSITE=1 SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
export SINGULARITYENV_CB_LIBERO="$HOME/crash_bench/third_party/LIBERO"
test "$(head_commit "$HOME/crash_bench/third_party/LIBERO")" = 8f1084e3132a39270c3a13ebe37270a43ece2a01
export SINGULARITYENV_MUJOCO_GL=glx SINGULARITYENV_PYOPENGL_PLATFORM=glx
export SINGULARITYENV_PYTHONPATH="$HOME/crash_bench/third_party/vlsa-aegis/openpi/packages/openpi-client/src"
singularity exec --bind "/projects,/home,$ROOT:$ROOT:rw,$ROOT/source:$ROOT/source:ro" \
 /projects/p33100/siosio/crashbench_safelibero/containers/aegis-py38.sif \
 /projects/p33100/siosio/crashbench_safelibero/envs/aegis_sim/bin/python -u -m verifier_dataset.hil_suite_run "$ROOT" "--${CB_SUITE_SUMMARY:?}"
