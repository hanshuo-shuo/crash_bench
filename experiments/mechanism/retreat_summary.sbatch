#!/usr/bin/env bash
# Retreat and redo, summary and gate (CPU). Submitted only by launch_retreat.py.
# Summary (after a smoke or the formal batch): first the cross-environment goal check in the LIBERO container (B's goal
# evaluated in A's environment against B's own environment at the final state, hil_suite_run.state_check, on an evenly spaced
# sample of CB_STATE_CHECK runs; a mismatch writes passed: false and the tool check fails), then the tool check, readings and
# tables -> ROOT/SUMMARY.json in the openpi environment.
# Gate (CB_RETREAT_GATE_SMOKE set, before a formal batch): this root's code summarizes the finished smoke again into
# ROOT/gate and fails unless it reads tool_ok with the committed script constants, which cancels the formal batch through its
# afterok dependency. The smoke's own files are not rewritten.
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=cb_mech_retreat_summary
set -euo pipefail
head_commit() { local h; h="$(cat "$1/.git/HEAD")"; case "$h" in "ref: "*) cat "$1/.git/${h#ref: }";; *) echo "$h";; esac; }
ROOT="${CB_FD_ROOT:?}"
cd "$ROOT/source"
sha256sum --quiet -c "$ROOT/source.sha256"
export CB_CODE_COMMIT="$(cat "$ROOT/SOURCE_COMMIT")"
unset OPENROUTER_API_KEY
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
if [ -n "${CB_RETREAT_GATE_SMOKE:-}" ]; then
  /projects/p33100/siosio/envs/openpi/bin/python -u "$ROOT/source/experiments/mechanism/retreat.py" summarize "$CB_RETREAT_GATE_SMOKE" \
    --out "$ROOT/gate" --require-tool-ok
  exit 0
fi
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
 'import sys; from pathlib import Path; from verifier_dataset.hil_suite_run import state_check; state_check(Path(sys.argv[1]), limit=int(sys.argv[2]))' \
 "$ROOT" "${CB_STATE_CHECK:?}" || echo "state check did not pass; the summary reads STATE_CHECK.json and fails the tool check"
/projects/p33100/siosio/envs/openpi/bin/python -u "$ROOT/source/experiments/mechanism/retreat.py" summarize "$ROOT"
