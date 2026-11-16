#!/usr/bin/env bash
# Script baseline of the other mid-task instructions (CPU; no policy, no rendering, no API). Submitted only by
# launch_train.py. In the LIBERO container, CB_TEACHER_WORKERS processes run the scripts of protocol section 8
# (verifier_dataset/hil_teacher.py, scripted) at the switch steps of the dev and test prefixes without cameras; then
# teacher.py baseline reads the runs -> ROOT/BASELINE.json.
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=cb_mech_teacher_baseline
set -euo pipefail
head_commit() { local h; h="$(cat "$1/.git/HEAD")"; case "$h" in "ref: "*) cat "$1/.git/${h#ref: }";; *) echo "$h";; esac; }
ROOT="${CB_FD_ROOT:?}"
SOURCE="$ROOT/source"
cd "$SOURCE"
sha256sum --quiet -c "$ROOT/source.sha256"
export CB_CODE_COMMIT="$(cat "$ROOT/SOURCE_COMMIT")"
export CB_ASSETS=/projects/p33100/siosio/crashbench_safelibero
export CB_UPSTREAM="$HOME/crash_bench/third_party/vlsa-aegis"
export CB_LIBERO="$HOME/crash_bench/third_party/LIBERO"
test "$(cat "$CB_UPSTREAM/.git/HEAD")" = 2457feed5968ae803926e178c8ce8243b9ecdcf9
test "$(head_commit "$CB_LIBERO")" = 8f1084e3132a39270c3a13ebe37270a43ece2a01
export CB_POLICY_CONFIG=pi05_libero
unset OPENROUTER_API_KEY
module purge
module load singularityce/4.3.1-gcc-8.5.0
mkdir -p "$ROOT/logs"
for name in CB_ASSETS CB_UPSTREAM CB_LIBERO CB_CODE_COMMIT CB_POLICY_CONFIG; do export "SINGULARITYENV_$name=${!name}"; done
export SINGULARITYENV_SLURM_JOB_ID="$SLURM_JOB_ID"
export SINGULARITYENV_MUJOCO_GL=glx SINGULARITYENV_PYOPENGL_PLATFORM=glx
export SINGULARITYENV_PYTHONNOUSERSITE=1 SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
export SINGULARITYENV_OMP_NUM_THREADS=1 SINGULARITYENV_OPENBLAS_NUM_THREADS=1
export SINGULARITYENV_PYTHONPATH="$CB_UPSTREAM/openpi/packages/openpi-client/src"
PIDS=()
for WORKER in $(seq 0 $(( ${CB_TEACHER_WORKERS:?} - 1 ))); do
  singularity exec --bind "/projects,/home,$ROOT:$ROOT:rw,$SOURCE:$SOURCE:ro" \
   "$CB_ASSETS/containers/aegis-py38.sif" "$CB_ASSETS/envs/aegis_sim/bin/python" \
   -u -m verifier_dataset.hil_teacher "$ROOT" --shard 0 --shards 1 --worker "$WORKER" \
   --workers "$CB_TEACHER_WORKERS" > "$ROOT/logs/baseline_${WORKER}.log" 2>&1 &
  PIDS+=($!)
done
STATUS=0
for PID in "${PIDS[@]}"; do wait "$PID" || STATUS=1; done
test "$STATUS" = 0
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
/projects/p33100/siosio/envs/openpi/bin/python -u "$SOURCE/experiments/mechanism/teacher.py" baseline "$ROOT"
