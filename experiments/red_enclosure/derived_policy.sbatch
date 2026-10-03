#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=cb_red_derived
set -euo pipefail
ROOT="${CB_RED_ROOT:?}"
SOURCE="$ROOT/source"
cd "$SOURCE"
sha256sum --quiet -c "$ROOT/source.sha256"
unset OPENROUTER_API_KEY
export CB_CODE_COMMIT="$(cat "$ROOT/SOURCE_COMMIT")"
export CB_ASSETS=/projects/p33100/siosio/crashbench_safelibero
export CB_UPSTREAM="$HOME/crash_bench/third_party/vlsa-aegis"
test "$(cat "$CB_UPSTREAM/.git/HEAD")" = 2457feed5968ae803926e178c8ce8243b9ecdcf9
module purge
module load singularityce/4.3.1-gcc-8.5.0
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HOME="$CB_ASSETS/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENPI_DATA_HOME="$CB_ASSETS/openpi_assets"
export PYTHONPATH="$CB_UPSTREAM/openpi/src:$CB_UPSTREAM/openpi/packages/openpi-client/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
for name in CB_ASSETS CB_UPSTREAM CB_CODE_COMMIT;do export "SINGULARITYENV_$name=${!name}";done
export SINGULARITYENV_MUJOCO_GL=egl SINGULARITYENV_PYOPENGL_PLATFORM=egl
export SINGULARITYENV_PYTHONNOUSERSITE=1 SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
export SINGULARITYENV_OMP_NUM_THREADS=1 SINGULARITYENV_OPENBLAS_NUM_THREADS=1
export SINGULARITYENV_PYTHONPATH="$CB_UPSTREAM/safelibero:$CB_UPSTREAM/openpi/packages/openpi-client/src"
BINDS="/projects,/home,$ROOT:$ROOT:rw,$SOURCE:$SOURCE:ro"
SIM="$CB_ASSETS/envs/aegis_sim/bin/python"
CONTAINER="$CB_ASSETS/containers/aegis-py38.sif"
# The separately completed CPU gate was hash-verified by derived_launch.py.
export SINGULARITYENV_CB_RED_HEADLESS=0
PORT=$((20000 + SLURM_JOB_ID % 30000))
SERVER_PID=""
trap 'if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; fi' EXIT
/projects/p33100/siosio/envs/openpi/bin/python "$SOURCE/experiments/feasibility/serve.py" --cache "$ROOT/policy_cache" --port "$PORT" --checkpoint "$CB_ASSETS/openpi_assets/openpi-assets/checkpoints/pi05_libero" > "$ROOT/policy.log" 2>&1 &
SERVER_PID=$!
/projects/p33100/siosio/envs/openpi/bin/python - "$PORT" "$SERVER_PID" <<'PY'
import os,socket,sys,time
for _ in range(180):
 try:
  with socket.create_connection(('127.0.0.1',int(sys.argv[1])),timeout=1):break
 except OSError:os.kill(int(sys.argv[2]),0);time.sleep(2)
else:raise RuntimeError('Policy server readiness timeout')
PY
singularity exec --nv --bind "$BINDS" "$CONTAINER" "$SIM" -u "$SOURCE/experiments/red_enclosure/run.py" "$ROOT" policy --port "$PORT"

singularity exec --nv --bind "$BINDS" "$CONTAINER" "$SIM" -u "$SOURCE/experiments/red_enclosure/render_saved.py" "$ROOT"
