#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=cb_feasibility
set -euo pipefail
ROOT="${CB_DIAGNOSTIC_ROOT:?}"
SOURCE="$ROOT/source"
STAGE="${CB_DIAGNOSTIC_STAGE:?}"
cd "$SOURCE"
sha256sum --quiet -c "$ROOT/source.sha256"
export CB_CODE_COMMIT="$(cat "$ROOT/SOURCE_COMMIT")"
export CB_ASSETS=/projects/p33100/siosio/crashbench_safelibero
export CB_UPSTREAM="$HOME/crash_bench/third_party/vlsa-aegis"
test -f "$CB_ASSETS/VERIFIED.json"
test -f "$CB_ASSETS/envs/aegis_sim/READY.json"
test "$(cat "$CB_UPSTREAM/.git/HEAD")" = 2457feed5968ae803926e178c8ce8243b9ecdcf9
module purge
module load singularityce/4.3.1-gcc-8.5.0
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$CB_ASSETS/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENPI_DATA_HOME="$CB_ASSETS/openpi_assets"
export PYTHONPATH="$CB_UPSTREAM/openpi/src:$CB_UPSTREAM/openpi/packages/openpi-client/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
BINDS="/projects,/home,$ROOT:$ROOT:rw,$SOURCE:$SOURCE:ro"
PORT=$((20000 + SLURM_JOB_ID % 30000))
SERVER_PID=""
trap 'if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; fi' EXIT
/projects/p33100/siosio/envs/openpi/bin/python "$SOURCE/experiments/feasibility/serve.py" --port "$PORT" --checkpoint "$CB_ASSETS/openpi_assets/openpi-assets/checkpoints/pi05_libero" > "$ROOT/policy.log" 2>&1 &
SERVER_PID=$!
/projects/p33100/siosio/envs/openpi/bin/python - "$PORT" "$SERVER_PID" <<'PY'
import os,socket,sys,time
for _ in range(180):
 try:
  with socket.create_connection(('127.0.0.1',int(sys.argv[1])),timeout=1):break
 except OSError:os.kill(int(sys.argv[2]),0);time.sleep(2)
else:raise RuntimeError('Policy server readiness timeout')
PY
for name in CB_ASSETS CB_UPSTREAM CB_CODE_COMMIT;do export "SINGULARITYENV_$name=${!name}";done
export SINGULARITYENV_MUJOCO_GL=egl SINGULARITYENV_PYOPENGL_PLATFORM=egl
export SINGULARITYENV_HF_HOME="$HF_HOME" SINGULARITYENV_HF_HUB_OFFLINE=1 SINGULARITYENV_TRANSFORMERS_OFFLINE=1
export SINGULARITYENV_TRANSFORMERS_CACHE="$CB_ASSETS/huggingface/hub" SINGULARITYENV_HF_HUB_CACHE="$CB_ASSETS/huggingface/hub"
export SINGULARITYENV_PYTHONNOUSERSITE=1 SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
export SINGULARITYENV_PYTHONPATH="$CB_UPSTREAM/safelibero:$CB_UPSTREAM/openpi/packages/openpi-client/src"
singularity exec --nv --bind "$BINDS" "$CB_ASSETS/containers/aegis-py38.sif" "$CB_ASSETS/envs/aegis_sim/bin/python" -u "$SOURCE/experiments/feasibility/execute.py" "$STAGE" "$ROOT" --port "$PORT" 2>&1 | tee "$ROOT/evaluation.log"
