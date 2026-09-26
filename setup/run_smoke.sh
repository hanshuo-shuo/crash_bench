#!/usr/bin/env bash
set -euo pipefail
MODE="${1:?nominal, prepare or aegis}"
cd "${SLURM_SUBMIT_DIR:-$HOME/crash_bench}"
A=/projects/p33100/siosio/crashbench_safelibero
UP="$PWD/third_party/vlsa-aegis"
test -f "$A/VERIFIED.json"
test -f "$A/envs/aegis_sim/READY.json"
export CB_ASSETS="$A" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
module load git/2.37.2
export CB_CODE_COMMIT="$(git rev-parse HEAD)"
RUN="$A/runs/${MODE}_${CB_CODE_COMMIT:0:12}_${SLURM_JOB_ID}"
test ! -e "$RUN"
mkdir -p "$RUN"
module purge
module load singularityce/4.3.1-gcc-8.5.0
export HF_HOME="$A/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENPI_DATA_HOME="$A/openpi_assets"
export PYTHONPATH="$UP/openpi/src:$UP/openpi/packages/openpi-client/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PORT=$((20000 + SLURM_JOB_ID % 30000))
SERVER_PID=""
trap 'if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; fi' EXIT
if [ "$MODE" != prepare ]; then
  CKPT="$A/openpi_assets/openpi-assets/checkpoints/pi05_libero"
  test -d "$CKPT/params"
  /projects/p33100/siosio/envs/openpi/bin/python "$UP/scripts/serve_policy.py" --port "$PORT" policy:checkpoint --policy.config pi05_libero --policy.dir "$CKPT" > "$RUN/policy.log" 2>&1 &
  SERVER_PID=$!
  /projects/p33100/siosio/envs/openpi/bin/python - "$PORT" "$SERVER_PID" <<'PY'
import os,socket,sys,time
for _ in range(180):
 try:
  with socket.create_connection(('127.0.0.1',int(sys.argv[1])),timeout=1):break
 except OSError:
  os.kill(int(sys.argv[2]),0);time.sleep(2)
else:raise RuntimeError('Policy server did not become ready')
PY
fi
export SINGULARITYENV_CB_ASSETS="$A" SINGULARITYENV_CB_CODE_COMMIT="$CB_CODE_COMMIT"
export SINGULARITYENV_MUJOCO_GL=egl SINGULARITYENV_PYOPENGL_PLATFORM=egl
export SINGULARITYENV_TRANSFORMERS_CACHE="$A/huggingface/hub" SINGULARITYENV_HF_HUB_CACHE="$A/huggingface/hub"
export SINGULARITYENV_HF_HOME="$HF_HOME" SINGULARITYENV_HF_HUB_OFFLINE=1 SINGULARITYENV_TRANSFORMERS_OFFLINE=1
export SINGULARITYENV_PYTHONNOUSERSITE=1 SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
export SINGULARITYENV_PYTHONPATH="$UP/safelibero:$UP/openpi/packages/openpi-client/src"
singularity exec --nv --bind "/projects,/home,$PWD:$PWD" "$A/containers/aegis-py38.sif" \
 "$A/envs/aegis_sim/bin/python" scripts/run_safelibero.py --mode "$MODE" --port "$PORT" --output "$RUN/evaluation" 2>&1 | tee "$RUN/evaluation.log"
