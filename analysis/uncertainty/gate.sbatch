#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@480
#SBATCH --job-name=cb_uncertainty_gate
set -euo pipefail
ROOT="${CB_UNCERTAINTY_ROOT:?}"
PARENT="$(dirname "$(dirname "$ROOT")")"
SOURCE="$PARENT/source"
cd "$SOURCE"
export CB_ALLOCATION_START="$(date +%s)"
export CB_CODE_COMMIT="$(cat "$PARENT/SOURCE_COMMIT")"
export CB_ASSETS=/projects/p33100/siosio/crashbench_safelibero
export CB_UPSTREAM="$(/projects/p33100/siosio/envs/openpi/bin/python -c 'import json,os; print(json.load(open(os.environ["CB_UNCERTAINTY_ROOT"]+"/plan.json"))["upstream_root"])')"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
/projects/p33100/siosio/envs/openpi/bin/python "$SOURCE/analysis/uncertainty/verify_inputs.py" "$ROOT"
/projects/p33100/siosio/envs/openpi/bin/python "$SOURCE/analysis/uncertainty/gate_threshold.py" "$ROOT"
module purge
module load singularityce/4.3.1-gcc-8.5.0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HOME="$CB_ASSETS/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENPI_DATA_HOME="$CB_ASSETS/openpi_assets"
export PYTHONPATH="$CB_UPSTREAM/openpi/src:$CB_UPSTREAM/openpi/packages/openpi-client/src"
PORT=$((20000 + SLURM_JOB_ID % 30000))
SERVER_PID="";MONITOR_PID=""
trap 'for pid in "$SERVER_PID" "$MONITOR_PID"; do if [ -n "$pid" ]; then kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; fi; done' EXIT
trap 'touch "$ROOT/DRAIN_SIGNAL"' USR1
/projects/p33100/siosio/envs/openpi/bin/python - "$ROOT" <<'PY' > "$ROOT/gpu_monitor.log" 2>&1 &
import os,subprocess,sys,time
from pathlib import Path
with (Path(sys.argv[1])/'gpu_memory.csv').open('w') as f:
 while True:
  r=subprocess.run(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=timestamp,name,memory.total,memory.used','--format=csv,noheader,nounits'],text=True,capture_output=True)
  if r.returncode:raise RuntimeError(r.stderr)
  f.write(r.stdout);f.flush();time.sleep(2)
PY
MONITOR_PID=$!
/projects/p33100/siosio/envs/openpi/bin/python "$SOURCE/analysis/uncertainty/serve.py" \
 --root "$ROOT" --port "$PORT" --checkpoint "$CB_ASSETS/openpi_assets/openpi-assets/checkpoints/pi05_libero" > "$ROOT/policy.log" 2>&1 &
SERVER_PID=$!
/projects/p33100/siosio/envs/openpi/bin/python - "$PORT" "$SERVER_PID" <<'PY'
import os,socket,sys,time
for _ in range(180):
 try:
  with socket.create_connection(('127.0.0.1',int(sys.argv[1])),timeout=1):break
 except OSError:
  os.kill(int(sys.argv[2]),0);time.sleep(2)
else:raise RuntimeError('Policy readiness timeout')
PY
for name in CB_ASSETS CB_UPSTREAM CB_CODE_COMMIT CB_ALLOCATION_START; do export "SINGULARITYENV_$name=${!name}"; done
export SINGULARITYENV_MUJOCO_GL=egl SINGULARITYENV_PYOPENGL_PLATFORM=egl
export SINGULARITYENV_HF_HOME="$HF_HOME" SINGULARITYENV_TRANSFORMERS_CACHE="$HF_HOME/hub"
export SINGULARITYENV_HF_HUB_OFFLINE=1 SINGULARITYENV_TRANSFORMERS_OFFLINE=1
export SINGULARITYENV_PYTHONNOUSERSITE=1 SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
export SINGULARITYENV_PYTHONPATH="$CB_UPSTREAM/safelibero:$CB_UPSTREAM/openpi/packages/openpi-client/src"
singularity exec --nv --bind "/projects,/home,$PARENT:$PARENT:rw,$SOURCE:$SOURCE:ro,$CB_UPSTREAM:$CB_UPSTREAM:ro" \
 "$CB_ASSETS/containers/aegis-py38.sif" "$CB_ASSETS/envs/aegis_sim/bin/python" \
 -u "$SOURCE/analysis/uncertainty/gate.py" "$ROOT" --port "$PORT" > "$ROOT/evaluation.log" 2>&1
/projects/p33100/siosio/envs/openpi/bin/python "$SOURCE/analysis/uncertainty/gate_export.py" "$ROOT"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv > "$ROOT/gpu_final.csv"
