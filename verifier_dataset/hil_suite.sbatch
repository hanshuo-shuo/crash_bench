#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=qgpu2014
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=cb_hil_suite
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
# Compute nodes have no git: the launcher checked both checkouts are clean at submission; here the pinned commits are read directly.
test "$(head_commit "$CB_LIBERO")" = 8f1084e3132a39270c3a13ebe37270a43ece2a01
PHASE="${CB_SUITE_PHASE:?}"
export CB_POLICY_CONFIG="${CB_POLICY_CONFIG:?}"
unset OPENROUTER_API_KEY
module purge
module load singularityce/4.3.1-gcc-8.5.0
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HOME="$CB_ASSETS/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENPI_DATA_HOME="$CB_ASSETS/openpi_assets"
export PYTHONPATH="$CB_UPSTREAM/openpi/src:$CB_UPSTREAM/openpi/packages/openpi-client/src"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
PORT=$((20000 + SLURM_JOB_ID % 30000))
SERVER_PID=""
trap 'if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; fi' EXIT
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$ROOT/GPU_${PHASE}_${SHARD}.txt"
if [ "$CB_POLICY_CONFIG" = openvla_oft ]; then
 OFT=/projects/p33100/siosio/third_party/openvla-oft
 test "$(head_commit "$OFT")" = e4287e94541f459edc4feabc4e181f537cd569a8
 /projects/p33100/siosio/envs/openvla-oft/bin/python "$SOURCE/experiments/feasibility/serve_oft.py" --port "$PORT" --oft-root "$OFT" \
  --openpi "$CB_UPSTREAM/openpi" --checkpoint "${CB_POLICY_CHECKPOINT:?}" > "$ROOT/policy_${PHASE}_${SHARD}.log" 2>&1 &
else
 /projects/p33100/siosio/envs/openpi/bin/python "$SOURCE/experiments/feasibility/serve.py" --port "$PORT" --config "$CB_POLICY_CONFIG" \
  --checkpoint "$CB_ASSETS/openpi_assets/openpi-assets/checkpoints/$CB_POLICY_CONFIG" > "$ROOT/policy_${PHASE}_${SHARD}.log" 2>&1 &
fi
SERVER_PID=$!
/projects/p33100/siosio/envs/openpi/bin/python - "$PORT" "$SERVER_PID" <<'PY'
import os,socket,sys,time
for _ in range(600):
 try:
  with socket.create_connection(('127.0.0.1',int(sys.argv[1])),timeout=1):break
 except OSError:os.kill(int(sys.argv[2]),0);time.sleep(2)
else:raise RuntimeError('Policy server readiness timeout')
PY
for name in CB_ASSETS CB_UPSTREAM CB_LIBERO CB_CODE_COMMIT CB_POLICY_CONFIG; do export "SINGULARITYENV_$name=${!name}"; done
export SINGULARITYENV_SLURM_JOB_ID="$SLURM_JOB_ID"
export SINGULARITYENV_MUJOCO_GL=egl SINGULARITYENV_PYOPENGL_PLATFORM=egl
export SINGULARITYENV_PYTHONNOUSERSITE=1 SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
export SINGULARITYENV_OMP_NUM_THREADS=1 SINGULARITYENV_OPENBLAS_NUM_THREADS=1
export SINGULARITYENV_PYTHONPATH="$CB_UPSTREAM/openpi/packages/openpi-client/src"
if [ "$PHASE" = smoke ]; then ARGS=(--smoke --port "$PORT"); else ARGS=(--phase "$PHASE" --shard "$SHARD" --shards "${CB_SUITE_SHARDS:?}" --port "$PORT"); fi
singularity exec --nv --bind "/projects,/home,$ROOT:$ROOT:rw,$SOURCE:$SOURCE:ro" \
 "$CB_ASSETS/containers/aegis-py38.sif" "$CB_ASSETS/envs/aegis_sim/bin/python" \
 -u -m verifier_dataset.hil_suite_run "$ROOT" "${ARGS[@]}"
