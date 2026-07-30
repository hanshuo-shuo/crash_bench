#!/bin/bash
# Qwen3-VL-32B monitor (GPU 0) + OpenVLA-7B evaluator (GPU 1), node-local HTTP.
# Defaults: five frozen on-path walls x K=3; no external network is needed.
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:2
#SBATCH --constraint=sxm
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --job-name=crashbench_vlmmon
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
nvidia-smi -L
module purge
module load mamba/24.3.0
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/projects/p33100/siosio/huggingface_cache
cd "$HOME/crash_bench"

PORT="${CB_MONITOR_PORT:-18765}"
QWEN_LOG="qwen_vlm_${SLURM_JOB_ID}.log"
CUDA_VISIBLE_DEVICES=0 "$HOME/pytorch-cuda-12-4/bin/python" \
  scripts/qwen_vlm_server.py \
  --model "${CB_QWEN_MODEL:-Qwen/Qwen3-VL-32B-Instruct}" \
  --port "$PORT" > "$QWEN_LOG" 2>&1 &
QWEN_PID=$!
cleanup() {
  kill "$QWEN_PID" 2>/dev/null || true
  wait "$QWEN_PID" 2>/dev/null || true
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 120); do
  if ! kill -0 "$QWEN_PID" 2>/dev/null; then
    echo "Qwen server exited during startup"
    tail -100 "$QWEN_LOG"
    exit 1
  fi
  if curl --silent --fail "http://127.0.0.1:${PORT}/health"; then
    ready=1
    break
  fi
  sleep 10
done
if [ "$ready" -ne 1 ]; then
  echo "Qwen server did not become ready"
  tail -100 "$QWEN_LOG"
  exit 1
fi

CUDA_VISIBLE_DEVICES=1 "$HOME/crash_bench/envs/openvla/bin/python" \
  scripts/safety_baseline_suite.py \
  --conditions vlm_monitor \
  --repeats "${CB_REPEATS:-3}" \
  --monitor-url "http://127.0.0.1:${PORT}" \
  --out "${CB_OUT:-results/safety_baselines_vlm.json}" \
  --video-dir "${CB_VIDEO_DIR:-results/safety_baseline_videos}"

echo "DONE @ $(date)"
