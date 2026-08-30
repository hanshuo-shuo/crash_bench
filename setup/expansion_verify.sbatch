#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --job-name=cb_exp_d1_verify
#SBATCH --output=%x_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/crash_bench}"

module purge
module load mamba/24.3.0
policy="${CB_POLICY:-openvla}"
task_id="${CB_TASK_ID:-0}"
repeats="${CB_REPEATS:-3}"
case "$policy" in
  openvla) source activate "$HOME/crash_bench/envs/openvla" ;;
  pi0) source /projects/p33100/siosio/envs/openpi/bin/activate ;;
  *) printf 'unsupported CB_POLICY=%s\n' "$policy" >&2; exit 2 ;;
esac
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH="$HOME/crash_bench"
export HF_HOME=/projects/p33100/siosio/huggingface_cache
export OPENPI_DATA_HOME=/projects/p33100/siosio/openpi_assets
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

commit="$(python scripts/expansion/hash_tree_manifest.py --print-git-head .)"
protocol_sha="$(sha256sum configs/expansion/benchmark_v1.yaml | awk '{print $1}')"
utc="$(date -u +%Y%m%dT%H%M%SZ)"
case "$task_id" in
  0|2) ;;
  *) printf 'unsupported CB_TASK_ID=%s\n' "$task_id" >&2; exit 2 ;;
esac
case "$repeats" in
  3|5) ;;
  *) printf 'unsupported CB_REPEATS=%s\n' "$repeats" >&2; exit 2 ;;
esac
run_id="${protocol_sha:0:12}_${commit:0:12}_${utc}"
result_root="results/expansion/d1_exactness/${policy}_task${task_id}_${run_id}"
test ! -e "$result_root"

python scripts/expansion/verify_exact_branching.py \
  --policy "$policy" \
  --suite libero_spatial \
  --task-id "$task_id" \
  --init-index -1 \
  --settle-steps 10 \
  --warmup-policy-steps 1 \
  --trace-policy-steps 3 \
  --repeats "$repeats" \
  --seed 0 \
  --exposure-registry results/expansion/governance/exposure_registry.json \
  --output "$result_root/conformance.json"

printf 'commit=%s\njob=%s\nresult=%s\n' "$commit" "$SLURM_JOB_ID" "$result_root"
