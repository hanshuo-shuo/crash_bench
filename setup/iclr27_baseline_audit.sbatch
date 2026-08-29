#!/usr/bin/env bash
# CPU-only strongest-baseline audit on the sealed development full capture.
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --job-name=cb_iclr27_baseline
#SBATCH --output=%x_%j.log

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$HOME/crash_bench}"
module purge
module load git/2.47.0-gcc-12.4.0
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

git_bin="$(command -v git)"
commit="$($git_bin rev-parse HEAD)"
commit_short="$($git_bin rev-parse --short=12 HEAD)"
started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
path_stamp="$(date -u +%Y%m%dT%H%M%SZ)"

capture="${CB_ICLR27_CAPTURE:-results/counterfactual_router/full_d4751330395e_20260814T152231Z}"
config="configs/iclr27/baseline_suite.yaml"
output="${CB_ICLR27_OUTPUT:-results/iclr27/baseline_audit_${commit_short}_${path_stamp}_job${SLURM_JOB_ID}}"

[[ "${CB_CODE_COMMIT:-$commit}" == "$commit" ]]
[[ -f "$capture/capture_manifest.json" ]]
[[ -f "$capture/decision_features.npz" ]]
[[ -f "$capture/decision_metadata.json" ]]
[[ -f "$capture/option_rollouts.jsonl" ]]
[[ -f "$config" ]]
if [[ -e "$output" ]]; then
  printf 'Refusing to overwrite existing output: %s\n' "$output" >&2
  exit 2
fi

printf 'commit=%s\njob=%s\nstarted_utc=%s\ncapture=%s\nconfig=%s\noutput=%s\n' \
  "$commit" "$SLURM_JOB_ID" "$started_utc" "$capture" "$config" "$output"

python scripts/iclr27/train_router_baseline_suite.py \
  --capture "$capture" \
  --config "$config" \
  --output-dir "$output"

printf 'commit=%s\nslurm_job_id=%s\nslurm_account=p33100\nslurm_partition=short\nstarted_utc=%s\nconfig=%s\ncapture=%s\n' \
  "$commit" "$SLURM_JOB_ID" "$started_utc" "$config" "$capture" \
  >"$output/slurm_provenance.txt"

python scripts/iclr27/analyze_router_baseline_suite.py --input-dir "$output"
python scripts/iclr27/plot_router_baseline_suite.py --input-dir "$output"

printf 'DONE output=%s @ %s\n' "$output" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
