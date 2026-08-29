#!/usr/bin/env bash
# Existing-data-only CPU run for the Phase 2.5B-R advantage-router resolver.
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=cb_iclr27_adr_resolve
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

git_bin="$(command -v git)"
commit="$($git_bin rev-parse HEAD)"
commit_short="$($git_bin rev-parse --short=12 HEAD)"
started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
path_stamp="$(date -u +%Y%m%dT%H%M%SZ)"

fingerprint="d4751330395e"
capture="${CB_ICLR27_CAPTURE:-results/counterfactual_router/full_d4751330395e_20260814T152231Z}"
support="${CB_ICLR27_SUPPORT_AUDIT:-results/iclr27/option_support_audit_787623226de0_20260829T112149Z}"
crossfit="${CB_ICLR27_SUPPORT_CROSSFIT:-results/iclr27/support_crossfit_df3168c75843_20260829T115455Z_job5137872}"
config="configs/iclr27/advantage_router_resolver.yaml"
output="${CB_ICLR27_OUTPUT:-results/iclr27/advantage_router_resolver_${fingerprint}_${path_stamp}_job${SLURM_JOB_ID}}"

[[ "${CB_CODE_COMMIT:-$commit}" == "$commit" ]]
[[ -f "$capture/capture_manifest.json" ]]
[[ -f "$capture/decision_features.npz" ]]
[[ -f "$capture/decision_metadata.json" ]]
[[ -f "$capture/option_rollouts.jsonl" ]]
[[ -f "$support/manifest.json" ]]
[[ -f "$support/gate_decision.json" ]]
[[ -f "$crossfit/manifest.json" ]]
[[ -f "$crossfit/all_predictions.npz" ]]
[[ -f "$crossfit/gate_decision.json" ]]
[[ -f "$config" ]]
if [[ -e "$output" ]]; then
  printf 'Refusing to overwrite existing output: %s\n' "$output" >&2
  exit 2
fi

printf 'commit=%s\njob=%s\nstarted_utc=%s\nfingerprint=%s\ncapture=%s\nsupport=%s\ncrossfit=%s\nconfig=%s\noutput=%s\n' \
  "$commit" "$SLURM_JOB_ID" "$started_utc" "$fingerprint" "$capture" \
  "$support" "$crossfit" "$config" "$output"

python scripts/iclr27/run_advantage_router_resolver.py \
  --capture "$capture" \
  --option-support "$support" \
  --support-crossfit "$crossfit" \
  --config "$config" \
  --output-dir "$output"

printf 'commit=%s\nslurm_job_id=%s\nslurm_account=p33100\nslurm_partition=short\nstarted_utc=%s\nconfig=%s\ncapture=%s\nsupport_audit=%s\nsupport_crossfit=%s\noutput=%s\n' \
  "$commit" "$SLURM_JOB_ID" "$started_utc" "$config" "$capture" "$support" \
  "$crossfit" "$output" >"$output/slurm_provenance.txt"

printf 'DONE output=%s @ %s\n' "$output" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
