#!/usr/bin/env bash
# CPU-only verification for the ICLR 2027 stage-1 paper truth source.
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=cb_iclr27_s1_verify
#SBATCH --output=%x_%j.log

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$HOME/crash_bench}"
module purge
module load git/2.47.0-gcc-12.4.0
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

git_executable="$(command -v git)"
commit="$("$git_executable" rev-parse HEAD)"
commit_short="$("$git_executable" rev-parse --short=12 HEAD)"
branch="$("$git_executable" branch --show-current)"
created_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
path_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
result_dir="results/iclr27/stage1_verify_${commit_short}_${path_stamp}_job${SLURM_JOB_ID}"

mkdir -p "$result_dir"
printf 'commit=%s\nbranch=%s\nhost=%s\njob=%s\nstarted=%s\n' \
  "$commit" "$branch" "$(hostname)" "$SLURM_JOB_ID" "$created_at_utc"

python -m pytest -p no:cacheprovider tests -q
python scripts/audit_repo.py
"$git_executable" diff --check

printf '{\n  "schema_version": 1,\n  "kind": "iclr27_stage1_verification",\n  "status": "passed",\n  "git_commit": "%s",\n  "git_branch": "%s",\n  "created_at_utc": "%s",\n  "slurm_job_id": "%s",\n  "slurm_account": "p33100",\n  "slurm_partition": "short",\n  "host": "%s",\n  "commands": {\n    "pytest": "passed",\n    "audit_repo": "passed",\n    "git_diff_check": "passed"\n  }\n}\n' \
  "$commit" "$branch" "$created_at_utc" "$SLURM_JOB_ID" "$(hostname)" \
  >"$result_dir/verification.json"

printf 'RESULT_PATH=%s\n' "$result_dir"
