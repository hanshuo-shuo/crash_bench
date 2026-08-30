#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:30:00
#SBATCH --job-name=cb_exp_pin_pi0
#SBATCH --output=%x_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/crash_bench}"

module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1

commit="$(python scripts/expansion/hash_tree_manifest.py --print-git-head .)"
protocol_sha="$(sha256sum configs/expansion/benchmark_v1.yaml | awk '{print $1}')"
utc="$(date -u +%Y%m%dT%H%M%SZ)"
run_id="${CB_EXPANSION_RUN_ID:-${protocol_sha:0:12}_${commit:0:12}_${utc}}"
case "$run_id" in
  *[!A-Za-z0-9._-]*) printf 'invalid CB_EXPANSION_RUN_ID\n' >&2; exit 2 ;;
esac

expected_openpi_commit="15a9616a00943ada6c20a0f158e3adb39df2ccac"
actual_openpi_commit="$(python scripts/expansion/hash_tree_manifest.py \
  --print-git-head /projects/p33100/siosio/third_party/openpi)"
test "$actual_openpi_commit" = "$expected_openpi_commit"

result_root="results/expansion/governance/pi0_identity_${run_id}"
test ! -e "$result_root"

python scripts/expansion/hash_tree_manifest.py \
  --root /projects/p33100/siosio/openpi_assets/openpi-assets/checkpoints/pi0_libero \
  --label pi0_libero_checkpoint_and_norm_stats \
  --output "$result_root/checkpoint_tree_manifest.json"

printf 'commit=%s\njob=%s\nresult=%s\n' \
  "$commit" "$SLURM_JOB_ID" "$result_root"
