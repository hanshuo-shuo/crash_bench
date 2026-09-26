#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=cb_aegis_container
#SBATCH --output=cb_aegis_container_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/crash_bench}"
A=/projects/p33100/siosio/crashbench_safelibero
module purge
module load git/2.37.2
module load singularityce/4.3.1-gcc-8.5.0
export SINGULARITY_CACHEDIR="$A/singularity_cache"
export SINGULARITY_TMPDIR="$A/tmp/container_$SLURM_JOB_ID"
mkdir -p "$A/containers" "$SINGULARITY_TMPDIR" "$A/runs"
TARGET="$A/containers/aegis-py38.sif"
test ! -e "$TARGET"
PARTIAL="$TARGET.build_$SLURM_JOB_ID.partial"
/projects/p33100/siosio/envs/openpi/bin/python -c 'import json,sys; s=json.load(open(sys.argv[1])); assert "oci" in s["stages"]' "$A/staging.json"
singularity build "$PARTIAL" "oci:$A/oci/cudagl-ubuntu2004:aegis"
singularity exec "$PARTIAL" /bin/bash -c 'ldd --version | head -1; cat /etc/os-release; nvcc --version'
mv "$PARTIAL" "$TARGET"
sha256sum "$TARGET" > "$TARGET.sha256"
printf 'commit=%s job=%s\n' "$(git rev-parse HEAD)" "$SLURM_JOB_ID" > "$TARGET.build.txt"

export CB_ASSETS="$A"
/projects/p33100/siosio/envs/openpi/bin/python scripts/verify_assets.py
