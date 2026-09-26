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
module load singularityce/4.3.1-gcc-8.5.0
export SINGULARITY_CACHEDIR="$A/singularity_cache"
export SINGULARITY_TMPDIR="$A/tmp/container_$SLURM_JOB_ID"
mkdir -p "$A/containers" "$SINGULARITY_TMPDIR" "$A/runs"
TARGET="$A/containers/aegis-py38.sif"
test ! -e "$TARGET"
python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); assert "oci" in s["stages"]' "$A/staging.json"
singularity build "$TARGET.partial" "oci:$A/oci/pytorch:aegis"
singularity exec "$TARGET.partial" python -c 'import torch,sys; print(sys.version,torch.__version__,torch.version.cuda)'
mv "$TARGET.partial" "$TARGET"
sha256sum "$TARGET" > "$TARGET.sha256"
printf 'commit=%s job=%s\n' "$(git rev-parse HEAD)" "$SLURM_JOB_ID" > "$TARGET.build.txt"
