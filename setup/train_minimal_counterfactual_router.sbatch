#!/bin/bash
# CPU-only fit/evaluation of the minimal exact-state counterfactual router.
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=cf_min_router
#SBATCH --output=%x_%j.log

set -euo pipefail

CAPTURE="${CB_CF_CAPTURE:?set CB_CF_CAPTURE to the sealed full capture}"
OUTPUT="${CB_CF_ROUTER_OUTPUT:?set CB_CF_ROUTER_OUTPUT to a new JSON path}"

module purge
module load mamba/24.3.0 git/2.47.0-gcc-12.4.0
GIT_BIN="$(command -v git)"
source activate "$HOME/crash_bench/envs/openvla"

REPO_ROOT="${CB_REPO_ROOT:-$HOME/crash_bench}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
"$GIT_BIN" diff --quiet
"$GIT_BIN" diff --cached --quiet
COMMIT="$("$GIT_BIN" rev-parse HEAD)"
[[ "${CB_CODE_COMMIT:-$COMMIT}" == "$COMMIT" ]]
[[ -f "$CAPTURE/decision_features.npz" ]]
[[ ! -e "$OUTPUT" ]]

python scripts/train_minimal_counterfactual_router.py \
  --capture "$CAPTURE" \
  --output "$OUTPUT" \
  --pca-components 16 \
  --ridge-alpha 1.0 \
  --catastrophe-cost 5.0

echo "DONE commit=$COMMIT output=$OUTPUT @ $(date)"
