#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --job-name=cb_uncertainty_preflight
set -euo pipefail
ROOT="${CB_UNCERTAINTY_ROOT:?}"
SOURCE="$ROOT/source"
cd "$SOURCE"
export CB_ASSETS=/projects/p33100/siosio/crashbench_safelibero
export CB_LIVE_UPSTREAM="$HOME/crash_bench/third_party/vlsa-aegis"
export CB_SMOKE_ROOT="$CB_ASSETS/uncertainty/20261008T042856Z_6ca85c909ebb"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
module load git/2.37.2
/projects/p33100/siosio/envs/openpi/bin/python - "$ROOT" <<'PY'
import json,sys,hashlib
from pathlib import Path
p=Path(sys.argv[1])
for item in json.loads((p/'SOURCE_FILES.json').read_text()):
 assert hashlib.sha256((p/'source'/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
PY
/projects/p33100/siosio/envs/openpi/bin/python "$SOURCE/analysis/uncertainty/preflight.py" fingerprints "$ROOT"
module purge
module load singularityce/4.3.1-gcc-8.5.0
export SINGULARITYENV_CB_SMOKE_ROOT="$CB_SMOKE_ROOT"
export SINGULARITYENV_MUJOCO_GL=glx
export SINGULARITYENV_PYTHONNOUSERSITE=1 SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
singularity exec --bind "/projects,/home,$ROOT:$ROOT:rw,$SOURCE:$SOURCE:ro,$ROOT/upstream:$ROOT/upstream:ro" \
 "$CB_ASSETS/containers/aegis-py38.sif" "$CB_ASSETS/envs/aegis_sim/bin/python" \
 -u "$SOURCE/analysis/uncertainty/preflight.py" geometry "$ROOT"
