#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=cb_uncertainty_pending
set -euo pipefail
ROOT="${CB_UNCERTAINTY_ROOT:?}"
SOURCE="$ROOT/source"
export CB_CODE_COMMIT="$(cat "$ROOT/SOURCE_COMMIT")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PYTHON=/projects/p33100/siosio/envs/openpi/bin/python
"$PYTHON" - "$ROOT" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1])
for item in json.loads((p/'SOURCE_FILES.json').read_text()):
 assert hashlib.sha256((p/'source'/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
PY
"$PYTHON" -m unittest discover -s "$SOURCE/analysis/uncertainty" -p test_provisional.py -v
"$PYTHON" "$SOURCE/analysis/uncertainty/verify_gate_queue.py" "$CB_GATE_UPSTREAM/main/main_aegis.py" "$ROOT/GATE_QUEUE_SYNTHETIC_PASS.json"
"$PYTHON" "$SOURCE/analysis/uncertainty/provisional.py" "$CB_TASK2_ANALYSIS" "$ROOT/report" "$CB_GATE_RECEIPT"
