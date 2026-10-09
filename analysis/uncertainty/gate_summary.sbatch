#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=cb_gate_report
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
if [ "${CB_VALIDATE_ONLY:-0}" = 1 ]; then
 "$PYTHON" -m unittest discover -s "$SOURCE/analysis/uncertainty" -p test_gate_summary.py -v
 "$PYTHON" "$SOURCE/analysis/uncertainty/validate_report.py" "$CB_TASK2_ANALYSIS"
 "$PYTHON" - "$ROOT" <<'PY'
import json,sys,os
from pathlib import Path
p=Path(sys.argv[1]);(p/'VALIDATION_PASS.json').write_text(json.dumps(dict(passed=True,code_commit=os.environ['CB_CODE_COMMIT'],job=os.environ['SLURM_JOB_ID'],real_gate_results_generated=False,api_calls=0,new_rollouts=0),indent=2)+'\n')
PY
else
 "$PYTHON" "$SOURCE/analysis/uncertainty/gate_summary.py" "$CB_GATE_ROOT" "$ROOT/analysis"
 "$PYTHON" "$SOURCE/analysis/uncertainty/report.py" "$CB_TASK2_ANALYSIS" "$ROOT/analysis" "$ROOT/report"
fi
