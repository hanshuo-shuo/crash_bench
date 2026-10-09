#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=cb_uncertainty_analysis
set -euo pipefail
ROOT="${CB_UNCERTAINTY_ROOT:?}"
AUDITED="${CB_UNCERTAINTY_AUDITED:?}"
SOURCE="$ROOT/source"
cd "$SOURCE"
export CB_CODE_COMMIT="$(cat "$ROOT/SOURCE_COMMIT")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/projects/p33100/siosio/envs/openpi/bin/python - "$ROOT" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1])
for item in json.loads((p/'SOURCE_FILES.json').read_text()):
 assert hashlib.sha256((p/'source'/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
PY
/projects/p33100/siosio/envs/openpi/bin/python "$SOURCE/analysis/uncertainty/analyze.py" "$AUDITED" "$ROOT/smoke" --smoke
/projects/p33100/siosio/envs/openpi/bin/python - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
from hashlib import sha256
p=Path(sys.argv[1]);receipt=json.loads((p/'smoke/ANALYSIS_COMPLETE.json').read_text())
assert receipt['passed'] and receipt['stage']=='smoke' and receipt['test_states']==2 and receipt['test_rollouts']==8
(p/'SMOKE_PASS.json').write_text(json.dumps(dict(passed=True,stage='Task2',test_states=2,repeats=[0,1],arms=['nominal','aegis'],test_rollouts=8,full_train_only_fit=True,smoke_receipt_sha256=sha256((p/'smoke/ANALYSIS_COMPLETE.json').read_bytes()).hexdigest()),indent=2)+'\n')
PY
/projects/p33100/siosio/envs/openpi/bin/python "$SOURCE/analysis/uncertainty/analyze.py" "$AUDITED" "$ROOT/analysis"
/projects/p33100/siosio/envs/openpi/bin/python - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
from hashlib import sha256
p=Path(sys.argv[1])
for name in ['PLATT_MODELS.json','TRAIN_THRESHOLDS.json','confident_failure_reference.csv']:
 assert (p/'smoke'/name).read_bytes()==(p/'analysis'/name).read_bytes(),name
receipt=json.loads((p/'analysis/ANALYSIS_COMPLETE.json').read_text())
assert receipt['passed'] and receipt['stage']=='full' and receipt['test_states']==18 and receipt['test_rollouts']==360
(p/'ANALYSIS_PASS.json').write_text(json.dumps(dict(passed=True,smoke_and_full_train_fits_byte_equal=True,complete_sha256=sha256((p/'analysis/ANALYSIS_COMPLETE.json').read_bytes()).hexdigest(),api_calls=0,new_rollouts=0),indent=2)+'\n')
PY
