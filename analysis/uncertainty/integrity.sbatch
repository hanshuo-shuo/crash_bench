#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --job-name=cb_uncertainty_integrity
set -euo pipefail
ROOT="${CB_UNCERTAINTY_ROOT:?}"
COLLECTION="${CB_UNCERTAINTY_COLLECTION:?}"
SHARD="${CB_UNCERTAINTY_SHARD:?}"
SOURCE="$ROOT/source"
cd "$SOURCE"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/projects/p33100/siosio/envs/openpi/bin/python - "$ROOT" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1])
for item in json.loads((p/'SOURCE_FILES.json').read_text()):
 assert hashlib.sha256((p/'source'/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
PY
if [[ "$SHARD" == "all" ]]; then
 /projects/p33100/siosio/envs/openpi/bin/python - "$COLLECTION" <<'PY'
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'analysis/uncertainty'))
from audit_shard import terminal_accounting
p=Path(sys.argv[1]);plan=json.loads((p/'SUBMITTED.json').read_text())
for job in plan['jobs']:terminal_accounting(job)
deadline=time.monotonic()+180
while not ((p/'COLLECTION_COMPLETE.json').exists() and (p/'API_FINAL.json').exists()):
 if (p/'STOP.json').exists() or time.monotonic()>deadline:raise RuntimeError('Successful complete collection/settled API receipt missing')
 time.sleep(2)
PY
 /projects/p33100/siosio/envs/openpi/bin/python "$SOURCE/analysis/uncertainty/audit_full.py" "$COLLECTION" "$ROOT/full_audit"
else
 /projects/p33100/siosio/envs/openpi/bin/python "$SOURCE/analysis/uncertainty/audit_shard.py" \
  "$COLLECTION" "$SHARD" "$ROOT/verified_shard_$SHARD"
fi
