#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=cb_fd1000_audit
set -euo pipefail
ROOT="${CB_FD_ROOT:?}"
cd "$ROOT/source"
sha256sum --quiet -c "$ROOT/source.sha256"
unset OPENROUTER_API_KEY
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
if [[ "$CB_FD_STAGE" == audit-constraints ]]; then
    /projects/p33100/siosio/envs/openpi/bin/python -m verifier_dataset.statistics_assets prepare "$ROOT"
    export PYTHONPATH="$ROOT/statistics_packages:$ROOT/source"
fi
/projects/p33100/siosio/envs/openpi/bin/python -u - "$ROOT" <<'PY'
from pathlib import Path
import sys
from verifier_dataset.inventory import read
root=Path(sys.argv[1])
if read(root/'CAMPAIGN.json')['stage']=='audit-constraints':
    from verifier_dataset.constraint_audit import main
else:
    from verifier_dataset.pool_audit import main
main(root,[Path(x) for x in read(root/'AUDIT_SOURCES.json')])
PY
