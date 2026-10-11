#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=cb_verifier_case_review
set -euo pipefail
cd "$CB_VERIFIER_RUN/source"
sha256sum --quiet -c "$CB_VERIFIER_RUN/source.sha256"
unset OPENROUTER_API_KEY
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CB_CODE_COMMIT="$(cat "$CB_VERIFIER_RUN/SOURCE_COMMIT")"
export PYTHONPATH="$CB_VERIFIER_RUN/source"
/projects/p33100/siosio/envs/openpi/bin/python -m unittest verifier_training.tests.test_review_cases -v > "$CB_VERIFIER_RUN/unit_tests.txt" 2>&1
/projects/p33100/siosio/envs/openpi/bin/python -u -m verifier_training.review_cases "$CB_VERIFIER_CAMPAIGN" "$CB_VERIFIER_RUN/review"
/projects/p33100/siosio/envs/openpi/bin/python - <<'PY'
import os,shutil
from pathlib import Path
from verifier_training.common import sha
root=Path(os.environ['CB_VERIFIER_RUN'])
out=Path.home()/'crash_bench/results/verifier_training'/root.name
out.mkdir(parents=True,exist_ok=True)
for p in root.rglob('*'):
    rel=p.relative_to(root)
    if p.is_file() and 'source' not in rel.parts and p.name!='source.tar':
        dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
(out/'DELIVERY.sha256').write_text(''.join(sha(p)+'  '+str(p.relative_to(out))+'\n' for p in sorted(out.rglob('*')) if p.is_file() and p.name!='DELIVERY.sha256'))
PY
