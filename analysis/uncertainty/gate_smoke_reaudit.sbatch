#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=cb_gate_reaudit
set -euo pipefail
ROOT="${CB_UNCERTAINTY_ROOT:?}"
SOURCE="$ROOT/source"
PYTHON=/projects/p33100/siosio/envs/openpi/bin/python
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export CB_CODE_COMMIT="$(cat "$ROOT/SOURCE_COMMIT")"
FAILED="$("$PYTHON" -c 'import json,os;print(json.load(open(os.environ["CB_UNCERTAINTY_ROOT"]+"/RECOVERY_PLAN.json"))["recovery_from"])')"
BASELINE="$("$PYTHON" -c 'import json,os;print(json.load(open(os.environ["CB_UNCERTAINTY_ROOT"]+"/RECOVERY_PLAN.json"))["configuration"]["collection_root"])')"
UPSTREAM="$("$PYTHON" -c 'import json,os;print(json.load(open(os.environ["CB_UNCERTAINTY_ROOT"]+"/RECOVERY_PLAN.json"))["upstream_root"])')"
TASK2="$("$PYTHON" -c 'import json,os;print(json.load(open(os.environ["CB_UNCERTAINTY_ROOT"]+"/RECOVERY_PLAN.json"))["analysis_root"]+"/analysis")')"
ASSET_ALIAS="$(readlink -f /projects/p33100/siosio/crashbench_safelibero)"
"$PYTHON" -m unittest discover -s "$SOURCE/analysis/uncertainty" -p 'test_gate*.py' -v
module purge
module load singularityce/4.3.1-gcc-8.5.0
export SINGULARITYENV_PYTHONNOUSERSITE=1 SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
export SINGULARITYENV_SLURM_JOB_ID="$SLURM_JOB_ID"
singularity exec --bind "/projects,/home,$ASSET_ALIAS:$ASSET_ALIAS:ro,$ROOT:$ROOT:rw,$SOURCE:$SOURCE:ro,$FAILED:$FAILED:ro,$BASELINE:$BASELINE:ro,$UPSTREAM:$UPSTREAM:ro" \
 /projects/p33100/siosio/crashbench_safelibero/containers/aegis-py38.sif \
 /projects/p33100/siosio/crashbench_safelibero/envs/aegis_sim/bin/python \
 "$SOURCE/analysis/uncertainty/gate_recover.py" audit "$ROOT"
"$PYTHON" "$SOURCE/analysis/uncertainty/validate_report.py" "$TASK2"
"$PYTHON" - "$ROOT" <<'PY'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]);(p/'REPORT_VALIDATION_PASS.json').write_text(json.dumps(dict(passed=True,code_commit=os.environ['CB_CODE_COMMIT'],job=os.environ['SLURM_JOB_ID'],real_gate_results_generated=False,api_calls=0,new_rollouts=0),indent=2)+'\n')
PY
