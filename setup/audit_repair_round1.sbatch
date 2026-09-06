#!/usr/bin/env bash
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=cb_repair_r1
#SBATCH --output=%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/crash_bench}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
repair_commit=$(git rev-parse HEAD)
test -z "$(git status --porcelain=v1 --untracked-files=all)"
repair_data=results/expansion/d5_statewise/e2a5264a802a_fe660c168e72_20260830T105726Z/postprocess_a4b1714
repair_output="results/repair_round1/${repair_commit:0:12}_${SLURM_JOB_ID}"
python scripts/expansion/run_round1_repair.py \
  --merged-dir "$repair_data/merged" \
  --feature-cache "$repair_data/feature_cache/features.npz" \
  --artifact-store /projects/p33100/siosio/crash_bench_provenance \
  --epochs 100 --output-dir "$repair_output"
python - "$repair_output" <<'PYTHON'
import json, os, platform, subprocess, sys
from pathlib import Path
import numpy, torch
out = Path(sys.argv[1])
provenance = dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                 slurm_job_id=os.environ['SLURM_JOB_ID'], hostname=platform.node(),
                 python=sys.version, torch=torch.__version__, numpy=numpy.__version__)
(out / 'runtime.json').write_text(json.dumps(provenance, indent=2) + '\n')
PYTHON
