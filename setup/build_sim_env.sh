#!/usr/bin/env bash
set -euo pipefail
A="${CB_ASSETS:?}"
ENV="$A/envs/aegis_sim"
test ! -e "$ENV/READY.json"
mkdir -p "$A/envs" "$A/tmp/build"
export TMPDIR="$A/tmp/build"
RUNTIMES=("$A"/python/cpython-3.8.20-*/bin/python3.8)
test "${#RUNTIMES[@]}" -eq 1
RUNTIME="${RUNTIMES[0]}"
test -x "$RUNTIME"
if [ ! -x "$ENV/bin/python" ]; then "$RUNTIME" -m venv "$ENV"; fi
P="$ENV/bin/python"
"$P" -m pip install --no-index --find-links "$A/wheelhouse" --no-deps pip==24.3.1 setuptools==75.3.0 wheel==0.45.1
"$P" -m pip install --no-index --find-links "$A/wheelhouse" --no-deps numpy==1.22.4 pillow==10.4.0 typing-extensions==4.12.2 torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113
"$P" -m pip install --no-index --find-links "$A/wheelhouse" --no-deps --no-build-isolation -r "$A/simulation-requirements.txt"
"$P" - <<'PY'
import json,os,pathlib,shutil,importlib.metadata as m
import torch,mujoco,cvxpy,open3d,robosuite,groundingdino
from groundingdino.util.inference import load_model
root=pathlib.Path(os.environ['CB_ASSETS'])
source=pathlib.Path(groundingdino.__file__).parent/'config/GroundingDINO_SwinT_OGC.py'
assert source.is_file(),source
shutil.copyfile(source,root/'GroundingDINO/GroundingDINO_SwinT_OGC.py')
assert torch.cuda.is_available()
packages={n:m.version(n) for n in ['torch','numpy','mujoco','robosuite','cvxpy','osqp','open3d','groundingdino-py']}
(root/'envs/aegis_sim/READY.json').write_text(json.dumps({'packages':packages,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'job':os.environ.get('SLURM_JOB_ID')},indent=2)+'\n')
print(packages)
PY
