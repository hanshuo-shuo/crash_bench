"""Submit one immutable ten-minute CPU audit; never synchronize live source."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from common import ASSETS, atomic_json
from preflight import stream_sha


def launch(root):
    source=root/'source'; checkout=Path.home()/'crash_bench'
    if Path(__file__).resolve().parents[2]!=source.resolve() or root.parent!=(ASSETS/'uncertainty').resolve():
        raise RuntimeError('Use a fresh immutable source archive in the uncertainty root')
    def git(*args): return subprocess.check_output(['git',*args],cwd=checkout)
    commit=(root/'SOURCE_COMMIT').read_text().strip()
    if git('status','--porcelain') or git('rev-parse','origin/codex/feasibility').decode().strip()!=commit:
        raise RuntimeError('Live checkout dirty or archive unpublished')
    if (root/'SUBMITTED.json').exists(): raise RuntimeError('Never resubmit a root')
    files=[]
    for name in git('ls-tree','-r','--name-only',commit).decode().splitlines():
        if (source/name).read_bytes()!=git('show',commit+':'+name): raise RuntimeError('Source mismatch: '+name)
        files.append(dict(path=name,sha256=stream_sha(source/name)))
    atomic_json(root/'SOURCE_FILES.json',files)
    env=dict(os.environ,CB_UNCERTAINTY_ROOT=str(root))
    job=subprocess.check_output(['sbatch','--parsable','--output='+str(root/'slurm_%j.log'),
        str(source/'analysis/uncertainty/preflight.sbatch')],env=env,cwd=checkout,text=True).strip().split(';')[0]
    if not job.isdigit(): raise RuntimeError('Invalid Slurm receipt')
    receipt=dict(job=job,root=str(root),code_commit=commit,submitted_unix=time.time(),
        resources=dict(account='p33100',partition='short',cpus=8,memory_gb=16,minutes=10),
        scope='static saved geometry and asset/source fingerprints; no policy/API/rollout')
    atomic_json(root/'SUBMITTED.json',receipt); print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();launch(a.root.resolve())
