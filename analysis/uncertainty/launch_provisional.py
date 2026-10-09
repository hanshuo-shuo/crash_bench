"""Launch bounded CPU-only pending report and synthetic gate engineering check."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from common import ASSETS,atomic_json,config
from preflight import stream_sha
from audit_shard import terminal_accounting


def main(root,task2,gate):
    source=root/'source';repo=Path.home()/'crash_bench';commit=(root/'SOURCE_COMMIT').read_text().strip()
    if Path(__file__).resolve().parents[2]!=source.resolve() or any(p.parent!=(ASSETS/'uncertainty').resolve() for p in [root,task2,gate]):raise RuntimeError('Immutable own-project roots required')
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo) or subprocess.run(['git','merge-base','--is-ancestor',commit,'origin/codex/feasibility'],cwd=repo).returncode:raise RuntimeError('Clean tested published code required')
    if (root/'SUBMITTED.json').exists():raise RuntimeError('Never resubmit provisional root')
    receipt=json.loads((task2/'SUBMITTED.json').read_text());terminal_accounting(receipt['job'])
    if not json.loads((task2/'ANALYSIS_PASS.json').read_text())['passed']:raise RuntimeError('Task2 pass required')
    complete=json.loads((task2/'analysis/ANALYSIS_COMPLETE.json').read_text())
    if not complete['passed'] or complete['test_rollouts']!=360:raise RuntimeError('Complete1200-input/full360-test analysis required')
    for name,digest in complete['files'].items():
        if stream_sha(task2/'analysis'/name)!=digest:raise RuntimeError('Task2 output changed')
    files=[]
    for name in subprocess.check_output(['git','ls-tree','-r','--name-only',commit],cwd=repo,text=True).splitlines():
        if (source/name).read_bytes()!=subprocess.check_output(['git','show',commit+':'+name],cwd=repo):raise RuntimeError('Provisional source changed')
        files.append(dict(path=name,sha256=stream_sha(source/name)))
    atomic_json(root/'SOURCE_FILES.json',files)
    plan=json.loads((gate/'SUBMITTED.json').read_text());cfg=config()['task2']['resources']
    if cfg!=dict(account='p33100',partition='short',cpus=2,memory_gb=8,minutes=60):raise RuntimeError('Existing CPU resource envelope changed')
    env=dict(os.environ,CB_UNCERTAINTY_ROOT=str(root),CB_TASK2_ANALYSIS=str(task2/'analysis'),CB_GATE_UPSTREAM=plan['upstream_root'],CB_GATE_RECEIPT=str(gate/'SUBMITTED.json'))
    job=subprocess.check_output(['sbatch','--parsable','--output='+str(root/'slurm_%j.log'),str(source/'analysis/uncertainty/provisional.sbatch')],cwd=repo,env=env,text=True).strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Invalid CPU Slurm receipt')
    result=dict(job=job,root=str(root),code_commit=commit,task2_root=str(task2),gate_root=str(gate),resources=cfg,submitted_unix=time.time(),scope='Task3 pending provisional report plus deterministic mocked gate queue/replan test; no model/simulator/new scientific rollouts/API/refit',primary_task2_outputs_unchanged=True)
    atomic_json(root/'SUBMITTED.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('task2',type=Path);p.add_argument('gate',type=Path);a=p.parse_args();main(a.root.resolve(),a.task2.resolve(),a.gate.resolve())
