"""Submit a frozen CPU evidence audit only for a completed600-case shard."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from audit_shard import terminal_accounting,validate_receipt
from common import ASSETS,atomic_json,config,full_schedule
from preflight import stream_sha


def remaining_dependencies(jobs):
    """Completed jobs can be purged from the controller; verify accounting first."""
    active=[];completed=[]
    allowed={'PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED','RESIZING','REQUEUED','REQUEUE_FED','STAGE_OUT'}
    for job in jobs:
        result=subprocess.run(['sacct','-j',str(job),'--format=JobID,State,ExitCode,ElapsedRaw','-n','-P'],capture_output=True,text=True,check=True)
        rows=[x.split('|') for x in result.stdout.splitlines() if x.split('|')[0]==str(job)]
        if len(rows)!=1:raise RuntimeError('Dependency accounting unavailable')
        _,state,exit_code,elapsed=rows[0]
        if state=='COMPLETED' and exit_code=='0:0':completed.append(dict(job=str(job),state=state,exit_code=exit_code,elapsed_seconds=int(elapsed)))
        elif state in allowed:active.append(str(job))
        else:raise RuntimeError('Dependency failed or unknown: '+str(rows[0]))
    return active,completed


def launch(root,collection,shard):
    source=root/'source';checkout=Path.home()/'crash_bench'
    if Path(__file__).resolve().parents[2]!=source.resolve() or root.parent!=(ASSETS/'uncertainty').resolve() or collection.parent!=(ASSETS/'uncertainty').resolve():
        raise RuntimeError('Use immutable audit and valid uncertainty collection roots')
    def git(*args):return subprocess.check_output(['git',*args],cwd=checkout)
    commit=(root/'SOURCE_COMMIT').read_text().strip()
    if git('status','--porcelain') or git('rev-parse','origin/codex/feasibility').decode().strip()!=commit:raise RuntimeError('Live checkout dirty or archive unpublished')
    if (root/'SUBMITTED.json').exists():raise RuntimeError('Never resubmit an audit root')
    plan=json.loads((collection/'SUBMITTED.json').read_text());accounting=None
    dependencies=[];completed_dependencies=[]
    if len(plan['jobs'])!=2 or len(set(plan['jobs']))!=2 or any(not str(job).isdigit() for job in plan['jobs']):raise RuntimeError('Exactly two frozen numeric job IDs required')
    if shard is not None:
        accounting=terminal_accounting(plan['jobs'][shard])
        validate_receipt(json.loads((collection/'shards'/str(shard)/'COMPLETE.json').read_text()),list(full_schedule(plan['configuration'],shard)),plan['code_commit'])
    else:dependencies,completed_dependencies=remaining_dependencies(plan['jobs'])
    files=[]
    for name in git('ls-tree','-r','--name-only',commit).decode().splitlines():
        if (source/name).read_bytes()!=git('show',commit+':'+name):raise RuntimeError('Audit source differs')
        files.append(dict(path=name,sha256=stream_sha(source/name)))
    atomic_json(root/'SOURCE_FILES.json',files)
    resources=config()['integrity_audit']['full_resources' if shard is None else 'resources']
    env=dict(os.environ,CB_UNCERTAINTY_ROOT=str(root),CB_UNCERTAINTY_COLLECTION=str(collection),CB_UNCERTAINTY_SHARD='all' if shard is None else str(shard))
    arguments=['sbatch','--parsable','--time='+str(resources['minutes']),'--output='+str(root/'slurm_%j.log')]
    if dependencies:arguments.append('--dependency=afterok:'+':'.join(dependencies))
    arguments.append(str(source/'analysis/uncertainty/integrity.sbatch'))
    job=subprocess.check_output(arguments,env=env,cwd=checkout,text=True).strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Invalid Slurm job receipt')
    receipt=dict(job=job,root=str(root),collection=str(collection),shard=shard,collection_accounting=accounting,code_commit=commit,
        submitted_unix=time.time(),resources=resources,dependency_jobs=dependencies,completed_dependencies=completed_dependencies,scope='Read-only terminal evidence audit/package; no policy/API/new rollouts/fitting')
    atomic_json(root/'SUBMITTED.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('collection',type=Path);p.add_argument('shard',choices=['0','1','all']);a=p.parse_args();launch(a.root.resolve(),a.collection.resolve(),None if a.shard=='all' else int(a.shard))
