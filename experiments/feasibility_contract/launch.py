"""Launch only a clean, published commit into an immutable project-owned root."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

BASE=Path(__file__).resolve().parents[2]
ASSETS=Path('/projects/p33100/siosio/crashbench_safelibero')
SCIENTIFIC=['experiments/feasibility_contract/'+x for x in ['contract.py','run.py','job.sbatch','README.md']]+[
           'experiments/feasibility/'+x for x in ['reference.py','geometry.py','reset_forward.py']]

def call(args): return subprocess.check_output(args,cwd=BASE,text=True).strip()
def put(path,value): path.write_text(json.dumps(value,indent=2)+'\n')

def main(stage,smoke):
    if BASE.resolve() != (Path.home()/'crash_bench').resolve(): raise RuntimeError('Wrong project checkout')
    if call(['git','remote','get-url','origin']) not in ['git@github.com:hanshuo-shuo/crash_bench.git','https://github.com/hanshuo-shuo/crash_bench.git']: raise RuntimeError('Wrong project identity')
    if call(['git','status','--porcelain=v1','--untracked-files=all']):raise RuntimeError('Dirty source')
    commit=call(['git','rev-parse','HEAD'])
    if commit!=call(['git','rev-parse','origin/codex/feasibility']):raise RuntimeError('Source not published on pilot branch')
    if call(['git','-C',str(BASE/'third_party/vlsa-aegis'),'rev-parse','HEAD'])!='2457feed5968ae803926e178c8ce8243b9ecdcf9': raise RuntimeError('Wrong upstream')
    # Existing historical project results are read only. No other jobs are modified.
    recorded=[]
    for folder,pattern in [('batches','*/batch.json'),('paired_repeats','*/plan.json'),('feasibility','*/plan.json'),('feasibility_contract','*/receipt.json')]:
        for file in (ASSETS/folder).glob(pattern):
            data=json.loads(file.read_text())
            recorded+=data.get('slurm_arrays',[])+list(data.get('jobs',{}).values())
            if data.get('job_id'):recorded.append(data['job_id'])
    queued=set(call(['squeue','-h','-u',os.environ['USER'],'-o','%A']).split())
    active=queued&{str(x) for x in recorded}
    if active:raise RuntimeError('Recorded CrashBench job active: '+str(sorted(active)))
    hashes={p:hashlib.sha256((BASE/p).read_bytes()).hexdigest() for p in SCIENTIFIC}
    if stage=='pilot':
        if smoke is None:raise RuntimeError('Smoke root required')
        if smoke.parent.resolve()!=(ASSETS/'feasibility_contract').resolve():raise RuntimeError('Wrong smoke root')
        receipt=json.loads((smoke/'receipt.json').read_text())
        complete=json.loads((smoke/'COMPLETE.json').read_text())
        if not complete.get('passed') or complete['stage']!='smoke':raise RuntimeError('Smoke incomplete')
        if receipt['scientific_hashes']!=hashes:raise RuntimeError('Scientific code changed since smoke')
        status=call(['sacct','-X','-j',str(receipt['job_id']),'--noheader','--parsable2','--format=State,ExitCode']).splitlines()
        if not status or any(x!='COMPLETED|0:0' for x in status):raise RuntimeError('Smoke Slurm did not complete cleanly: '+str(status))
    identifier=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+stage+'_'+commit[:12]
    root=ASSETS/'feasibility_contract'/identifier
    root.mkdir(parents=True,exist_ok=False)
    archive=root/'source.tar';source=root/'source';source.mkdir()
    subprocess.check_call(['git','archive','--format=tar','--output='+str(archive),commit],cwd=BASE)
    with tarfile.open(archive) as tar: tar.extractall(source)
    checks=[]
    for p in call(['git','ls-tree','-r','--name-only',commit]).splitlines():
        expected=subprocess.check_output(['git','show',commit+':'+p],cwd=BASE)
        if (source/p).read_bytes()!=expected:raise RuntimeError('Source archive mismatch '+p)
        checks.append(hashlib.sha256(expected).hexdigest()+'  '+p)
    (root/'source.sha256').write_text('\n'.join(checks)+'\n');(root/'SOURCE_COMMIT').write_text(commit+'\n')
    receipt=dict(stage=stage,root=str(root),source=str(source),code_commit=commit,scientific_hashes=hashes,
          smoke_root=str(smoke) if smoke else None,resources=dict(account='p33100',partition='short',cpus=4,mem_gib=16,
          time_minutes=15 if stage=='smoke' else 30,gpus=0),api_budget_usd=0,
          created_unix=time.time(),status='prepared_not_submitted')
    put(root/'receipt.json',receipt)
    env=os.environ.copy();env.pop('OPENROUTER_API_KEY',None)
    env.update(CB_CONTRACT_ROOT=str(root),CB_CONTRACT_STAGE=stage)
    job=subprocess.check_output(['sbatch','--parsable','--time=00:%02d:00'%receipt['resources']['time_minutes'],
          '--output='+str(root/'slurm_%j.log'),str(source/'experiments/feasibility_contract/job.sbatch')],
          cwd=BASE,env=env,text=True).strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Unparseable Slurm submission receipt')
    receipt.update(job_id=job,status='submitted');put(root/'receipt.json',receipt)
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','pilot']);p.add_argument('--smoke',type=Path)
    a=p.parse_args();main(a.stage,a.smoke)
