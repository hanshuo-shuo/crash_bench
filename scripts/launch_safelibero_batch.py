"""Submit the frozen 3200-episode matrix and its bounded network worker."""
from datetime import datetime,timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from api_budget import atomic_json
from safelibero_batch import matrix

ROOT=Path(__file__).resolve().parents[1]
ASSETS=Path('/projects/p33100/siosio/crashbench_safelibero')
PYTHON='/projects/p33100/siosio/envs/openpi/bin/python'

def command(args):return subprocess.check_output(args,text=True,cwd=ROOT).strip()

def main():
    if ROOT.resolve()!=(Path.home()/'crash_bench').resolve():raise RuntimeError('Not the authorized Quest checkout')
    if command(['git','status','--porcelain=v1','--untracked-files=all']):raise RuntimeError('Dirty source tree')
    commit=command(['git','rev-parse','HEAD'])
    if commit!=command(['git','rev-parse','@{upstream}']):raise RuntimeError('Commit not published')
    for existing in (ASSETS/'batches').glob('*/batch.json'):
        if not (existing.parent/'COMPLETE.json').exists() and not (existing.parent/'STOP.json').exists():raise RuntimeError('An unfinished baseline batch already exists')
    prior=Decimal('0')
    for path in (ASSETS/'perception_cache').glob('*/response.json'):
        value=json.loads(path.read_text())['response'].get('usage',{}).get('cost')
        if value is None:raise RuntimeError('Prior API cost is unknown')
        prior+=Decimal(str(value))
    if list((ASSETS/'perception_cache').glob('*/failed_response.json')):raise RuntimeError('Prior failed response needs billing reconciliation')
    # No new $5 allowance from relaunching: carry settled and reserved prior batches.
    for path in (ASSETS/'batches').glob('*/budget.json'):
        for call in json.loads(path.read_text())['calls'].values():prior+=Decimal(call.get('cost_usd',call['reserved_usd']))
    if prior+Decimal('.10')>5:raise RuntimeError('Reproduction API budget already exhausted')
    name=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+commit[:12]
    root=ASSETS/'batches'/name;root.mkdir(parents=True,exist_ok=False)
    (root/'runs').mkdir();(root/'perception_cache').mkdir()
    link=ROOT/'results/safelibero_batches';link.parent.mkdir(exist_ok=True)
    if link.is_symlink():
        if link.resolve()!=(ASSETS/'batches').resolve():raise RuntimeError('Unexpected results symlink')
    elif link.exists():raise RuntimeError('Results path already occupied')
    else:link.symlink_to(ASSETS/'batches',target_is_directory=True)
    plan={'name':name,'code_commit':commit,'upstream_commit':'2457feed5968ae803926e178c8ce8243b9ecdcf9','cells':matrix(),'prior_api_spend_usd':str(prior),'limit_usd':'5.00','concurrent_gpus':2,'slurm_arrays':[],'created_unix':time.time(),'policy_rng_protocol':'Fresh seed-0 OpenPI server per cell; episodes 0..49 in order, preserving the upstream persistent server RNG within each cell.'}
    atomic_json(root/'batch.json',plan)
    env=os.environ.copy();env['CB_BATCH_ROOT']=str(root)
    try:
        for indices,dependency in [('0,32%2',None),('1-31,33-63%2','first')]:
            args=['sbatch','--parsable','--hold','--array='+indices]
            if dependency:args+=['--dependency=afterok:'+plan['slurm_arrays'][0]]
            args+=['setup/safelibero_full.sbatch']
            job=subprocess.check_output(args,cwd=ROOT,env=env,text=True).strip().split(';')[0]
            if not job.isdigit():raise RuntimeError('Unexpected Slurm submission reply')
            plan['slurm_arrays'].append(job);atomic_json(root/'batch.json',plan)
        worker=shlex.join([PYTHON,'-u',str(ROOT/'scripts/batch_api_worker.py'),str(root)])+' > '+shlex.quote(str(root/'api_worker.log'))+' 2>&1'
        exists=subprocess.run(['tmux','has-session','-t','=crashbench'],capture_output=True).returncode==0
        args=['tmux','new-window','-d','-t','crashbench','-n','api_'+name[:16]] if exists else ['tmux','new-session','-d','-s','crashbench','-n','api_'+name[:16]]
        subprocess.run(args+['-c',str(ROOT),worker],check=True)
        for _ in range(45):
            if (root/'STOP.json').exists():raise RuntimeError('Network worker startup failed')
            if (root/'WORKER_READY.json').exists():break
            time.sleep(1)
        else:raise RuntimeError('Network worker did not become ready')
        subprocess.run(['scontrol','release',','.join(plan['slurm_arrays'])],check=True)
        atomic_json(root/'LAUNCHED.json',{'unix':time.time(),'slurm_arrays':plan['slurm_arrays']})
    except BaseException as e:
        atomic_json(root/'STOP.json',{'reason':str(e),'unix':time.time()})
        if plan['slurm_arrays']:subprocess.run(['scancel',*plan['slurm_arrays']],check=False)
        raise
    print(json.dumps({'batch_root':str(root),'slurm_arrays':plan['slurm_arrays'],'episodes':3200,'max_concurrent_gpus':2,'api_budget_usd':5},indent=2))

if __name__=='__main__':main()
