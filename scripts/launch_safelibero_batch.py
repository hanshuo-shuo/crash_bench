"""Submit the frozen 3200-episode matrix and its bounded network worker."""
from datetime import datetime,timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import argparse
import ast
from api_budget import atomic_json
from safelibero_batch import matrix,recovery_selection

ROOT=Path(__file__).resolve().parents[1]
ASSETS=Path('/projects/p33100/siosio/crashbench_safelibero')
PYTHON='/projects/p33100/siosio/envs/openpi/bin/python'

def command(args):return subprocess.check_output(args,text=True,cwd=ROOT).strip()

def main(recover_from=None):
    if ROOT.resolve()!=(Path.home()/'crash_bench').resolve():raise RuntimeError('Not the authorized Quest checkout')
    if command(['git','status','--porcelain=v1','--untracked-files=all']):raise RuntimeError('Dirty source tree')
    commit=command(['git','rev-parse','HEAD'])
    if commit!=command(['git','rev-parse','@{upstream}']):raise RuntimeError('Commit not published')
    for existing in (ASSETS/'batches').glob('*/batch.json'):
        if not (existing.parent/'COMPLETE.json').exists() and not (existing.parent/'STOP.json').exists():raise RuntimeError('An unfinished baseline batch already exists')
    inherited={};partial={};pending=list(range(64));cache_sources=[];parent=None
    if recover_from:
        if Path(recover_from).name!=recover_from:raise RuntimeError('Recovery requires one batch directory name')
        parent=ASSETS/'batches'/recover_from
        parent_plan=json.loads((parent/'batch.json').read_text())
        if command(['squeue','-h','-j',','.join(parent_plan['slurm_arrays'])]):raise RuntimeError('Parent Slurm jobs are still active')
        scientific=['configs/reproduction.json','scripts/run_safelibero.py','setup/run_smoke.sh','tests/fixtures/main_aegis_upstream.txt']
        subprocess.run(['git','diff','--quiet',parent_plan['code_commit'],'HEAD','--',*scientific],cwd=ROOT,check=True)
        old=ast.parse(command(['git','show',parent_plan['code_commit']+':scripts/openrouter_perception.py']))
        new=ast.parse((ROOT/'scripts/openrouter_perception.py').read_text())
        for name in ['make_request','export_request','read_response']:
            bodies=[ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)) for tree in [old,new]]
            if bodies[0]!=bodies[1]:raise RuntimeError('Scientific perception settings changed')
        inherited,partial,pending=recovery_selection(parent)
        if not pending:raise RuntimeError('No incomplete cells to recover')
        cache_sources=[str(parent/'perception_cache'),*parent_plan.get('cache_sources',[])]
    known=Decimal('0');reserved=Decimal('0')
    for path in (ASSETS/'perception_cache').glob('*/response.json'):
        value=json.loads(path.read_text())['response'].get('usage',{}).get('cost')
        if value is None:raise RuntimeError('Prior API cost is unknown')
        known+=Decimal(str(value))
    if list((ASSETS/'perception_cache').glob('*/failed_response.json')):raise RuntimeError('Prior failed response needs billing reconciliation')
    # No new $5 allowance from relaunching: carry settled and reserved prior batches.
    for path in (ASSETS/'batches').glob('*/budget.json'):
        for call in json.loads(path.read_text())['calls'].values():
            if 'cost_usd' in call:known+=Decimal(call['cost_usd'])
            else:reserved+=Decimal(call['reserved_usd'])
    prior=known+reserved
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
    plan.update(prior_known_cost_usd=str(known),prior_reserved_usd=str(reserved),inherited_cells=inherited,excluded_partial_cells=partial,pending_cells=pending,cache_sources=cache_sources,recovery_parent=str(parent) if parent else None)
    atomic_json(root/'batch.json',plan)
    env=os.environ.copy();env['CB_BATCH_ROOT']=str(root)
    try:
        submissions=[(','.join(map(str,pending))+'%2',None)] if parent else [('0,32%2',None),('1-31,33-63%2','first')]
        for indices,dependency in submissions:
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
    print(json.dumps({'batch_root':str(root),'slurm_arrays':plan['slurm_arrays'],'episodes':3200,'inherited_complete_cells':len(inherited),'newly_scheduled_episodes':50*len(pending),'max_concurrent_gpus':2,'api_budget_usd':5,'prior_known_cost_usd':str(known),'prior_reserved_usd':str(reserved)},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--recover-from');args=parser.parse_args();main(args.recover_from)
