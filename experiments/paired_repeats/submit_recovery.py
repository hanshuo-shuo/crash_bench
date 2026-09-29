"""Publish a new recovery root; never edit a stopped parent or its frozen source."""
import argparse
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

SOURCE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(SOURCE/'scripts'))
from api_budget import atomic_json
from paired_budget import PairedBudget
from api_worker import active_jobs,prior_spend
from recovery import select,state_references,load_inherited
from execute import review_gate

ASSETS=Path('/projects/p33100/siosio/crashbench_safelibero')
PYTHON='/projects/p33100/siosio/envs/openpi/bin/python'
SCIENCE_FILES=['configs/reproduction.json','scripts/run_safelibero.py','scripts/openrouter_perception.py',
               *['experiments/paired_repeats/'+name for name in ['runtime.py','adapter.py','protocol.py','serve_seeded.py','execute.py']]]

def command(args):
    return subprocess.check_output(args,cwd=Path.home()/'crash_bench',text=True).strip()

def submit(root,parent):
    root=root.resolve();parent=parent.resolve();checkout=Path.home()/'crash_bench'
    if root.parent!=(ASSETS/'paired_repeats').resolve() or parent.parent!=root.parent or SOURCE!=root/'source':
        raise RuntimeError('Only isolated CrashBench result snapshots are allowed')
    if command(['git','remote','get-url','origin']) not in ['git@github.com:hanshuo-shuo/crash_bench.git','https://github.com/hanshuo-shuo/crash_bench.git']:
        raise RuntimeError('Wrong checkout')
    if command(['git','status','--porcelain=v1','--untracked-files=all']):raise RuntimeError('Live checkout dirty')
    commit=(root/'SOURCE_COMMIT').read_text().strip()
    if command(['git','rev-parse','origin/codex/paired-repeats'])!=commit:raise RuntimeError('Unpublished recovery snapshot')
    checksums=[]
    for relative in command(['git','ls-tree','-r','--name-only',commit]).splitlines():
        actual=(SOURCE/relative).read_bytes()
        if actual!=subprocess.check_output(['git','show',commit+':'+relative],cwd=checkout):
            raise RuntimeError('Recovery archive differs from published source')
        checksums.append(hashlib.sha256(actual).hexdigest()+'  '+relative)
    (root/'source.sha256').write_text('\n'.join(checksums)+'\n')
    old=json.loads((parent/'plan.json').read_text())
    if active_jobs(list(old['jobs'].values())):raise RuntimeError('Parent still active')
    for other in (ASSETS/'paired_repeats').glob('*/plan.json'):
        if active_jobs(list(json.loads(other.read_text())['jobs'].values())):
            raise RuntimeError('Another paired experiment is active')
    if (root/'plan.json').exists():raise RuntimeError('Recovery already prepared; inspect before retrying')
    science=Path(old.get('science_source',old['source'])).resolve()
    for relative in SCIENCE_FILES:
        if (science/relative).read_bytes()!=(SOURCE/relative).read_bytes():
            raise RuntimeError('Scientific implementation changed: '+relative)
    check_root=Path(old.get('checks_root',str(parent))).resolve()
    threshold=review_gate(check_root)
    inherited,pending,unfinished=select(parent)
    references=state_references(check_root)
    for row in load_inherited({'inherited_records':inherited}):
        ref=references[row['scenario']+'|'+str(row['episode'])]
        if any(row[key]!=ref[key] for key in ['qpos_sha256','active_obstacle']):
            raise RuntimeError('Parent differs from checked initial state')
    known,reserved,evidence=prior_spend(ASSETS,root)
    limit=Decimal(old['api_limit_usd'])
    if limit!=Decimal('65.00'):raise RuntimeError('Recovery requires the existing authorized ceiling')
    vlm=sum(r['method']=='aegis' for r in pending)
    if known+reserved+Decimal('.20')*vlm>limit:
        raise RuntimeError('Conserved ceiling cannot cover all bounded recovery attempts')
    plan={'name':root.name,'code_commit':old['code_commit'],'execution_commit':commit,'source':str(SOURCE),
        'science_source':str(science),'checks_root':str(check_root),'recovery_parent':str(parent),
        'assets':str(ASSETS),'jobs':{},'baseline_jobs':[],'baseline_roots':old['baseline_roots'],
        'latest_baseline':old['latest_baseline'],'inherited_records':inherited,'pending_runs':pending,
        'excluded_incomplete_runs':unfinished,'expected_initial_states':references,'api_limit_usd':str(limit),
        'max_fresh_vlm_calls':vlm,'prior_known_usd':str(known),'prior_reserved_usd':str(reserved),
        'prior_budget_evidence':evidence,'action_threshold':threshold,'max_gpus':1,'walltime':'24:00:00',
        'known_quality_warning':'Parent has 11 first-action mismatch pairs; no completed runs are replaced.',
        'created_unix':time.time()}
    atomic_json(root/'plan.json',plan)
    budget=PairedBudget(root,str(known+reserved),str(limit));budget.close()
    atomic_json(root/'cost_estimate.json',{'fresh_calls':vlm,'forecast_usd':str(Decimal('0.41832271')/224*vlm),
        'conservative_new_reservation_usd':str(Decimal('.20')*vlm),'committed_before_recovery_usd':str(known+reserved),
        'same_total_ceiling_usd':str(limit)})
    env=os.environ.copy();env.update(CB_PAIRED_ROOT=str(root),CB_PAIRED_STAGE='full',
        CB_SCIENCE_SOURCE=str(science),CB_SCIENCE_COMMIT=old['code_commit'],
        CB_PARENT_RESULTS=str(check_root),CB_EVAL_ENTRYPOINT=str(SOURCE/'experiments/paired_repeats/recover_execute.py'))
    job=subprocess.check_output(['sbatch','--parsable','--hold','--time=24:00:00',
        '--job-name=cb_paired_recovery','--output='+str(root/'slurm_full_%j.log'),
        str(SOURCE/'experiments/paired_repeats/run_gpu.sbatch')],cwd=checkout,env=env,text=True).strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Invalid submission receipt')
    plan['jobs']['full']=job;atomic_json(root/'plan.json',plan)
    worker=shlex.join([PYTHON,'-u',str(SOURCE/'experiments/paired_repeats/api_worker.py'),str(root),'full'])+' > '+shlex.quote(str(root/'api_full.log'))+' 2>&1'
    try:
        exists=subprocess.run(['tmux','has-session','-t','=crashbench'],capture_output=True).returncode==0
        tmux=(['tmux','new-window','-d','-t','crashbench','-n','paired_recovery'] if exists else
              ['tmux','new-session','-d','-s','crashbench','-n','paired_recovery'])
        subprocess.run(tmux+['-c',str(checkout),worker],check=True)
        for _ in range(45):
            if (root/'STOP.json').exists():raise RuntimeError('Recovery worker failed to start')
            if (root/'WORKER_READY_full.json').exists():break
            time.sleep(1)
        else:raise RuntimeError('Recovery worker readiness timeout')
        subprocess.run(['scontrol','release',job],check=True)
    except BaseException as error:
        atomic_json(root/'STOP.json',{'stage':'recovery_submission','reason':str(error),'unix':time.time()})
        subprocess.run(['scancel',job],check=False);raise
    receipt={'job':job,'inherited_runs':len(inherited),'new_runs':len(pending),'total_runs':600,
        'new_nominal_runs':sum(r['method']=='nominal' for r in pending),'new_aegis_runs':vlm,
        'api_limit_usd':str(limit),'science_commit':old['code_commit'],'execution_commit':commit,
        'parent_root':str(parent),'root':str(root)}
    atomic_json(root/'RECOVERY_SUBMITTED.json',receipt);print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('parent',type=Path);a=p.parse_args()
    submit(a.root,a.parent)
