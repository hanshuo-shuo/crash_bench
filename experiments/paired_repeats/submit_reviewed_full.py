"""Submit reviewed 600 runs using the unchanged, already-tested science snapshot.

Only this newly published orchestration snapshot changes the budget and walltime.
The original checks, source/ archive, all raw evidence and provenance stay intact.
"""
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

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(SOURCE/'scripts'))
from api_budget import Budget, atomic_json
from api_worker import active_jobs, prior_spend
from execute import review_gate
PYTHON='/projects/p33100/siosio/envs/openpi/bin/python'
SCIENTIFIC=['configs/reproduction.json','scripts/run_safelibero.py','scripts/openrouter_perception.py',
            *['experiments/paired_repeats/'+name for name in
              ['runtime.py','adapter.py','protocol.py','execute.py','serve_seeded.py','run_gpu.sbatch']]]

def command(args):
    return subprocess.check_output(args,cwd=Path.home()/'crash_bench',text=True).strip()

def submit(root):
    root=root.resolve();checkout=Path.home()/'crash_bench'
    if root.parent.resolve()!=Path('/projects/p33100/siosio/crashbench_safelibero/paired_repeats').resolve():
        raise RuntimeError('Wrong experiment root')
    if SOURCE.parent.name!='control' or SOURCE.parent.parent!=root:
        raise RuntimeError('Control snapshot must be isolated under this experiment root')
    if command(['git','remote','get-url','origin']) not in ['git@github.com:hanshuo-shuo/crash_bench.git','https://github.com/hanshuo-shuo/crash_bench.git']:
        raise RuntimeError('Wrong checkout')
    if command(['git','status','--porcelain=v1','--untracked-files=all']):
        raise RuntimeError('Live checkout is dirty')
    control_commit=command(['git','rev-parse','origin/codex/paired-repeats'])
    for relative in command(['git','ls-tree','-r','--name-only',control_commit]).splitlines():
        if (SOURCE/relative).read_bytes()!=subprocess.check_output(['git','show',control_commit+':'+relative],cwd=checkout):
            raise RuntimeError('Control archive is not the published commit: '+relative)
    science=root/'source';plan=json.loads((root/'plan.json').read_text())
    for relative in SCIENTIFIC:
        if (SOURCE/relative).read_bytes()!=(science/relative).read_bytes():
            raise RuntimeError('Scientific code changed after checks: '+relative)
    subprocess.run(['sha256sum','--quiet','-c',str(root/'source.sha256')],cwd=science,check=True)
    threshold=review_gate(root)
    if (root/'STOP.json').exists() or 'full' in plan['jobs']:
        raise RuntimeError('Already submitted or stopped; no duplicate experiment')
    if active_jobs(list(plan['jobs'].values())+plan['baseline_jobs']):
        raise RuntimeError('Checks or baseline jobs are still active')
    states=command(['sacct','-X','-n','-P','-j',plan['jobs']['checks'],'--format=State,ExitCode']).splitlines()
    if states!=['COMPLETED|0:0']:
        raise RuntimeError('Checks did not finish successfully')
    if not (Path(plan['latest_baseline'])/'COMPLETE.json').exists():
        raise RuntimeError('Baseline is incomplete')
    assets=Path(plan['assets']);known,reserved,_=prior_spend(assets,root)
    budget=Budget(root)
    try:
        if Decimal(budget.state['initial_spent_usd'])!=known+reserved:
            raise RuntimeError('Other project reproduction charges changed')
        old=budget.state.copy(); committed=budget.committed();limit=Decimal('65.00')
        if committed+600*Decimal('.10')>limit:
            raise RuntimeError('Ceiling no longer covers 600 worst-case attempts')
        backup=root/'budget_before_full_limit_update.json'
        if backup.exists():
            raise RuntimeError('Budget update was already attempted; inspect receipt')
        atomic_json(backup,old)
        atomic_json(root/'budget_authorization.json',{
            'user_request':'三项自检全部通过 提交600次的实验 只提交 这个usage limit更新下 这个还是有钱跑的应该 保证能跑完',
            'old_limit_usd':old['limit_usd'],'new_limit_usd':str(limit),
            'committed_before_submission_usd':str(committed),
            'logical_new_vlm_calls':300,'max_attempts_per_call':2,'reserved_per_attempt_usd':'0.10',
            'conservative_total_bound_usd':str(committed+Decimal('60')),
            'forecast_new_usd':'0.5544','control_commit':control_commit,'unix':time.time()})
        budget.state['limit_usd']=str(limit);atomic_json(budget.path,budget.state)
    finally:
        budget.close()
    plan.update(api_limit_usd='65.00',orchestration_commit=control_commit,orchestration_source=str(SOURCE),
                full_walltime='48:00:00',reviewed_action_threshold=threshold)
    atomic_json(root/'plan.json',plan)
    env=os.environ.copy();env.update(CB_PAIRED_ROOT=str(root),CB_PAIRED_STAGE='full')
    job=subprocess.check_output(['sbatch','--parsable','--hold','--time=48:00:00',
        '--output='+str(root/'slurm_full_%j.log'),str(science/'experiments/paired_repeats/run_gpu.sbatch')],
        cwd=checkout,env=env,text=True).strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError('Invalid Slurm receipt')
    plan['jobs']['full']=job;atomic_json(root/'plan.json',plan)
    worker=shlex.join([PYTHON,'-u',str(SOURCE/'experiments/paired_repeats/api_worker.py'),str(root),'full'])+' > '+shlex.quote(str(root/'api_full.log'))+' 2>&1'
    try:
        exists=subprocess.run(['tmux','has-session','-t','=crashbench'],capture_output=True).returncode==0
        tmux=(['tmux','new-window','-d','-t','crashbench','-n','paired_full'] if exists else
              ['tmux','new-session','-d','-s','crashbench','-n','paired_full'])
        subprocess.run(tmux+['-c',str(checkout),worker],check=True)
        for _ in range(45):
            if (root/'STOP.json').exists():
                raise RuntimeError('Worker startup failed')
            if (root/'WORKER_READY_full.json').exists():
                break
            time.sleep(1)
        else:
            raise RuntimeError('Worker readiness timeout')
        subprocess.run(['scontrol','release',job],check=True)
    except BaseException as error:
        atomic_json(root/'STOP.json',{'stage':'full_submission','reason':str(error),'unix':time.time()})
        subprocess.run(['scancel',job],check=False)
        raise
    receipt={'job':job,'runs':600,'fresh_vlm_calls':300,'science_commit':plan['code_commit'],
             'orchestration_commit':control_commit,'limit_usd':'65.00','walltime':'48:00:00','root':str(root)}
    atomic_json(root/'FULL_SUBMITTED.json',receipt);print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);submit(p.parse_args().root)
