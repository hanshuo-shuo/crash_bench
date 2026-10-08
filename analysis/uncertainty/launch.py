"""Launch the one authorized immutable 30-minute job, never change the live checkout."""
import argparse
from pathlib import Path
import os
import shlex
import subprocess
import time
from common import ASSETS, atomic_json, config, sha


def command(args, checkout):
    return subprocess.check_output(args,cwd=checkout,text=True).strip()


def launch(root):
    checkout=Path.home()/'crash_bench'; source=root/'source'; cfg=config()
    if Path(__file__).resolve().parents[2] != source.resolve() or root.parent != (ASSETS/'uncertainty').resolve():
        raise RuntimeError('Launch only from the new immutable uncertainty source archive')
    if command(['git','status','--porcelain=v1','--untracked-files=all'],checkout): raise RuntimeError('Live checkout is dirty')
    if command(['git','remote','get-url','origin'],checkout) not in ['git@github.com:hanshuo-shuo/crash_bench.git','https://github.com/hanshuo-shuo/crash_bench.git']:
        raise RuntimeError('Wrong project checkout')
    commit=(root/'SOURCE_COMMIT').read_text().strip()
    if command(['git','rev-parse','origin/codex/feasibility'],checkout)!=commit: raise RuntimeError('Archive not published')
    if (root/'plan.json').exists(): raise RuntimeError('Never resubmit a run root')
    files=command(['git','ls-tree','-r','--name-only',commit],checkout).splitlines(); hashes=[]
    for name in files:
        expected=subprocess.check_output(['git','show',commit+':'+name],cwd=checkout)
        if (source/name).read_bytes()!=expected: raise RuntimeError('Source archive differs: '+name)
        hashes.append(sha(source/name)+'  '+name)
    (root/'source.sha256').write_text('\n'.join(hashes)+'\n')
    campaign=(root.parent/cfg['campaign']).resolve(); campaign.mkdir(exist_ok=True)
    # No other project jobs are inspected or changed; refuse an already active uncertainty job.
    listing=command(['squeue','-h','-u',os.environ['USER'],'-A','p33100','-o','%i|%j'],checkout)
    if any('cb_uncertainty_smoke' in x for x in listing.splitlines()): raise RuntimeError('Uncertainty job already active')
    env=os.environ.copy();env['CB_UNCERTAINTY_ROOT']=str(root)
    job=subprocess.check_output(
        ['sbatch','--parsable','--hold','--output='+str(root/'slurm_%j.log'),str(source/'analysis/uncertainty/run.sbatch')],env=env,cwd=checkout,text=True).strip().split(';')[0]
    if not job.isdigit(): raise RuntimeError('Invalid Slurm receipt')
    plan=dict(job=job,code_commit=commit,live_checkout_commit=command(['git','rev-parse','HEAD'],checkout),
              campaign_root=str(campaign),configuration=cfg,submitted_unix=time.time())
    parent = ASSETS/'uncertainty/20261008T035359Z_5e20522082f7'
    if (parent/'SAMPLING_VALIDATION.json').exists():
        import json
        failed=json.loads((parent/'SAMPLING_VALIDATION.json').read_text())
        if failed['serial_batch_passed'] or not (parent/'STOP.json').exists():
            raise RuntimeError('Expected retained failed B=8 numeric-gate evidence')
        plan.update(validation_input_root=str(parent), validation_input_prompt='pick up the black bowl on the ramekin and place it on the plate')
    atomic_json(root/'plan.json',plan)
    worker=shlex.join(['/projects/p33100/siosio/envs/openpi/bin/python','-u',str(source/'analysis/uncertainty/worker.py'),str(root)])+' > '+shlex.quote(str(root/'api_worker.log'))+' 2>&1'
    try:
        exists=subprocess.run(['tmux','has-session','-t','=crashbench'],capture_output=True).returncode==0
        tmux=['tmux','new-window','-d','-t','crashbench','-n','uncertainty'] if exists else ['tmux','new-session','-d','-s','crashbench','-n','uncertainty']
        subprocess.run(tmux+['-c',str(checkout),worker],check=True)
        deadline=time.monotonic()+60
        while not (root/'WORKER_READY.json').exists():
            if (root/'STOP.json').exists() or time.monotonic()>deadline: raise RuntimeError('API worker readiness failed')
            time.sleep(1)
        subprocess.run(['scontrol','release',job],check=True)
        atomic_json(root/'SUBMITTED.json',plan)
    except BaseException as error:
        atomic_json(root/'STOP.json',dict(stage='launch',reason=str(error)));subprocess.run(['scancel',job],check=False);raise
    print(__import__('json').dumps(plan,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();launch(a.root.resolve())
