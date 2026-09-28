"""Launch only from a published Git archive; leave the active checkout unchanged."""
import argparse
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
from api_budget import atomic_json
from api_worker import active_jobs

ASSETS = Path('/projects/p33100/siosio/crashbench_safelibero')
PYTHON = '/projects/p33100/siosio/envs/openpi/bin/python'

def cmd(args):
    return subprocess.check_output(args, text=True, cwd=Path.home()/'crash_bench').strip()

def launch(root, stage):
    if SOURCE != root/'source' or root.parent.resolve() != (ASSETS/'paired_repeats').resolve():
        raise RuntimeError('Source must be the isolated archive within the new experiment root')
    checkout = Path.home()/'crash_bench'
    if cmd(['git','remote','get-url','origin']) not in ['git@github.com:hanshuo-shuo/crash_bench.git','https://github.com/hanshuo-shuo/crash_bench.git']:
        raise RuntimeError('Wrong project')
    if cmd(['git','status','--porcelain=v1','--untracked-files=all']):
        raise RuntimeError('Live checkout is unexpectedly dirty')
    commit = (root/'SOURCE_COMMIT').read_text().strip()
    if cmd(['git','rev-parse','origin/codex/paired-repeats']) != commit:
        raise RuntimeError('Archive commit is not published on the paired branch')
    # Verify all archive file bytes against Git, not merely against a local marker.
    files = cmd(['git','ls-tree','-r','--name-only',commit]).splitlines()
    checksums=[]
    for relative in files:
        expected = subprocess.check_output(['git','show',commit+':'+relative],cwd=checkout)
        actual = (SOURCE/relative).read_bytes()
        if expected != actual:
            raise RuntimeError('Archive bytes differ from published commit: '+relative)
        checksums.append(hashlib.sha256(actual).hexdigest()+'  '+relative)
    if stage == 'checks':
        if (root/'plan.json').exists():
            raise RuntimeError('Do not relaunch an existing checks root')
        for other in (ASSETS/'paired_repeats').glob('*/plan.json'):
            if other.parent == root:
                continue
            jobs = list(json.loads(other.read_text())['jobs'].values())
            if active_jobs(jobs):
                raise RuntimeError('Another paired experiment is active')
        baseline = sorted((ASSETS/'batches').glob('*/batch.json'))
        roots = [str(p.parent) for p in baseline]
        jobs = [j for p in baseline for j in json.loads(p.read_text())['slurm_arrays']]
        active = [j for j in jobs if active_jobs([j])]
        plan = {'code_commit':commit, 'assets':str(ASSETS), 'source':str(SOURCE), 'jobs':{},
                'baseline_roots':roots, 'baseline_jobs':active, 'latest_baseline':roots[-1],
                'live_checkout_commit_at_submission':cmd(['git','rev-parse','HEAD']),
                'design':'3 scenarios x 20 states x 5 repeats x 2 methods',
                'max_gpus':1, 'created_unix':time.time()}
        (root/'source.sha256').write_text('\n'.join(checksums)+'\n')
    else:
        from execute import review_gate
        review_gate(root)
        plan = json.loads((root/'plan.json').read_text())
        if 'full' in plan['jobs'] or (root/'STOP.json').exists():
            raise RuntimeError('Full stage already submitted or stopped')
        all_jobs = [j for p in plan['baseline_roots'] for j in json.loads((Path(p)/'batch.json').read_text())['slurm_arrays']]
        if active_jobs(all_jobs):
            raise RuntimeError('Reproduction GPU jobs still active')
        if not (Path(plan['latest_baseline'])/'COMPLETE.json').exists():
            raise RuntimeError('Original reproduction is not complete')
        active = []
    atomic_json(root/'plan.json',plan)
    env = os.environ.copy(); env.update(CB_PAIRED_ROOT=str(root),CB_PAIRED_STAGE=stage)
    args = ['sbatch','--parsable','--hold','--output='+str(root/('slurm_'+stage+'_%j.log'))]
    if active:
        args += ['--dependency=afterany:'+':'.join(active)]
    if stage == 'checks':
        args += ['--time=06:00:00']
    args += [str(SOURCE/'experiments/paired_repeats/run_gpu.sbatch')]
    job = subprocess.check_output(args,env=env,text=True,cwd=checkout).strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError('Invalid Slurm receipt')
    plan['jobs'][stage]=job; atomic_json(root/'plan.json',plan)
    worker = shlex.join([PYTHON,'-u',str(SOURCE/'experiments/paired_repeats/api_worker.py'),str(root),stage])+' > '+shlex.quote(str(root/('api_'+stage+'.log')))+' 2>&1'
    try:
        exists = subprocess.run(['tmux','has-session','-t','=crashbench'],capture_output=True).returncode == 0
        tmux = (['tmux','new-window','-d','-t','crashbench','-n','paired_'+stage] if exists else
                ['tmux','new-session','-d','-s','crashbench','-n','paired_'+stage])
        subprocess.run(tmux+['-c',str(checkout),worker],check=True)
        subprocess.run(['scontrol','release',job],check=True)
    except BaseException as error:
        atomic_json(root/'STOP.json',{'stage':'submission_'+stage,'reason':type(error).__name__+': '+str(error),'unix':time.time()})
        subprocess.run(['scancel',job],check=False)
        raise
    print(json.dumps({'root':str(root),'stage':stage,'job':job,'baseline_dependencies':active,'live_checkout_unchanged':cmd(['git','rev-parse','HEAD'])},indent=2))

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['checks','full']);p.add_argument('root',type=Path);a=p.parse_args()
    launch(a.root.resolve(),a.stage)
