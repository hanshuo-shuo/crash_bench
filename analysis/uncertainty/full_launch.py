"""Launch two published frozen shards, releasing shard1 only after stage smoke."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from common import ASSETS, atomic_json, config
from preflight import stream_sha


def launch(root, preflight):
    checkout=Path.home()/'crash_bench'; source=root/'source'; cfg=config('full')
    if Path(__file__).resolve().parents[2]!=source.resolve() or root.parent!=(ASSETS/'uncertainty').resolve():
        raise RuntimeError('Launch only from a fresh immutable published source archive')
    def command(args):return subprocess.check_output(args,cwd=checkout,text=True).strip()
    commit=(root/'SOURCE_COMMIT').read_text().strip()
    if command(['git','status','--porcelain','--untracked-files=all']):raise RuntimeError('Live checkout dirty')
    if command(['git','remote','get-url','origin']) not in ['git@github.com:hanshuo-shuo/crash_bench.git','https://github.com/hanshuo-shuo/crash_bench.git']:
        raise RuntimeError('Wrong repository identity')
    if command(['git','rev-parse','origin/codex/feasibility'])!=commit:raise RuntimeError('Source archive unpublished')
    if (root/'plan.json').exists():raise RuntimeError('Never resubmit an existing run root')
    fingerprints=json.loads((preflight/'FINGERPRINTS.json').read_text())
    resolution=json.loads((preflight/'GEOMETRY_RESOLUTION.json').read_text())
    if not fingerprints['passed'] or not resolution['passed']:raise RuntimeError('Preflight not accepted')
    if stream_sha(source/'analysis/uncertainty/geometry.py')!=stream_sha(preflight/'source/analysis/uncertainty/geometry.py'):
        raise RuntimeError('Distance implementation changed after preflight validation')
    preflight_job=json.loads((preflight/'SUBMITTED.json').read_text())['job']
    accounting=command(['sacct','-X','-n','-P','-j',preflight_job,'--format=JobIDRaw,State,ExitCode'])
    if accounting.splitlines()!=[preflight_job+'|COMPLETED|0:0']:raise RuntimeError('Preflight not confirmed successful')
    listing=command(['squeue','-h','-u',os.environ['USER'],'-A','p33100','-o','%i|%j'])
    if any('cb_uncertainty_full' in line for line in listing.splitlines()):raise RuntimeError('Full uncertainty job already active')
    entries=[]
    for name in command(['git','ls-tree','-r','--name-only',commit]).splitlines():
        if (source/name).read_bytes()!=subprocess.check_output(['git','show',commit+':'+name],cwd=checkout):
            raise RuntimeError('Source archive differs: '+name)
        entries.append(dict(path=name,sha256=stream_sha(source/name)))
    atomic_json(root/'SOURCE_FILES.json',entries)
    for name in ['FINGERPRINTS.json','GEOMETRY_AUDIT.json','GEOMETRY_RESOLUTION.json']:
        shutil.copyfile(preflight/name,root/name)
    campaign=root.parent/cfg['campaign'];campaign.mkdir(exist_ok=True)
    (root/'shards').mkdir();jobs=[]
    validation=dict(validation_input_root=str(root.parent/'20261008T035359Z_5e20522082f7'),
        validation_input_prompt='pick up the black bowl on the ramekin and place it on the plate')
    base=dict(code_commit=commit,live_checkout_commit=command(['git','rev-parse','HEAD']),
        campaign_root=str(campaign),configuration=cfg,upstream_root=fingerprints['upstream'],
        preflight=str(preflight),preflight_job=preflight_job,submitted_unix=time.time(),
        states_sha256=stream_sha(source/'analysis/uncertainty/states.json'),
        split_sha256=stream_sha(source/'analysis/uncertainty/split.json'),
        stats_plan_sha256=stream_sha(source/'analysis/uncertainty/STATS_PLAN.md'),
        expected_rollouts=1200,requested_gpu_hours=48,
        authority='User2026-10-09 approved2A100×24h and completing the series; API dollar cap removed',**validation)
    try:
        for i in [0,1]:
            shard=root/'shards'/str(i);shard.mkdir()
            env=dict(os.environ,CB_UNCERTAINTY_ROOT=str(shard))
            job=subprocess.check_output(['sbatch','--parsable','--hold','--output='+str(shard/'slurm_%j.log'),
                str(source/'analysis/uncertainty/full.sbatch')],env=env,cwd=checkout,text=True).strip().split(';')[0]
            if not job.isdigit():raise RuntimeError('Invalid Slurm receipt')
            jobs.append(job);atomic_json(shard/'plan.json',dict(base,job=job,shard=i,expected_rollouts=600))
            atomic_json(root/'plan.json',dict(base,jobs=jobs))
        worker=shlex.join(['/projects/p33100/siosio/envs/openpi/bin/python','-u',
            str(source/'analysis/uncertainty/full_worker.py'),str(root)])+' > '+shlex.quote(str(root/'api_worker.log'))+' 2>&1'
        exists=subprocess.run(['tmux','has-session','-t','=crashbench'],capture_output=True).returncode==0
        tmux=['tmux','new-window','-d','-t','crashbench','-n','uncertainty_full'] if exists else ['tmux','new-session','-d','-s','crashbench','-n','uncertainty_full']
        subprocess.run(tmux+['-c',str(checkout),worker],check=True)
        deadline=time.monotonic()+60
        while not (root/'WORKER_READY.json').exists():
            if (root/'STOP.json').exists() or time.monotonic()>deadline:raise RuntimeError('API worker readiness failed')
            time.sleep(1)
        subprocess.run(['scontrol','release',jobs[0]],check=True)
        atomic_json(root/'SUBMITTED.json',dict(base,jobs=jobs,second_shard_held_until_stage_smoke=True))
    except BaseException as error:
        atomic_json(root/'STOP.json',dict(stage='launch',reason=str(error),jobs=jobs))
        for job in jobs:subprocess.run(['scancel',job],check=False)
        raise
    print(json.dumps(json.loads((root/'SUBMITTED.json').read_text()),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--preflight',type=Path,required=True)
    a=p.parse_args();launch(a.root.resolve(),a.preflight.resolve())
