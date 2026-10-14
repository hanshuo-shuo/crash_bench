"""Launch exact-commit collection stages from the verified project checkout."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from .inventory import ASSETS, sha, write, read


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def main(stage, inventory=None, canonicalize=False, limit=12, source_plan=None):
    repo = Path.cwd().resolve()
    if repo != (Path.home()/'crash_bench').resolve():
        raise RuntimeError('Wrong project checkout')
    if command('git', 'status', '--porcelain', '--untracked-files=all'):
        raise RuntimeError('Dirty source')
    if command('git', 'remote', 'get-url', 'origin') not in (
            'git@github.com:hanshuo-shuo/crash_bench.git', 'https://github.com/hanshuo-shuo/crash_bench.git'):
        raise RuntimeError('Wrong repository identity')
    commit = command('git', 'rev-parse', 'HEAD')
    if 'origin/' not in command('git', 'branch', '-r', '--contains', commit):
        raise RuntimeError('Source must be published')
    if stage == 'replay-smoke':
        if not 1<=limit<=12: raise RuntimeError('Smoke limit must be 1..12')
        if inventory is None or not read(inventory/'inventory/SUMMARY.json')['passed']:
            raise RuntimeError('Passed inventory required')
        inventory_job=read(inventory/'RECEIPT.json')['job']
        accounting=command('sacct','-X','-n','-P','-j',inventory_job,'--format=JobIDRaw,State,ExitCode')
        if accounting.splitlines()!=[inventory_job+'|COMPLETED|0:0']:
            raise RuntimeError('Inventory must be terminal COMPLETED/0:0')
        hashes=read(inventory/'inventory/HASHES.json')
        for name, expected in hashes.items():
            if sha(inventory/'inventory'/name)!=expected: raise RuntimeError('Inventory hash changed')
    if stage=='collect-smoke':
        if source_plan is None or not read(source_plan/'source_plan/SUMMARY.json')['passed']:
            raise RuntimeError('Passed broader-source inventory required')
        source_job=read(source_plan/'RECEIPT.json')['job']
        if command('sacct','-X','-n','-P','-j',source_job,'--format=JobIDRaw,State,ExitCode').splitlines()!=[source_job+'|COMPLETED|0:0']:
            raise RuntimeError('Source inventory must be terminal COMPLETED/0:0')
        for name,expected in read(source_plan/'source_plan/HASHES.json').items():
            if sha(source_plan/'source_plan'/name)!=expected:raise RuntimeError('Broader source manifest changed')
        active=command('squeue','-h','-u',os.environ['USER'],'-A','p33100','-p','gengpu','-o','%j|%Z')
        if sum('crash_bench' in x or 'crashbench_safelibero' in x or x.startswith('cb_') for x in active.splitlines())>=2:
            raise RuntimeError('Two CrashBench GPU jobs already active')
    root = ASSETS/'verifier_dataset'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+stage+'_'+commit[:12])
    root.mkdir(parents=True, exist_ok=False); (root/'source').mkdir()
    with (root/'source.tar').open('wb') as stream:
        subprocess.run(['git', 'archive', commit], stdout=stream, check=True)
    subprocess.run(['tar', '-xf', str(root/'source.tar'), '-C', str(root/'source')], check=True)
    (root/'SOURCE_COMMIT').write_text(commit+'\n')
    (root/'source.sha256').write_text(sha(root/'source.tar')+'  '+str(root/'source.tar')+'\n')
    manifest = dict(stage=stage, run_root=str(root), code_commit=commit,
        upstream_commit='2457feed5968ae803926e178c8ce8243b9ecdcf9', seed=1014,
        protocol_sha256=sha(repo/'verifier_dataset/protocol.json'),
        authorization='2026-10-14 user: 1000 evidence-backed rows on Quest; train only after audit; API cap $20',
        api_calls=0, api_limit_usd='20.00', account='p33100', partition='short',canonicalize=canonicalize,smoke_limit=limit)
    write(root/'CAMPAIGN.json', manifest)
    if stage == 'replay-smoke':
        write(root/'CASES.json',read(inventory/'inventory/REPLAY_SMOKE.json')[:limit])
        write(root/'INHERITED.json',dict(inventory=str(inventory),hashes=hashes))
    if stage=='collect-smoke':
        entries=read(source_plan/'source_plan/GOAL20_CANDIDATES.json')
        chosen=[next(r for r in entries if r['scene']['task']==task) for task in range(4)]
        if len({r['source_id'] for r in chosen})!=4:raise RuntimeError('Four unique Goal source states required')
        write(root/'COLLECTION_SOURCES.json',chosen)
        sys.path.insert(0,str(repo/'analysis/uncertainty'))
        from common import config
        cfg=config();cfg['save_diagnostic_samples']=False
        write(root/'plan.json',dict(configuration=cfg))
        manifest.update(partition='gengpu',budget_root=str(ASSETS/'verifier_dataset/budget_fd1000_20261014'),
            source_plan=str(source_plan),max_logical_requests=4,max_paid_attempts=8)
        write(root/'CAMPAIGN.json',manifest)
    env = dict(os.environ, CB_FD_ROOT=str(root), CB_FD_STAGE=stage)
    script='collect.sbatch' if stage=='collect-smoke' else ('cpu.sbatch' if stage in ('inventory','source-plan') else 'replay.sbatch')
    submit=['sbatch','--parsable','--output='+str(root/'slurm_%j.log')]
    if stage=='collect-smoke':submit.append('--hold')
    job = subprocess.check_output(submit+[str(root/'source/verifier_dataset'/script)],env=env,text=True).strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError('Invalid Slurm receipt')
    receipt = dict(manifest, job=job)
    write(root/'RECEIPT.json', receipt)
    local = repo/'results/verifier_dataset'/root.name; local.mkdir(parents=True, exist_ok=False)
    write(local/'RECEIPT.json', receipt)
    print(receipt, flush=True)
    if stage=='collect-smoke':
        try:
            worker=shlex.join(['/projects/p33100/siosio/envs/openpi/bin/python','-u','-m','verifier_dataset.api_worker',str(root)])
            worker+=' > '+shlex.quote(str(root/'api_worker.log'))+' 2>&1'
            exists=subprocess.run(['tmux','has-session','-t','=crashbench'],capture_output=True).returncode==0
            tmux=['tmux','new-window','-d','-t','crashbench','-n','fd1000_api'] if exists else ['tmux','new-session','-d','-s','crashbench','-n','fd1000_api']
            subprocess.run(tmux+['-c',str(root/'source'),worker],check=True)
            deadline=time.monotonic()+55
            while not (root/'WORKER_READY.json').exists():
                if (root/'STOP.json').exists() or time.monotonic()>deadline:raise RuntimeError('API worker readiness failed')
                time.sleep(1)
            subprocess.run(['scontrol','release',job],check=True)
            write(root/'RELEASED.json',dict(job=job,pricing_and_budget_ready=True))
        except BaseException as error:
            write(root/'STOP.json',dict(stage='launch',error=type(error).__name__,reason=str(error)))
            subprocess.run(['scancel',job],check=False)
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['inventory','source-plan','replay-smoke','collect-smoke'])
    p.add_argument('--inventory',type=Path);p.add_argument('--canonicalize',action='store_true')
    p.add_argument('--limit',type=int,default=12)
    p.add_argument('--source-plan',type=Path)
    a=p.parse_args();main(a.stage,a.inventory,a.canonicalize,a.limit,a.source_plan)
