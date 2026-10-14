"""Launch exact-commit collection stages from the verified project checkout."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
from .inventory import ASSETS, sha, write


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def main(stage):
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
        api_calls=0, api_limit_usd='20.00', account='p33100', partition='short')
    write(root/'CAMPAIGN.json', manifest)
    env = dict(os.environ, CB_FD_ROOT=str(root), CB_FD_STAGE=stage)
    job = subprocess.check_output(['sbatch', '--parsable', '--output='+str(root/'slurm_%j.log'),
        str(root/'source/verifier_dataset/cpu.sbatch')], env=env, text=True).strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError('Invalid Slurm receipt')
    receipt = dict(manifest, job=job)
    write(root/'RECEIPT.json', receipt)
    local = repo/'results/verifier_dataset'/root.name; local.mkdir(parents=True, exist_ok=False)
    write(local/'RECEIPT.json', receipt)
    print(receipt, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['inventory'])
    main(p.parse_args().stage)
