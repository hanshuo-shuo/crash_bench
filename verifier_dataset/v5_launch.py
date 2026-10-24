"""Launch the V5 step 1 geometry scan from the clean, published project checkout."""
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
from .inventory import ASSETS, sha, write
from .v5_scan import SHARDS, V4_ROOTS

UPSTREAM = '2457feed5968ae803926e178c8ce8243b9ecdcf9'
PROTOCOL = 'docs/generative_verifier/V5_STEP1_PROTOCOL.md'


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def main():
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
    if command('git', '-C', str(repo/'third_party/vlsa-aegis'), 'rev-parse', 'HEAD') != UPSTREAM:
        raise RuntimeError('Upstream checkout changed')
    for base in V4_ROOTS:
        if not read_passed(base):
            raise RuntimeError('V4 input root did not complete: ' + str(base))
    root = ASSETS/'verifier_dataset'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_v5-scan_'+commit[:12])
    root.mkdir(parents=True, exist_ok=False); (root/'source').mkdir()
    with (root/'source.tar').open('wb') as stream:
        subprocess.run(['git', 'archive', commit], stdout=stream, check=True)
    subprocess.run(['tar', '-xf', str(root/'source.tar'), '-C', str(root/'source')], check=True)
    (root/'SOURCE_COMMIT').write_text(commit+'\n')
    (root/'source.sha256').write_text(sha(root/'source.tar')+'  '+str(root/'source.tar')+'\n')
    manifest = dict(stage='v5-scan', run_root=str(root), code_commit=commit, upstream_commit=UPSTREAM, seed=1014,
        protocol=PROTOCOL, protocol_sha256=sha(repo/PROTOCOL),
        authorization='2026-10-24 user: V5 step 1 CPU geometry scan; no rendering, training, API or dataset rows',
        api_calls=0, account='p33100', partition='short', shards=len(SHARDS),
        v4_inputs={str(b): sha(b/'CONSTRAINT_COMPLETE.json') for b in V4_ROOTS})
    write(root/'CAMPAIGN.json', manifest)
    env = dict(os.environ, CB_FD_ROOT=str(root))
    script = str(root/'source/verifier_dataset/v5_scan.sbatch')
    job = command_env(env, 'sbatch', '--parsable', '--array=0-%d%%4' % (len(SHARDS)-1),
                      '--output='+str(root/'slurm_%A_%a.log'), script)
    summary = command_env(env, 'sbatch', '--parsable', '--dependency=afterok:'+job,
                          '--output='+str(root/'slurm_%j.log'), script)
    receipt = dict(manifest, job=job, summarize_job=summary)
    write(root/'RECEIPT.json', receipt)
    local = repo/'results/verifier_dataset'/root.name; local.mkdir(parents=True, exist_ok=False)
    write(local/'RECEIPT.json', receipt)
    print(receipt, flush=True)


def read_passed(base):
    from .inventory import read
    return read(base/'CONSTRAINT_COMPLETE.json').get('passed') is True


def command_env(env, *args):
    out = subprocess.check_output(args, env=env, text=True).strip().split(';')[0]
    if not out.isdigit():
        raise RuntimeError('Invalid Slurm receipt')
    return out


if __name__ == '__main__':
    main()
