"""Launch HIL suite stages from the clean, published project checkout.

smoke: three prefixes and their branches in one draft run on one A100 (protocol section 9).
prefix: the formal prefixes of stages 0-2 on two A100 shards, then the CPU prefix summary (stage 0 gate).
branch PREFIX_ROOT: the formal branches of a finished, env_ok prefix run on two A100 shards, then the CPU summary.
--policy pi05_libero (default) or pi0_libero picks the openpi checkpoint (stage 4).
"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
from .inventory import ASSETS, read, sha, write
from .v5_launch import UPSTREAM, command, prompt_tokens, submit
from . import hil_suite as S
from .hil_suite_run import LIBERO_COMMIT

PROTOCOL = 'docs/generative_verifier/HIL_SUITE_PROTOCOL.md'
AUTHORIZATION = ('2026-10-29 late: the user asked for a more concrete analysis of VLA robustness to human-in-the-loop interventions with more '
                 'scenarios and tasks ("帮我做一下更具体的分析吧 ... 多弄点场景任务啥的"), then approved the whole plan (HIL_SUITE_PLAN.md): '
                 '"都可以 同意 干吧" (design, readings, every stage with its GPU smoke and formal batch, pi0 and OpenVLA-OFT); '
                 'policies alone, no verifier, no API, at most two GPUs')
DETAIL = dict(smoke='smoke: 3 prefixes and their branches in a draft run, results only in protocol section 9',
              prefix='formal: stage 0-2 prefixes after the protocol is frozen', branch='formal: stage 1-2 branches of an env_ok prefix run')


def prepare(stage, policy, repo):
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
    libero = repo/'third_party/LIBERO'
    if command('git', '-C', str(libero), 'rev-parse', 'HEAD') != LIBERO_COMMIT or command('git', '-C', str(libero), 'status', '--porcelain', '--untracked-files=no'):
        raise RuntimeError('LIBERO checkout changed')
    root = ASSETS/'hil_suite'/('%s_%s_%s_%s' % (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'), stage, policy, commit[:12]))
    root.mkdir(parents=True, exist_ok=False); (root/'source').mkdir()
    with (root/'source.tar').open('wb') as stream:
        subprocess.run(['git', 'archive', commit], stdout=stream, check=True)
    subprocess.run(['tar', '-xf', str(root/'source.tar'), '-C', str(root/'source')], check=True)
    (root/'SOURCE_COMMIT').write_text(commit + '\n')
    (root/'source.sha256').write_text(sha(root/'source.tar') + '  ' + str(root/'source.tar') + '\n')
    manifest = dict(stage='hil-suite-' + stage, run_root=str(root), code_commit=commit, upstream_commit=UPSTREAM, libero_commit=LIBERO_COMMIT,
                    policy=policy, policy_checkpoint=str(ASSETS/'openpi_assets/openpi-assets/checkpoints'/policy),
                    seed='per prefix: hil_suite.policy_seed (salt %s), shared by every branch of the prefix' % S.SEED_SALT,
                    env_seed=S.ENV_SEED, protocol=PROTOCOL, protocol_sha256=sha(repo/PROTOCOL), authorization=AUTHORIZATION + '; ' + DETAIL[stage],
                    api_calls=0, account='p33100')
    return root, manifest


def gpu_room(shards):
    active = command('squeue', '-h', '-u', os.environ['USER'], '-A', 'p33100', '-p', 'gengpu', '-o', '%j')
    if sum(x.startswith('cb_') for x in active.splitlines()) + shards > 2:
        raise RuntimeError('At most two CrashBench GPUs at a time')


def main(stage, policy, prefix_root=None):
    repo = Path.cwd().resolve(); names = S.task_names((repo/'third_party/LIBERO/libero/libero/benchmark/libero_suite_task_map.py').read_text())
    if stage != 'smoke' and '状态：已冻结' not in (repo/PROTOCOL).read_text():
        raise RuntimeError('Freeze the protocol before a formal batch')
    shards = 1 if stage == 'smoke' else 2
    if stage == 'branch':
        done = read(prefix_root/'PREFIX_COMPLETE.json')
        if not done['passed'] or done['mode'] != 'formal' or read(prefix_root/'CAMPAIGN.json')['policy'] != policy:
            raise RuntimeError('Branches need a finished formal prefix run of the same policy')
        if read(prefix_root/'STAGE0.json')['reading'] != 'env_ok':
            raise RuntimeError('Stage 0 read env_suspect: stop and check the environment')
        for n, digest in read(prefix_root/'PREFIX_HASHES.json').items():
            if sha(prefix_root/n) != digest:
                raise RuntimeError('Prefix run output changed: ' + n)
        prefix_plan = read(prefix_root/'PLAN.json'); units, excluded = S.branch_plan(names, read(prefix_root/'PREFIXES.json'), 'formal')
        plan = dict(mode='formal', shards=shards, prefix_root=str(prefix_root), prefix_hashes=read(prefix_root/'PREFIX_HASHES.json'),
                    prefix_order=prefix_plan['prefixes'], branches=units, excluded=excluded)
    else:
        prefixes = S.prefix_plan(names, 'smoke' if stage == 'smoke' else 'formal')
        plan = dict(mode='smoke' if stage == 'smoke' else 'formal', shards=shards, prefixes=prefixes)
    texts = [u['prompt'] for u in plan.get('branches', plan.get('prefixes'))] + ([S.STOP, S.RETURN] if stage == 'smoke' else [])
    plan['prompt_tokens'] = prompt_tokens(texts)
    if max(plan['prompt_tokens'].values()) >= S.MAX_TOKENS:
        raise RuntimeError('A prompt would be truncated')
    gpu_room(shards)
    root, manifest = prepare(stage, policy, repo)
    write(root/'NAMES.json', names)
    manifest.update(partition='gengpu', mode=plan['mode'], shards=shards, prefixes=len(plan.get('prefixes', [])),
                    branches=len(plan.get('branches', [])), excluded=len(plan.get('excluded', [])))
    if prefix_root is not None:
        manifest.update(prefix_root=str(prefix_root), prefix_hashes=plan['prefix_hashes'])
    write(root/'CAMPAIGN.json', manifest); write(root/'PLAN.json', plan)
    env = dict(os.environ, CB_FD_ROOT=str(root), CB_POLICY_CONFIG=policy, CB_SUITE_SHARDS=str(shards),
               CB_SUITE_PHASE='smoke' if stage == 'smoke' else stage)
    script = str(root/'source/verifier_dataset/hil_suite.sbatch')
    if stage == 'smoke':
        receipt = dict(manifest, job=submit(env, '--time=03:00:00', '--output=' + str(root/'slurm_%j.log'), script))
    else:
        job = submit(env, '--array=0-%d' % (shards - 1), '--time=%s' % ('04:00:00' if stage == 'prefix' else '12:00:00'),
                     '--output=' + str(root/'slurm_%A_%a.log'), script)
        env['CB_SUITE_SUMMARY'] = 'summarize-prefix' if stage == 'prefix' else 'summarize'
        receipt = dict(manifest, job=job, summarize_job=submit(env, '--dependency=afterok:' + job, '--output=' + str(root/'slurm_%j.log'),
                                                               str(root/'source/verifier_dataset/hil_suite_summary.sbatch')))
    write(root/'RECEIPT.json', receipt)
    local = repo/'results/hil_suite'/root.name; local.mkdir(parents=True, exist_ok=False)
    write(local/'RECEIPT.json', receipt)
    print(receipt, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['smoke', 'prefix', 'branch'])
    p.add_argument('prefix_root', type=Path, nargs='?')
    p.add_argument('--policy', default='pi05_libero', choices=['pi05_libero', 'pi0_libero'])
    a = p.parse_args()
    if (a.stage == 'branch') != (a.prefix_root is not None):
        raise SystemExit('branch needs exactly one prefix run root; smoke and prefix take none')
    main(a.stage, a.policy, a.prefix_root)
