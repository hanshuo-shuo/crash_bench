"""Launch HIL suite stages from the clean, published project checkout.

smoke: three prefixes and their branches in one draft run on one A100 (protocol section 9).
prefix: the formal prefixes of stages 0-2 on two A100 shards, then the CPU prefix summary (stage 0 gate).
branch PREFIX_ROOT: the formal branches of a finished, env_ok prefix run on six shards, two A100s at a time, then the CPU summary.
physical-smoke / physical PREFIX_ROOT BRANCH_ROOT: stage 3 perturbations (C0 baselines from the stage 1-2 branch run).
probe-smoke / probe PREFIX_ROOT BRANCH_ROOT: stage 5 action-sensitivity probe along the stage-1 prefixes.
--policy pi05_libero (default), pi0_libero or openvla_oft picks the policy (stage 4).
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
              prefix='formal: stage 0-2 prefixes after the protocol is frozen', branch='formal: stage 1-2 branches of an env_ok prefix run',
              **{'physical-smoke': 'stage 3 smoke: 2 tasks x 1 state x 4 perturbations in a draft run, results only in appendix A',
                 'physical': 'stage 3 formal: 8 tasks x 10 states x 4 perturbations after appendix A is frozen',
                 'probe-smoke': 'stage 5 smoke: 2 prefixes in a draft run, results only in appendix B',
                 'probe': 'stage 5 formal: every stage-1a/1b prefix after appendix B is frozen'})
OFT_CHECKPOINT = '/projects/p33100/siosio/openvla_checkpoints/openvla-7b-oft-finetuned-libero-spatial-object-goal-10@638918f3d1c2'
OFT_COMMIT = 'e4287e94541f459edc4feabc4e181f537cd569a8'
# gs://openpi-assets/checkpoints/pi0_libero as fetched on 2026-07-07; its 19 files match the bucket's sizes and MD5s (2026-10-30).
PI0_CHECKPOINT = '/projects/p33100/siosio/openpi_assets/openpi-assets/checkpoints/pi0_libero'


def checkpoint_path(policy):
    return dict(openvla_oft=OFT_CHECKPOINT, pi0_libero=PI0_CHECKPOINT).get(policy, str(ASSETS/'openpi_assets/openpi-assets/checkpoints'/policy))


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
                    policy=policy, policy_checkpoint=checkpoint_path(policy), replan_steps=S.REPLANS[policy],
                    oft_commit=OFT_COMMIT if policy == 'openvla_oft' else None,
                    seed='per prefix: hil_suite.policy_seed (salt %s), shared by every branch of the prefix' % S.SEED_SALT,
                    env_seed=S.ENV_SEED, protocol=PROTOCOL, protocol_sha256=sha(repo/PROTOCOL), authorization=AUTHORIZATION + '; ' + DETAIL[stage],
                    api_calls=0, account='p33100')
    return root, manifest


def gpu_room(shards):
    active = command('squeue', '-h', '-u', os.environ['USER'], '-A', 'p33100', '-p', 'gengpu', '-o', '%j')
    if sum(x.startswith('cb_') for x in active.splitlines()) + shards > 2:
        raise RuntimeError('At most two CrashBench GPUs at a time')


def finished_prefix(prefix_root, policy):
    done = read(prefix_root/'PREFIX_COMPLETE.json')
    if not done['passed'] or done['mode'] != 'formal' or read(prefix_root/'CAMPAIGN.json')['policy'] != policy:
        raise RuntimeError('Branches need a finished formal prefix run of the same policy')
    if read(prefix_root/'STAGE0.json')['reading'] != 'env_ok':
        raise RuntimeError('Stage 0 read env_suspect: stop and check the environment')
    for n, digest in read(prefix_root/'PREFIX_HASHES.json').items():
        if sha(prefix_root/n) != digest:
            raise RuntimeError('Prefix run output changed: ' + n)


def finished_branch(branch_root, prefix_root):
    done = read(branch_root/'SUITE_COMPLETE.json'); plan = read(branch_root/'PLAN.json')
    if not done['passed'] or plan['mode'] != 'formal' or plan.get('experiment', 'switch') != 'switch' or Path(plan['prefix_root']) != prefix_root:
        raise RuntimeError('Stages 3 and 5 need the finished formal stage 1-2 branch run of this prefix run')
    for n, digest in read(branch_root/'HASHES.json').items():
        if sha(branch_root/n) != digest:
            raise RuntimeError('Branch run output changed: ' + n)
    return read(branch_root/'HASHES.json')


FROZEN = dict(smoke=None, prefix='状态：已冻结', branch='状态：已冻结', **{'physical-smoke': None, 'physical': '附录 A 状态：已冻结',
              'probe-smoke': None, 'probe': '附录 B 状态：已冻结'})


def main(stage, policy, prefix_root=None, branch_root=None):
    repo = Path.cwd().resolve(); names = S.task_names((repo/'third_party/LIBERO/libero/libero/benchmark/libero_suite_task_map.py').read_text())
    if FROZEN[stage] is not None and FROZEN[stage] not in (repo/PROTOCOL).read_text():
        raise RuntimeError('Freeze the protocol before a formal batch')
    # Six branch shards keep the slowest node's worst case (27 h of branches on two shards) under the 12 h limit.
    shards = 1 if stage.endswith('smoke') else 6 if stage == 'branch' else 2
    if stage.startswith(('physical', 'probe')):
        finished_prefix(prefix_root, policy); hashes = finished_branch(branch_root, prefix_root)
        prefixes = read(prefix_root/'PREFIXES.json'); mode = 'smoke' if stage.endswith('smoke') else 'formal'
        common = dict(mode=mode, shards=shards, prefix_root=str(prefix_root), prefix_hashes=read(prefix_root/'PREFIX_HASHES.json'),
                      prefix_order=read(prefix_root/'PLAN.json')['prefixes'])
        if stage.startswith('physical'):
            units, excluded = S.stage3_plan(prefixes, mode)
            plan = dict(common, experiment='physical', baseline_root=str(branch_root), baseline_hashes=hashes, branches=units, excluded=excluded)
        else:
            probes = [dict(p, group=g) for p in prefixes for g in ['1a' if p['suite'] == 'libero_goal' else '1b']
                      if p['success'] and p['others'] and p['suite'] in ('libero_goal', 'libero_10')]
            plan = dict(common, experiment='probe', branch_root=str(branch_root), branch_hashes=hashes, probes=probes[:2] if mode == 'smoke' else probes)
    elif stage == 'branch':
        finished_prefix(prefix_root, policy)
        prefix_plan = read(prefix_root/'PLAN.json'); units, excluded = S.branch_plan(names, read(prefix_root/'PREFIXES.json'), 'formal')
        plan = dict(mode='formal', shards=shards, prefix_root=str(prefix_root), prefix_hashes=read(prefix_root/'PREFIX_HASHES.json'),
                    prefix_order=prefix_plan['prefixes'], branches=units, excluded=excluded)
    else:
        prefixes = S.prefix_plan(names, 'smoke' if stage == 'smoke' else 'formal')
        plan = dict(mode='smoke' if stage == 'smoke' else 'formal', shards=shards, prefixes=prefixes)
    texts = [u['prompt'] for u in plan.get('branches', plan.get('prefixes', plan.get('probes')))] + [S.STOP, S.RETURN]
    # Prompt lengths in openpi's PaliGemma tokenizer against the policy's max_token_len (pi0.5 200, pi0 48); OFT's Llama
    # prompt has no such cap in its LIBERO evaluation.
    plan['prompt_tokens'] = None if policy == 'openvla_oft' else prompt_tokens(texts)
    if plan['prompt_tokens'] and max(plan['prompt_tokens'].values()) >= dict(pi0_libero=48).get(policy, S.MAX_TOKENS):
        raise RuntimeError('A prompt would be truncated')
    gpu_room(min(shards, 2))
    root, manifest = prepare(stage, policy, repo)
    write(root/'NAMES.json', names)
    manifest.update(partition='gengpu', mode=plan['mode'], shards=shards, prefixes=len(plan.get('prefixes', [])),
                    branches=len(plan.get('branches', [])), excluded=len(plan.get('excluded', [])))
    if prefix_root is not None:
        manifest.update(prefix_root=str(prefix_root), prefix_hashes=plan['prefix_hashes'])
    if branch_root is not None:
        manifest.update(branch_root=str(branch_root), experiment=plan['experiment'], probes=len(plan.get('probes', [])))
    write(root/'CAMPAIGN.json', manifest); write(root/'PLAN.json', plan)
    phase = dict(smoke='smoke', prefix='prefix', branch='branch').get(stage, 'probe' if stage.startswith('probe') else 'branch')
    env = dict(os.environ, CB_FD_ROOT=str(root), CB_POLICY_CONFIG=policy, CB_POLICY_CHECKPOINT=checkpoint_path(policy), CB_SUITE_SHARDS=str(shards),
               CB_SUITE_PHASE=phase)
    script = str(root/'source/verifier_dataset/hil_suite.sbatch')
    if stage == 'smoke':
        receipt = dict(manifest, job=submit(env, '--time=03:00:00', '--output=' + str(root/'slurm_%j.log'), script))
    else:
        hours = dict(prefix='04', branch='12', physical='06', probe='04').get(stage, '03')
        job = submit(env, '--array=0-%d%%2' % (shards - 1), '--time=%s:00:00' % hours, '--output=' + str(root/'slurm_%A_%a.log'), script)
        env['CB_SUITE_SUMMARY'] = 'summarize-prefix' if stage == 'prefix' else 'summarize'
        receipt = dict(manifest, job=job, summarize_job=submit(env, '--dependency=afterok:' + job, '--output=' + str(root/'slurm_%j.log'),
                                                               str(root/'source/verifier_dataset/hil_suite_summary.sbatch')))
    write(root/'RECEIPT.json', receipt)
    local = repo/'results/hil_suite'/root.name; local.mkdir(parents=True, exist_ok=False)
    write(local/'RECEIPT.json', receipt)
    print(receipt, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=list(FROZEN))
    p.add_argument('roots', type=Path, nargs='*')
    p.add_argument('--policy', default='pi05_libero', choices=list(S.POLICIES))
    a = p.parse_args()
    need = 0 if a.stage in ('smoke', 'prefix') else 1 if a.stage == 'branch' else 2
    if len(a.roots) != need:
        raise SystemExit('%s takes %d run roots' % (a.stage, need))
    # Keep the /projects spelling: the container binds /projects, not the /gpfs/projects target of its symlink.
    roots = [Path(os.path.abspath(r)) for r in a.roots]
    if any(ASSETS/'hil_suite' not in r.parents for r in roots):
        raise SystemExit('Run roots must be given under ' + str(ASSETS/'hil_suite'))
    main(a.stage, a.policy, *roots)
