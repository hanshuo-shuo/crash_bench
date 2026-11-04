"""Published launcher for the user-authorized pi0.5 random/empty instruction experiment."""
import argparse
import os
from pathlib import Path
from .inventory import ASSETS, read, sha, write
from .v5_launch import command, prompt_tokens, submit
from .hil_suite_launch import checkpoint_path, gpu_room, prepare
from . import hil_suite as S
from . import language_ablation as A

PROTOCOL = 'docs/generative_verifier/LANGUAGE_ABLATION_PROTOCOL.md'
AUTHORIZATION = ('2026-11-03 user requested a new experiment on existing tasks to test whether language matters: '
                 'replace task instructions with random generated A/B/C/D/E/F/+ strings and see the result '
                 '("在现有的任务上 ... 给它一些 random 的指令 ... 先做一个这样子的一个实验吧"). '
                 'pi0.5 alone, original/empty/random from t=0, unchanged standard LIBERO, no verifier or paid API.')


def main(mode, smoke_root=None):
    repo = Path.cwd().resolve()
    active = command('squeue', '-h', '-u', os.environ['USER'], '-o', '%j')
    if any(n.startswith('cb_') for n in active.splitlines()):
        raise RuntimeError('Finish existing CrashBench jobs before publishing/syncing and launching')
    names = S.task_names((repo/'third_party/LIBERO/libero/libero/benchmark/libero_suite_task_map.py').read_text())
    smoke_hashes = None
    if mode == 'formal':
        if '状态：已冻结' not in (repo/PROTOCOL).read_text():
            raise RuntimeError('Freeze the protocol after smoke')
        if smoke_root is None or ASSETS/'language_ablation' not in smoke_root.parents:
            raise RuntimeError('Formal run needs the finished language ablation smoke root')
        smoke_hashes = read(smoke_root/'HASHES.json')
        for n, digest in smoke_hashes.items():
            if sha(smoke_root/n) != digest:
                raise RuntimeError('Smoke output changed')
        complete = read(smoke_root/'COMPLETE.json'); old = read(smoke_root/'CAMPAIGN.json')
        if (not complete['passed'] or complete['mode'] != 'smoke' or complete['runs'] != 16
                or old['policy'] != 'pi05_libero' or old['protocol'] != PROTOCOL
                or read(smoke_root/'PLAN.json')['units'] != A.plan(names, 'smoke')):
            raise RuntimeError('Wrong or incomplete smoke')
    elif smoke_root is not None:
        raise RuntimeError('Smoke does not take an input root')
    units = A.plan(names, mode); shards = 1 if mode == 'smoke' else 2
    tokens = prompt_tokens([u['prompt'] for u in units])
    if max(tokens.values()) >= S.MAX_TOKENS:
        raise RuntimeError('Prompt truncation')
    gpu_room(shards)
    root, manifest = prepare('smoke' if mode == 'smoke' else 'prefix', 'pi05_libero', repo,
                             protocol=PROTOCOL, authorization=AUTHORIZATION, family='language_ablation')
    manifest.update(stage='language-ablation-' + mode, mode=mode, runs=len(units), states=len({u['group'] for u in units}),
                    shards=shards, partition='gengpu', smoke_root=None if smoke_root is None else str(smoke_root), smoke_hashes=smoke_hashes,
                    random_salt=A.RANDOM_SALT, random_alphabet=list(A.ALPHABET), random_symbols=A.SYMBOLS,
                    random_bank={str(i): A.random_prompt(i) for i in sorted({u['init'] for u in units})})
    plan = dict(mode=mode, shards=shards, units=units, prompt_tokens=tokens)
    write(root/'PLAN.json', plan); write(root/'NAMES.json', names); write(root/'CAMPAIGN.json', manifest)
    env = dict(os.environ, CB_FD_ROOT=str(root), CB_POLICY_CONFIG='pi05_libero', CB_POLICY_CHECKPOINT=checkpoint_path('pi05_libero'),
               CB_SUITE_SHARDS=str(shards), CB_SUITE_PHASE='smoke' if mode == 'smoke' else 'ablation',
               CB_SUITE_RUNNER='verifier_dataset.language_ablation_run')
    script = str(root/'source/verifier_dataset/hil_suite.sbatch')
    if mode == 'smoke':
        job = submit(env, '--job-name=cb_lang_smoke', '--time=01:00:00', '--output=' + str(root/'slurm_%j.log'), script)
        receipt = dict(manifest, job=job)
    else:
        job = submit(env, '--job-name=cb_lang', '--array=0-1%2', '--time=08:00:00', '--output=' + str(root/'slurm_%A_%a.log'), script)
        env['CB_SUITE_SUMMARY'] = 'summarize'
        receipt = dict(manifest, job=job, summarize_job=submit(env, '--job-name=cb_lang_summary', '--dependency=afterok:' + job,
                        '--output=' + str(root/'slurm_%j.log'), str(root/'source/verifier_dataset/hil_suite_summary.sbatch')))
    write(root/'RECEIPT.json', receipt)
    local = repo/'results/language_ablation'/root.name; local.mkdir(parents=True, exist_ok=False); write(local/'RECEIPT.json', receipt)
    (local/'REPORT.md').symlink_to(root/'REPORT.md')
    print(receipt, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['smoke', 'formal']); parser.add_argument('smoke_root', nargs='?', type=Path)
    args = parser.parse_args()
    main(args.mode, None if args.smoke_root is None else Path(os.path.abspath(args.smoke_root)))
