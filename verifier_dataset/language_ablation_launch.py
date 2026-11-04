"""Published launcher: pipeline = smoke -> integrity gate -> formal array -> automatic report."""
import argparse
import os
from pathlib import Path
from .inventory import read, write
from .v5_launch import command, prompt_tokens, submit
from .hil_suite_launch import checkpoint_path, gpu_room, prepare
from .language_ablation_gate import validate
from . import hil_suite as S
from . import language_ablation as A

PROTOCOL = 'docs/generative_verifier/LANGUAGE_ABLATION_PROTOCOL.md'
AUTHORIZATION = ('2026-11-03 user requested a new experiment on existing tasks to test whether language matters: '
                 'replace task instructions with random generated A/B/C/D/E/F/+ strings and see the result '
                 '("在现有的任务上 ... 给它一些 random 的指令 ... 先做一个这样子的一个实验吧"). '
                 'pi0.5 alone, original/empty/random from t=0, unchanged standard LIBERO, no verifier or paid API.')


def make_root(repo, names, mode, smoke_root=None):
    units = A.plan(names, mode); shards = 1 if mode == 'smoke' else 4
    tokens = prompt_tokens([u['prompt'] for u in units])
    if max(tokens.values()) >= S.MAX_TOKENS:
        raise RuntimeError('Prompt truncation')
    root, manifest = prepare(mode, 'pi05_libero', repo, protocol=PROTOCOL, authorization=AUTHORIZATION, family='language_ablation')
    manifest.update(stage='language-ablation-' + mode, mode=mode, runs=len(units), states=len({u['group'] for u in units}),
                    shards=shards, partition='gengpu', smoke_root=None if smoke_root is None else str(smoke_root),
                    seed='hil_suite.policy_seed(suite, task, init), shared by all conditions and repeats of the state',
                    random_salt=A.RANDOM_SALT, random_alphabet=list(A.ALPHABET), random_symbols=A.SYMBOLS,
                    random_bank={str(i): A.random_prompt(i) for i in sorted({u['init'] for u in units})})
    plan = dict(mode=mode, shards=shards, units=units, prompt_tokens=tokens, smoke_root=manifest['smoke_root'])
    write(root/'PLAN.json', plan); write(root/'NAMES.json', names); write(root/'CAMPAIGN.json', manifest)
    return root, manifest


def environment(root, mode):
    return dict(os.environ, CB_FD_ROOT=str(root), CB_POLICY_CONFIG='pi05_libero', CB_POLICY_CHECKPOINT=checkpoint_path('pi05_libero'),
                CB_SUITE_SHARDS=str(read(root/'PLAN.json')['shards']), CB_SUITE_PHASE='smoke' if mode == 'smoke' else 'ablation',
                CB_SUITE_RUNNER='verifier_dataset.language_ablation_run')


def receipt(root, info):
    write(root/'RECEIPT.json', info)
    local = Path.cwd()/'results/language_ablation'/root.name; local.mkdir(parents=True, exist_ok=False)
    write(local/'RECEIPT.json', info); (local/'REPORT.md').symlink_to(root/'REPORT.md')
    return local


def submit_smoke(root):
    return submit(environment(root, 'smoke'), '--job-name=cb_lang_smoke', '--time=00:30:00',
                  '--output=' + str(root/'slurm_%j.log'), str(root/'source/verifier_dataset/hil_suite.sbatch'))


def submit_formal(root, manifest, dependency=None, submitted=None):
    env = environment(root, 'formal')
    after = [] if dependency is None else ['--dependency=afterok:' + dependency, '--kill-on-invalid-dep=yes']
    job = submit(env, '--job-name=cb_lang', '--array=0-3%2', '--time=08:00:00', *after,
                 '--output=' + str(root/'slurm_%A_%a.log'), str(root/'source/verifier_dataset/hil_suite.sbatch'))
    if submitted is not None:
        submitted.append(job)
    env['CB_SUITE_SUMMARY'] = 'summarize'
    summary = submit(env, '--job-name=cb_lang_summary', '--dependency=afterok:' + job, '--kill-on-invalid-dep=yes',
                     '--output=' + str(root/'slurm_%j.log'), str(root/'source/verifier_dataset/hil_suite_summary.sbatch'))
    if submitted is not None:
        submitted.append(summary)
    return dict(manifest, job=job, summarize_job=summary, integrity_job=dependency)


def main(mode, smoke_root=None):
    repo = Path.cwd().resolve()
    active = command('squeue', '-h', '-u', os.environ['USER'], '-o', '%j')
    if any(n.startswith('cb_') for n in active.splitlines()):
        raise RuntimeError('Finish existing CrashBench jobs before publishing/syncing and launching')
    if mode != 'smoke' and '状态：已冻结' not in (repo/PROTOCOL).read_text():
        raise RuntimeError('Freeze the design before submitting a formal run or pipeline')
    if mode != 'formal' and smoke_root is not None:
        raise RuntimeError('Only formal mode takes an existing smoke root')
    if mode == 'formal' and smoke_root is None:
        raise RuntimeError('Formal mode needs a finished smoke root')
    names = S.task_names((repo/'third_party/LIBERO/libero/libero/benchmark/libero_suite_task_map.py').read_text())
    gpu_room(1 if mode == 'smoke' else 2)
    # Prepare both immutable, published source snapshots BEFORE submitting any GPU job.
    if mode == 'pipeline':
        smoke, sm = make_root(repo, names, 'smoke')
        formal, fm = make_root(repo, names, 'formal', smoke)
        jobs = []
        try:
            smoke_job = submit_smoke(smoke); jobs.append(smoke_job)
            gate_env = dict(environment(formal, 'formal'), CB_LANGUAGE_SMOKE=str(smoke))
            gate = submit(gate_env, '--dependency=afterok:' + smoke_job, '--kill-on-invalid-dep=yes',
                          '--output=' + str(formal/'gate_%j.log'), str(formal/'source/verifier_dataset/language_ablation_gate.sbatch'))
            jobs.append(gate)
            fr = submit_formal(formal, fm, gate, submitted=jobs)
            local = receipt(smoke, dict(sm, job=smoke_job, integrity_job=gate, formal_root=str(formal), formal_receipt=fr))
            receipt(formal, fr)
            (local/'FORMAL_REPORT.md').symlink_to(formal/'REPORT.md')
            write(formal/'PIPELINE.json', dict(smoke_job=smoke_job, integrity_job=gate, formal_job=fr['job'], summarize_job=fr['summarize_job']))
            print(dict(smoke_root=str(smoke), formal_root=str(formal), jobs=read(formal/'PIPELINE.json')), flush=True)
        except BaseException as error:
            if jobs:
                command('scancel', *jobs)
            write(formal/'STOP_submission.json', dict(reason=str(error), cancelled_jobs=jobs))
            raise
    else:
        root, manifest = make_root(repo, names, mode, smoke_root)
        if mode == 'formal':
            validate(smoke_root, root)
            info = submit_formal(root, manifest)
        else:
            info = dict(manifest, job=submit_smoke(root))
        receipt(root, info); print(info, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['smoke', 'formal', 'pipeline'])
    parser.add_argument('smoke_root', nargs='?', type=Path); args = parser.parse_args()
    main(args.mode, None if args.smoke_root is None else Path(os.path.abspath(args.smoke_root)))
