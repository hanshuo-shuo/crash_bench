"""Published launcher for the held-out open-loop localization (docs/generative_verifier/HIL_MECHANISM_HELDOUT_OPEN.md).
Run from the repo root on the login node with the openpi environment:
  /projects/p33100/siosio/envs/openpi/bin/python experiments/mechanism/launch_heldout_open.py
It prepares the smoke root (inits 0-1) and the formal root (inits 2-19) from the same published commit, then chains:
smoke array (one GPU) -> gate (CPU summary of the smoke, fails unless tool_ok) -> formal array (two GPUs) -> formal summary.
A failed gate cancels the formal batch through the dependency. The readings must be frozen in the protocol before launch.
Inference only on stored HIL suite stage 1 switch inputs: pi0.5 alone, no simulator, no training, no API.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from verifier_dataset.inventory import read, sha, write  # noqa: E402
from verifier_dataset.v5_launch import command, submit  # noqa: E402
from verifier_dataset.hil_suite_launch import checkpoint_path, finished_branch, gpu_room, prepare  # noqa: E402
import heldout_open as H  # noqa: E402
from launch import BRANCH_ROOT  # noqa: E402

PROTOCOL = 'docs/generative_verifier/HIL_MECHANISM_HELDOUT_OPEN.md'
FROZEN = '第 5 节状态：已冻结'
AUTHORIZATION = ('2026-11-05 after the P4 neuron report Claude recommended the held-out open-loop localization as the next step '
                 '(inference only on stored inputs, about 1-2 GPU hours) and the user replied "可以 先继续吧". pi0.5 alone, '
                 'inference only on stored HIL suite stage 1 switch inputs, no simulator, no training, no verifier, no paid API, '
                 'SafeLIBERO untouched, at most two GPUs.')
SHARDS = dict(smoke=1, formal=2)
HOURS = dict(smoke='01', formal='03')


def make_root(repo, mode):
    finished_branch(BRANCH_ROOT, Path(read(BRANCH_ROOT/'PLAN.json')['prefix_root']))
    for source in (H.REPAIR_ROOT/'RUNS.jsonl.gz', H.PILOT_ROOT/'SUMMARY.json'):
        if not source.exists():
            raise RuntimeError('Missing source: %s' % source)
    root, manifest = prepare('mech-hold-' + mode, 'pi05_libero', repo, protocol=PROTOCOL, authorization=AUTHORIZATION, family='mechanism')
    states = H.plan(BRANCH_ROOT, root/'STATES.json', mode)['states']
    plan = dict(mode=mode, experiment='mechanism-heldout-open', shards=SHARDS[mode], seeds=H.SEEDS, inits=list(H.SMOKE_INITS if mode == 'smoke' else H.READING_INITS),
                state_ids=[s['id'] for s in states], states_sha256=sha(root/'STATES.json'), branch_root=str(BRANCH_ROOT),
                branch_hashes_sha256=sha(BRANCH_ROOT/'HASHES.json'), repair_root=str(H.REPAIR_ROOT),
                repair_hashes_sha256=sha(H.REPAIR_ROOT/'HASHES.json'), pilot_root=str(H.PILOT_ROOT), layers=H.LAYERS)
    manifest.update(stage='mechanism-heldout-open-' + mode, states=len(states), shards=SHARDS[mode], seeds=H.SEEDS, partition='gengpu',
                    kinds={k: sum(s['kind'] == k for s in states) for k in ('full', 'steer')}, layers=H.LAYERS,
                    seed='per state: the switch unit seed (shared by its A twin and SW branch), then seed + 1 .. seed + %d; the '
                         'rand2 direction is closed_loop.direction_seed(SW id) folded with request 1' % (H.SEEDS - 1))
    write(root/'PLAN.json', plan); write(root/'CAMPAIGN.json', manifest)
    return root, manifest


def environment(root, mode):
    return dict(os.environ, CB_FD_ROOT=str(root), CB_POLICY_CHECKPOINT=checkpoint_path('pi05_libero'), CB_MECH_SHARDS=str(SHARDS[mode]))


def gpu_array(root, mode, after):
    return submit(environment(root, mode), '--job-name=cb_mech_hold', '--array=0-%d' % (SHARDS[mode] - 1), '--time=%s:00:00' % HOURS[mode],
                  *after, '--output=' + str(root/'slurm_%A_%a.log'), str(root/'source/experiments/mechanism/heldout_open.sbatch'))


def summary_job(root, mode, after, gate=False):
    env = dict(environment(root, mode), CB_HOLD_GATE='1' if gate else '0')
    return submit(env, '--job-name=cb_mech_hold_' + ('gate' if gate else 'summary'), '--dependency=afterok:' + after, '--kill-on-invalid-dep=yes',
                  '--output=' + str(root/'slurm_%j.log'), str(root/'source/experiments/mechanism/heldout_open_summary.sbatch'))


def main():
    repo = Path.cwd().resolve()
    active = command('squeue', '-h', '-u', os.environ['USER'], '-o', '%j')
    if any(n.startswith('cb_') for n in active.splitlines()):
        raise RuntimeError('Finish existing CrashBench jobs before publishing/syncing and launching')
    if FROZEN not in (repo/PROTOCOL).read_text():
        raise RuntimeError('Freeze the readings (protocol section 5) before launching the pipeline')
    gpu_room(2)
    # Both immutable source snapshots and plans exist before any job is submitted.
    smoke, sm = make_root(repo, 'smoke'); formal, fm = make_root(repo, 'formal')
    jobs = []
    try:
        smoke_job = gpu_array(smoke, 'smoke', []); jobs.append(smoke_job)
        gate = summary_job(smoke, 'smoke', smoke_job, gate=True); jobs.append(gate)
        formal_job = gpu_array(formal, 'formal', ['--dependency=afterok:' + gate, '--kill-on-invalid-dep=yes']); jobs.append(formal_job)
        formal_summary = summary_job(formal, 'formal', formal_job); jobs.append(formal_summary)
    except BaseException:
        if jobs:
            command('scancel', *jobs)
        raise
    pipeline = dict(smoke_root=str(smoke), formal_root=str(formal), smoke_job=smoke_job, gate_job=gate, formal_job=formal_job,
                    summarize_job=formal_summary)
    for root, manifest in ((smoke, sm), (formal, fm)):
        info = dict(manifest, pipeline=pipeline)
        write(root/'RECEIPT.json', info); write(root/'PIPELINE.json', pipeline)
        local = REPO/'results/mechanism'/root.name; local.mkdir(parents=True, exist_ok=False); write(local/'RECEIPT.json', info)
    print(pipeline, flush=True)
    return pipeline


if __name__ == '__main__':
    main()
