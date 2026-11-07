"""Published launcher for the unit study (docs/generative_verifier/HIL_MECHANISM_UNITS.md).
Run from the repo root on the login node with the openpi environment:
  /projects/p33100/siosio/envs/openpi/bin/python experiments/mechanism/launch_units.py
It prepares the smoke root (inits 0-1) and the formal root (inits 2-19) from the same published commit, then chains:
smoke array (one GPU) -> gate (CPU summary of the smoke, fails unless tool_ok) -> formal array (two GPUs) -> formal summary.
The readings must be frozen in the protocol before launch. Inference only on stored HIL suite stage 1 switch inputs:
pi0.5 alone, no simulator, no training, no API.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from verifier_dataset.inventory import read, sha, write  # noqa: E402
from verifier_dataset.v5_launch import command, submit  # noqa: E402
from verifier_dataset.hil_suite_launch import checkpoint_path, finished_branch, gpu_room, prepare  # noqa: E402
from launch import BRANCH_ROOT  # noqa: E402  (before run_units, which puts experiments/feasibility, with its own launch.py, first)
import run_units as RU  # noqa: E402
import units as U  # noqa: E402

PROTOCOL = 'docs/generative_verifier/HIL_MECHANISM_UNITS.md'
FROZEN = '第 5 节状态：已冻结'
AUTHORIZATION = ('2026-11-07 after comparing P4 with Goodfire\'s "Neural Networks Are Already Clean" essay, Claude proposed correcting the '
                 'wording and an open-loop follow-up that ranks neuron x image-row units by their effect on the action (about 1-2 GPU '
                 'hours); the user replied "可以 改 然后做第二个". pi0.5 alone, inference only on stored HIL suite stage 1 switch inputs, '
                 'no simulator, no training, no verifier, no paid API, SafeLIBERO untouched, at most two GPUs.')
SHARDS = dict(smoke=1, formal=2)
HOURS = dict(smoke='02', formal='04')


def make_root(repo, mode):
    finished_branch(BRANCH_ROOT, Path(read(BRANCH_ROOT/'PLAN.json')['prefix_root']))
    root, manifest = prepare('mech-units-' + mode, 'pi05_libero', repo, protocol=PROTOCOL, authorization=AUTHORIZATION, family='mechanism')
    states = RU.plan(BRANCH_ROOT, root/'STATES.json', mode)['states']
    plan = dict(mode=mode, experiment='mechanism-units', shards=SHARDS[mode], seeds=RU.SEEDS,
                inits=list(RU.SMOKE_INITS if mode == 'smoke' else RU.READING_INITS), state_ids=[s['id'] for s in states],
                states_sha256=sha(root/'STATES.json'), branch_root=str(BRANCH_ROOT), branch_hashes_sha256=sha(BRANCH_ROOT/'HASHES.json'),
                unit_grid=list(U.UNIT_GRID), neuron_grid=list(U.NEURON_GRID), k_max=U.K_MAX, specs=[list(s) for s in U.specs()])
    manifest.update(stage='mechanism-units-' + mode, states=len(states), shards=SHARDS[mode], seeds=RU.SEEDS, partition='gengpu',
                    seed='per state: noise seed = the switch unit seed (ranking), seed + 1 .. seed + 3 (evaluation); random unit and '
                         'neuron sets from run_units.set_seed(SW id)')
    write(root/'PLAN.json', plan); write(root/'CAMPAIGN.json', manifest)
    return root, manifest


def environment(root, mode):
    return dict(os.environ, CB_FD_ROOT=str(root), CB_POLICY_CHECKPOINT=checkpoint_path('pi05_libero'), CB_MECH_SHARDS=str(SHARDS[mode]))


def gpu_array(root, mode, after):
    return submit(environment(root, mode), '--job-name=cb_mech_units', '--array=0-%d' % (SHARDS[mode] - 1), '--time=%s:00:00' % HOURS[mode],
                  *after, '--output=' + str(root/'slurm_%A_%a.log'), str(root/'source/experiments/mechanism/units.sbatch'))


def summary_job(root, mode, after, gate=False):
    env = dict(environment(root, mode), CB_UNITS_GATE='1' if gate else '0')
    return submit(env, '--job-name=cb_mech_units_' + ('gate' if gate else 'summary'), '--dependency=afterok:' + after, '--kill-on-invalid-dep=yes',
                  '--output=' + str(root/'slurm_%j.log'), str(root/'source/experiments/mechanism/units_summary.sbatch'))


def main():
    repo = Path.cwd().resolve()
    active = command('squeue', '-h', '-u', os.environ['USER'], '-o', '%j')
    if any(n.startswith('cb_') for n in active.splitlines()):
        raise RuntimeError('Finish existing CrashBench jobs before publishing/syncing and launching')
    if FROZEN not in (repo/PROTOCOL).read_text():
        raise RuntimeError('Freeze the readings (protocol section 5) before launching the pipeline')
    gpu_room(2)
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
