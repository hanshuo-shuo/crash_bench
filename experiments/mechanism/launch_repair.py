"""Published launcher for the mechanism pilot's repair batch (docs/generative_verifier/HIL_MECHANISM_REPAIR.md).
Run from the repo root on the login node with the openpi environment:
  /projects/p33100/siosio/envs/openpi/bin/python experiments/mechanism/launch_repair.py
It prepares the smoke root (inits 0-1) and the formal root (inits 2-19) from the same published commit, then chains:
smoke array (two GPUs) -> gate (CPU summary of the smoke, fails unless tool_ok) -> formal array (four shards, at most two
GPUs at a time) -> formal summary. A failed gate cancels the formal batch through the dependency. The readings must be
frozen in the protocol before launch. Each run reruns a HIL suite stage 1 switch branch from its stored prefix; pi0.5
alone, no training, no API.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from verifier_dataset.inventory import read, write  # noqa: E402
from verifier_dataset.v5_launch import command, submit  # noqa: E402
from verifier_dataset.hil_suite_launch import checkpoint_path, finished_branch, gpu_room, prepare  # noqa: E402
import repair as R  # noqa: E402
from launch import BRANCH_ROOT  # noqa: E402

PROTOCOL = 'docs/generative_verifier/HIL_MECHANISM_REPAIR.md'
FROZEN = '第 5 节状态：已冻结'
AUTHORIZATION = ('2026-11-04 the user approved running the action-matched random control and the P5 repair on held-out pairs as one '
                 'batch, before P4 ("把动作上匹配的随机对照和 P5 修复合成一个批次先做，P4 神经元放在后面"), agreed to the three design '
                 'points (online action matching, lam 2 fixed, an output-level B - A extrapolation control) and asked to start the '
                 'formal batch directly ("可以 这个直接开始正式的吧 反正gpu够用"). pi0.5 alone, reruns of HIL suite stage 1 switch '
                 'branches, no training, no verifier, no paid API, SafeLIBERO untouched, at most two GPUs.')
SHARDS = dict(smoke=2, formal=4)
ARRAY = dict(smoke='0-1', formal='0-3%2')
HOURS = dict(smoke='03', formal='10')


def make_root(repo, mode):
    finished_branch(BRANCH_ROOT, Path(read(BRANCH_ROOT/'PLAN.json')['prefix_root']))
    root, manifest = prepare('mech-repair-' + mode, 'pi05_libero', repo, protocol=PROTOCOL, authorization=AUTHORIZATION, family='mechanism')
    body = R.plan(BRANCH_ROOT, mode, SHARDS[mode])
    write(root/'PLAN.json', body)
    manifest.update(stage='mechanism-repair-' + mode, runs=len(body['branches']), shards=body['shards'], partition='gengpu',
                    layers=body['layers'], arms=body['arms'], cell_arms=body['cell_arms'], in_failure=body['in_failure'], held=body['held'],
                    inits=body['inits'], stage1_root=str(BRANCH_ROOT),
                    seed='per run: the stage 1 SW unit seed (prefix seed), shared by every arm of the state; rand2 adds '
                         'closed_loop.direction_seed(SW id) folded with the request index; actrand draws its fixed push directions '
                         'from repair.push_seed(SW id)')
    write(root/'CAMPAIGN.json', manifest)
    return root, manifest


def environment(root, mode):
    return dict(os.environ, CB_FD_ROOT=str(root), CB_POLICY_CHECKPOINT=checkpoint_path('pi05_libero'), CB_MECH_SHARDS=str(SHARDS[mode]))


def gpu_array(root, mode, after):
    return submit(environment(root, mode), '--job-name=cb_mech_repair', '--array=' + ARRAY[mode], '--time=%s:00:00' % HOURS[mode], *after,
                  '--output=' + str(root/'slurm_%A_%a.log'), str(root/'source/experiments/mechanism/closed_loop.sbatch'))


def summary_job(root, mode, after, gate=False):
    env = dict(environment(root, mode), CB_REPAIR_GATE='1' if gate else '0')
    return submit(env, '--job-name=cb_mech_repair_' + ('gate' if gate else 'summary'), '--dependency=afterok:' + after, '--kill-on-invalid-dep=yes',
                  '--output=' + str(root/'slurm_%j.log'), str(root/'source/experiments/mechanism/repair_summary.sbatch'))


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
