"""Published launcher for the P4 closed loop (docs/generative_verifier/HIL_MECHANISM_NEURONS.md 4.2). From the repo root:
  /projects/p33100/siosio/envs/openpi/bin/python experiments/mechanism/launch_neuron_loop.py NEURON_ROOT
NEURON_ROOT is the formal open-loop root (tool_ok); its k* sets go into both roots' NEURON_SETS.json. It prepares the smoke
root (inits 0-1) and the formal root (inits 2-19) from the same published commit, then chains: smoke array (two GPUs) -> gate
(CPU summary, fails unless tool_ok) -> formal array (four shards, at most two GPUs at a time) -> formal summary. Each run reruns
a HIL suite stage 1 switch branch from its stored prefix; pi0.5 alone, no training, no API.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from verifier_dataset.inventory import read, write  # noqa: E402
from verifier_dataset.v5_launch import command, submit  # noqa: E402
from verifier_dataset.hil_suite_launch import checkpoint_path, finished_branch, gpu_room, prepare  # noqa: E402
import neuron_loop as L  # noqa: E402
from launch import BRANCH_ROOT  # noqa: E402
from launch_neurons import AUTHORIZATION, FROZEN, PROTOCOL  # noqa: E402

SHARDS = dict(smoke=2, formal=4)
ARRAY = dict(smoke='0-1', formal='0-3%2')
HOURS = dict(smoke='03', formal='10')


def make_root(repo, mode, neuron_root):
    finished_branch(BRANCH_ROOT, Path(read(BRANCH_ROOT/'PLAN.json')['prefix_root']))
    root, manifest = prepare('mech-nrnloop-' + mode, 'pi05_libero', repo, protocol=PROTOCOL, authorization=AUTHORIZATION, family='mechanism')
    sets = L.neuron_sets(neuron_root); body = L.plan(BRANCH_ROOT, sets, mode, SHARDS[mode])
    write(root/'NEURON_SETS.json', sets); write(root/'PLAN.json', body)
    manifest.update(stage='mechanism-neurons-closed-' + mode, runs=len(body['branches']), shards=body['shards'], partition='gengpu',
                    arms=body['arms'], cell_arms=body['cell_arms'], inits=body['inits'], stage1_root=str(BRANCH_ROOT),
                    neuron_root=str(neuron_root), k_used=sets['k_used'],
                    seed='per run: the stage 1 SW unit seed (prefix seed), shared by every arm of the state; nrndir2 draws its random '
                         'direction from neuron_loop.direction_seed(SW id) folded with the request index')
    write(root/'CAMPAIGN.json', manifest)
    return root, manifest


def environment(root, mode):
    return dict(os.environ, CB_FD_ROOT=str(root), CB_POLICY_CHECKPOINT=checkpoint_path('pi05_libero'), CB_MECH_SHARDS=str(SHARDS[mode]))


def gpu_array(root, mode, after):
    return submit(environment(root, mode), '--job-name=cb_mech_nrnloop', '--array=' + ARRAY[mode], '--time=%s:00:00' % HOURS[mode], *after,
                  '--output=' + str(root/'slurm_%A_%a.log'), str(root/'source/experiments/mechanism/closed_loop.sbatch'))


def summary_job(root, mode, after, gate=False):
    env = dict(environment(root, mode), CB_NRNLOOP_GATE='1' if gate else '0')
    return submit(env, '--job-name=cb_mech_nrnloop_' + ('gate' if gate else 'summary'), '--dependency=afterok:' + after, '--kill-on-invalid-dep=yes',
                  '--output=' + str(root/'slurm_%j.log'), str(root/'source/experiments/mechanism/neuron_loop_summary.sbatch'))


def main(neuron_root):
    repo = Path.cwd().resolve(); neuron_root = Path(os.path.abspath(neuron_root))
    active = command('squeue', '-h', '-u', os.environ['USER'], '-o', '%j')
    if any(n.startswith('cb_') for n in active.splitlines()):
        raise RuntimeError('Finish existing CrashBench jobs before publishing/syncing and launching')
    if FROZEN not in (repo/PROTOCOL).read_text():
        raise RuntimeError('Freeze section 6 of the protocol before the closed loop')
    gpu_room(2)
    smoke, sm = make_root(repo, 'smoke', neuron_root); formal, fm = make_root(repo, 'formal', neuron_root)
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
                    summarize_job=formal_summary, neuron_root=str(neuron_root))
    for root, manifest in ((smoke, sm), (formal, fm)):
        info = dict(manifest, pipeline=pipeline)
        write(root/'RECEIPT.json', info); write(root/'PIPELINE.json', pipeline)
        local = REPO/'results/mechanism'/root.name; local.mkdir(parents=True, exist_ok=False); write(local/'RECEIPT.json', info)
    print(pipeline, flush=True)
    return pipeline


if __name__ == '__main__':
    main(sys.argv[1])
