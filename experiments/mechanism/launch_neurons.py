"""Published launcher for the P4 open-loop stage (docs/generative_verifier/HIL_MECHANISM_NEURONS.md). From the repo root:
  /projects/p33100/siosio/envs/openpi/bin/python experiments/mechanism/launch_neurons.py smoke
      inits 0-1 (24 states; discovery = their 6 f = 0 states), one GPU at a time.
  /projects/p33100/siosio/envs/openpi/bin/python experiments/mechanism/launch_neurons.py formal SMOKE_ROOT
      all 240 states (discovery = f = 0 of inits 0-9), measurement on two GPUs. Needs the smoke's tool_ok and section 6 frozen.
Chain (Slurm dependencies): screen (GPU) -> select (CPU) -> measure (GPU array) -> summarize (CPU). Inference only on stored
HIL suite stage 1 inputs; pi0.5 alone, no simulator, no training, no API.
"""
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from verifier_dataset.inventory import read, sha, write  # noqa: E402
from verifier_dataset.v5_launch import command, submit  # noqa: E402
from verifier_dataset.hil_suite_launch import checkpoint_path, finished_branch, gpu_room, prepare  # noqa: E402
import neuron_readings as NR  # noqa: E402
import selection  # noqa: E402
from launch import BRANCH_ROOT  # noqa: E402

PROTOCOL = 'docs/generative_verifier/HIL_MECHANISM_NEURONS.md'
FROZEN = '第 6 节状态：已冻结'
AUTHORIZATION = ('2026-11-05 after the repair report the user approved the next step, P4 neurons ("可以 开始下一步吧 点头了"): the '
                 'protocol, code and tests, the open-loop smoke and (after section 6 is frozen) the formal open-loop batch, then the '
                 'closed-loop smoke, gate and formal batch, the blind test and the report. pi0.5 alone, inference on stored HIL suite '
                 'stage 1 inputs and reruns of stage 1 switch branches, no training, no verifier, no paid API, SafeLIBERO untouched, '
                 'at most two GPUs.')
SEEDS = 4
WINDOW = [12, 13, 14]
HOOKS = list(range(14))
SHARDS = dict(smoke=1, formal=2)


def make_root(repo, mode):
    finished_branch(BRANCH_ROOT, Path(read(BRANCH_ROOT/'PLAN.json')['prefix_root']))
    root, manifest = prepare('mech-neurons-' + mode, 'pi05_libero', repo, protocol=PROTOCOL, authorization=AUTHORIZATION, family='mechanism')
    states = selection.main(BRANCH_ROOT, root/'STATES.json', smoke=mode == 'smoke')['states']
    inits = NR.SMOKE_INITS if mode == 'smoke' else NR.DISCOVERY_INITS
    discovery = [s['id'] for s in states if s['phase'] == 0 and s['init'] in inits]
    plan = dict(mode=mode, shards=SHARDS[mode], seeds=SEEDS, state_ids=[s['id'] for s in states], discovery_ids=discovery,
                states_sha256=sha(root/'STATES.json'), hooks=HOOKS, window=WINDOW, k_grid=list(NR.K_GRID),
                branch_root=str(BRANCH_ROOT), branch_hashes_sha256=sha(BRANCH_ROOT/'HASHES.json'))
    manifest.update(stage='mechanism-neurons-' + mode, states=len(states), discovery=len(discovery), shards=SHARDS[mode], seeds=SEEDS,
                    partition='gengpu', seed='per state: the switch unit seed, then seed + 1 .. seed + %d (noise of the action rows); '
                                             'random neuron sets from neuron_readings.seed_of(pair, k)' % (SEEDS - 1))
    write(root/'PLAN.json', plan); write(root/'CAMPAIGN.json', manifest)
    return root, manifest


def launch(root, manifest):
    shards = read(root/'PLAN.json')['shards']
    env = dict(os.environ, CB_FD_ROOT=str(root), CB_POLICY_CHECKPOINT=checkpoint_path('pi05_libero'), CB_MECH_SHARDS=str(shards))
    gpu = str(root/'source/experiments/mechanism/neurons.sbatch'); cpu = str(root/'source/experiments/mechanism/neurons_cpu.sbatch')
    jobs = []
    try:
        jobs.append(submit(dict(env, CB_NEURON_ACTION='screen'), '--job-name=cb_mech_neurons', '--output=' + str(root/'slurm_screen_%j.log'), gpu))
        dep = lambda: ['--dependency=afterok:' + jobs[-1], '--kill-on-invalid-dep=yes']
        jobs.append(submit(dict(env, CB_NEURON_ACTION='select'), '--job-name=cb_mech_neurons_cpu', *dep(), '--output=' + str(root/'slurm_select_%j.log'), cpu))
        jobs.append(submit(dict(env, CB_NEURON_ACTION='measure'), '--job-name=cb_mech_neurons', '--array=0-%d' % (shards - 1), *dep(),
                           '--output=' + str(root/'slurm_measure_%A_%a.log'), gpu))
        jobs.append(submit(dict(env, CB_NEURON_ACTION='summarize'), '--job-name=cb_mech_neurons_cpu', *dep(), '--output=' + str(root/'slurm_summary_%j.log'), cpu))
    except BaseException:
        if jobs:
            command('scancel', *jobs)
        raise
    info = dict(manifest, jobs=dict(screen=jobs[0], select=jobs[1], measure=jobs[2], summarize=jobs[3]))
    write(root/'RECEIPT.json', info)
    local = REPO/'results/mechanism'/root.name; local.mkdir(parents=True, exist_ok=False); write(local/'RECEIPT.json', info)
    return info


def main(mode, smoke_root=None):
    repo = Path.cwd().resolve()
    active = command('squeue', '-h', '-u', os.environ['USER'], '-o', '%j')
    if any(n.startswith('cb_') for n in active.splitlines()):
        raise RuntimeError('Finish existing CrashBench jobs before publishing/syncing and launching')
    if mode == 'formal':
        if FROZEN not in (repo/PROTOCOL).read_text():
            raise RuntimeError('Freeze section 6 of the protocol before the formal run')
        if smoke_root is None or read(Path(smoke_root)/'SUMMARY.json')['tool'] != 'tool_ok':
            raise RuntimeError('The formal run needs a smoke root whose tool check is tool_ok')
    gpu_room(SHARDS[mode])
    root, manifest = make_root(repo, mode)
    if smoke_root is not None:
        manifest['smoke_root'] = str(smoke_root)
    print(launch(root, manifest), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['smoke', 'formal'])
    parser.add_argument('smoke_root', nargs='?', type=Path); args = parser.parse_args()
    main(args.mode, None if args.smoke_root is None else Path(os.path.abspath(args.smoke_root)))
