"""Published launcher for the mechanism pilot's closed-loop causal test (docs/generative_verifier/HIL_MECHANISM_CLOSED_LOOP.md).
Run from the repo root on the login node with the openpi environment:
  /projects/p33100/siosio/envs/openpi/bin/python experiments/mechanism/launch_closed.py smoke
      inits 0-1 of every planned cell under every arm of the cell (86 runs), two GPUs, then the summary.
  /projects/p33100/siosio/envs/openpi/bin/python experiments/mechanism/launch_closed.py formal SMOKE_ROOT
      inits 2-19 (774 runs) on two GPUs. Needs the smoke's tool_ok and the protocol's readings frozen.
Each run reruns a HIL suite stage 1 switch branch from its stored prefix; pi0.5 alone, no training, no API.
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
import closed_loop as C  # noqa: E402
from launch import BRANCH_ROOT  # noqa: E402

PROTOCOL = 'docs/generative_verifier/HIL_MECHANISM_CLOSED_LOOP.md'
FROZEN = '第 5 节状态：已冻结'
AUTHORIZATION = ('2026-11-04 after the formal open-loop report the user chose next step 1, the closed-loop causal test ("先做第一项吧"): '
                 'restore the faded layer 12-14 image-token route in the failure cells, with random-direction and selectivity '
                 'controls; new steering server and its smoke and formal runs. pi0.5 alone, reruns of HIL suite stage 1 switch '
                 'branches, no training, no verifier, no paid API, SafeLIBERO untouched, at most two GPUs.')
HOURS = dict(smoke='02', formal='10')


def make_root(repo, mode):
    finished_branch(BRANCH_ROOT, Path(read(BRANCH_ROOT/'PLAN.json')['prefix_root']))
    root, manifest = prepare('mech-closed-' + mode, 'pi05_libero', repo, protocol=PROTOCOL, authorization=AUTHORIZATION, family='mechanism')
    body = C.plan(BRANCH_ROOT, mode)
    write(root/'PLAN.json', body)
    manifest.update(stage='mechanism-closed-loop-' + mode, runs=len(body['branches']), shards=body['shards'], partition='gengpu',
                    layers=body['layers'], arms=body['arms'], cell_arms=body['cell_arms'], inits=body['inits'], stage1_root=str(BRANCH_ROOT),
                    seed='per run: the stage 1 SW unit seed (prefix seed), shared by every arm of the state; rand arms add '
                         'closed_loop.direction_seed(SW id) folded with the request index')
    write(root/'CAMPAIGN.json', manifest)
    return root, manifest


def launch(root, manifest, mode):
    shards = read(root/'PLAN.json')['shards']
    env = dict(os.environ, CB_FD_ROOT=str(root), CB_POLICY_CHECKPOINT=checkpoint_path('pi05_libero'), CB_MECH_SHARDS=str(shards))
    job = submit(env, '--array=0-%d' % (shards - 1), '--time=%s:00:00' % HOURS[mode], '--output=' + str(root/'slurm_%A_%a.log'),
                 str(root/'source/experiments/mechanism/closed_loop.sbatch'))
    summary = submit(env, '--dependency=afterok:' + job, '--kill-on-invalid-dep=yes', '--output=' + str(root/'slurm_%j.log'),
                     str(root/'source/experiments/mechanism/closed_loop_summary.sbatch'))
    info = dict(manifest, job=job, summarize_job=summary)
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
            raise RuntimeError('Freeze the readings (protocol section 5) before the formal run')
        summary = read(Path(smoke_root)/'SUMMARY.json') if smoke_root is not None else {}
        if summary.get('tool') != 'tool_ok' or summary.get('mode') != 'smoke':
            raise RuntimeError('The formal run needs a closed-loop smoke root whose tool check is tool_ok')
    gpu_room(2)
    root, manifest = make_root(repo, mode)
    if smoke_root is not None:
        manifest['smoke_root'] = str(smoke_root); write(root/'CAMPAIGN.json', manifest)
    print(launch(root, manifest, mode), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['smoke', 'formal'])
    parser.add_argument('smoke_root', nargs='?', type=Path); args = parser.parse_args()
    main(args.mode, None if args.smoke_root is None else Path(os.path.abspath(args.smoke_root)))
