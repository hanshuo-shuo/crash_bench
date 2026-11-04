"""Published launcher for the mechanism pilot's open-loop GPU runs. Run from the repo root with the openpi environment:
  /projects/p33100/siosio/envs/openpi/bin/python experiments/mechanism/launch.py smoke
      P1 tool check + first look: inits 0-1 of the three pairs at all four phases (24 states), one GPU, then the summary.
  /projects/p33100/siosio/envs/openpi/bin/python experiments/mechanism/launch.py formal SMOKE_ROOT
      All 240 states on two GPUs. Needs the smoke's tool_ok and the protocol section frozen (HIL_MECHANISM_IDEA.md 7).
Each GPU stage needs the user's go-ahead (AGENTS.md); nothing here trains, calls an API or runs the simulator.
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
import selection  # noqa: E402

BRANCH_ROOT = Path('/projects/p33100/siosio/crashbench_safelibero/hil_suite/20261030T092049Z_branch_pi05_libero_4f435c7d06db')
PROTOCOL = 'docs/generative_verifier/HIL_MECHANISM_IDEA.md'
FROZEN = '第 7 节状态：已冻结'
AUTHORIZATION = ('2026-11-04 user asked to turn HIL_MECHANISM_IDEA.md into the pilot plan and to start the P0 state selection '
                 'and the P1 patching code and unit tests ("开始改吧"); each GPU stage needs a separate go-ahead. pi0.5 alone, '
                 'inference only on stored HIL suite stage 1 inputs, no training, no verifier, no paid API, SafeLIBERO untouched.')
SEEDS = 4


def make_root(repo, mode):
    finished_branch(BRANCH_ROOT, Path(read(BRANCH_ROOT/'PLAN.json')['prefix_root']))
    root, manifest = prepare('mech-' + mode, 'pi05_libero', repo, protocol=PROTOCOL, authorization=AUTHORIZATION, family='mechanism')
    states = selection.main(BRANCH_ROOT, root/'STATES.json', smoke=mode == 'smoke')['states']
    shards = 1 if mode == 'smoke' else 2
    plan = dict(mode=mode, shards=shards, seeds=SEEDS, state_ids=[s['id'] for s in states], states_sha256=sha(root/'STATES.json'),
                branch_root=str(BRANCH_ROOT), branch_hashes_sha256=sha(BRANCH_ROOT/'HASHES.json'))
    manifest.update(stage='mechanism-open-loop-' + mode, states=len(states), shards=shards, seeds=SEEDS, partition='gengpu',
                    seed='per state: the switch unit seed (shared by its A twin and SW branch), then seed + 1 .. seed + %d' % (SEEDS - 1))
    write(root/'PLAN.json', plan); write(root/'CAMPAIGN.json', manifest)
    return root, manifest


def launch(root, manifest):
    shards = read(root/'PLAN.json')['shards']
    env = dict(os.environ, CB_FD_ROOT=str(root), CB_POLICY_CHECKPOINT=checkpoint_path('pi05_libero'), CB_MECH_SHARDS=str(shards))
    job = submit(env, '--job-name=cb_mech', '--array=0-%d' % (shards - 1), '--output=' + str(root/'slurm_%A_%a.log'),
                 str(root/'source/experiments/mechanism/mechanism.sbatch'))
    summary = submit(env, '--job-name=cb_mech_summary', '--dependency=afterok:' + job, '--kill-on-invalid-dep=yes',
                     '--output=' + str(root/'slurm_%j.log'), str(root/'source/experiments/mechanism/mechanism_summary.sbatch'))
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
            raise RuntimeError('Freeze section 7 of the plan before the formal run')
        if smoke_root is None or read(Path(smoke_root)/'SUMMARY.json')['tool'] != 'tool_ok':
            raise RuntimeError('The formal run needs a smoke root whose P1 reading is tool_ok')
    gpu_room(1 if mode == 'smoke' else 2)
    root, manifest = make_root(repo, mode)
    if smoke_root is not None:
        manifest['smoke_root'] = str(smoke_root)
    print(launch(root, manifest), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['smoke', 'formal'])
    parser.add_argument('smoke_root', nargs='?', type=Path); args = parser.parse_args()
    main(args.mode, None if args.smoke_root is None else Path(os.path.abspath(args.smoke_root)))
