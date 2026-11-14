"""Published launcher for the retreat-and-redo batch (docs/generative_verifier/HIL_RETREAT_PROTOCOL.md). Run from the repo
root on the login node with the openpi environment:
  /projects/p33100/siosio/envs/openpi/bin/python experiments/mechanism/launch_retreat.py dry [SETTINGS_JSON]
  /projects/p33100/siosio/envs/openpi/bin/python experiments/mechanism/launch_retreat.py smoke
  /projects/p33100/siosio/envs/openpi/bin/python experiments/mechanism/launch_retreat.py formal SMOKE_ROOT
dry     CPU only: the scripted segment alone on the smoke states (no policy), to choose the script's constants; SETTINGS_JSON
        maps setting names to overrides of hil_suite.RETREAT_PARAMS (default: the constants as committed).
smoke   two GPUs: three arms on initial state 0 with B = (A + 3) mod 10, then the CPU summary (tool check C1).
formal  needs the protocol's readings frozen and a smoke that reads tool_ok with the committed constants. One chain: gate
        (the smoke summarized again by this commit's code; anything but tool_ok cancels the rest) -> formal batch (six
        shards, two GPUs at a time) -> summary.
Every run root is a fresh snapshot of the clean, published commit. pi0.5 alone, the finished census's states and prefixes, no
training, no API.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from verifier_dataset.inventory import ASSETS, read, write  # noqa: E402
from verifier_dataset.v5_launch import command, submit  # noqa: E402
from verifier_dataset.hil_suite_launch import checkpoint_path, gpu_room, prepare  # noqa: E402
from verifier_dataset import hil_suite as S  # noqa: E402
import retreat as R  # noqa: E402

PROTOCOL = 'docs/generative_verifier/HIL_RETREAT_PROTOCOL.md'
FROZEN = '第 5 节状态：已冻结'
CENSUS_ROOT = ASSETS/'mechanism/20261110T062732Z_mech-census-formal_pi05_libero_d1f4b0269a0f'
AUTHORIZATION = ('2026-11-12: after Claude reported that mid-task switching already has fixes elsewhere and recommended, as step 1, a scripted '
                 'release-and-retreat to the start pose followed by native pi0.5 with B (upper bound, trivial baseline and teacher), saying it '
                 'needs a new scripted retreat not approved before, the user replied "开始吧 然后看看我这地基稳不稳 然后开始进行plan的修正和下一个agent的说明". '
                 '2026-11-14: asked whether GPU runs for step 1 are fine (formal batch about 25-30 GPU hours), the user replied "我都批准". '
                 'pi0.5 alone on the census states of unchanged standard LIBERO-Goal, no training, no verifier, no paid API, SafeLIBERO untouched, '
                 'at most two GPUs.')
POLICY = 'pi05_libero'
HOURS = dict(smoke='02', formal='10')
STATE_CHECK = dict(smoke=1000, formal=400)
MECH = 'experiments/mechanism/'


def env(root, **extra):
    return dict(os.environ, CB_FD_ROOT=str(root), CB_POLICY_CONFIG=POLICY, CB_POLICY_CHECKPOINT=checkpoint_path(POLICY),
                **{k: str(v) for k, v in extra.items()})


def after(*jobs):
    return ['--dependency=' + ','.join('afterok:%s' % j for j in jobs), '--kill-on-invalid-dep=yes']


def gpu_job(root, name, array, hours, deps, **extra):
    return submit(env(root, **extra), '--job-name=' + name, '--array=' + array, '--time=%s:00:00' % hours, *deps,
                  '--output=' + str(root/'slurm_%A_%a.log'), str(root/'source'/MECH/'closed_loop.sbatch'))


def cpu_job(root, name, deps, script, log='slurm_%j.log', **extra):
    return submit(env(root, **extra), '--job-name=' + name, *deps, '--output=' + str(root/log), str(root/'source'/MECH/script))


def usable_smoke(smoke_root):
    """The smoke the formal batch stands on: a finished smoke of this batch, tool_ok, run with the committed constants."""
    plan = read(smoke_root/'PLAN.json'); summary = read(smoke_root/'SUMMARY.json')
    if plan.get('experiment') != 'mechanism-retreat' or plan['mode'] != 'smoke' or summary['tool'] != 'tool_ok':
        raise RuntimeError('The formal batch needs a retreat smoke that reads tool_ok')
    if plan['params'] != dict(S.RETREAT_PARAMS):
        raise RuntimeError('The smoke ran with other script constants than the committed ones: rerun the smoke')


def main(stage, argument=None):
    repo = Path.cwd().resolve()
    active = command('squeue', '-h', '-u', os.environ['USER'], '-o', '%j')
    if any(n.startswith('cb_') for n in active.splitlines()):
        raise RuntimeError('Finish existing CrashBench jobs before publishing/syncing and launching')
    settings = smoke_root = None
    if stage == 'formal':
        if FROZEN not in (repo/PROTOCOL).read_text():
            raise RuntimeError('Freeze the readings (protocol section 5) before launching the formal batch')
        smoke_root = Path(argument).resolve(); usable_smoke(smoke_root)
    elif stage == 'dry' and argument is not None:
        settings = read(argument)
        if any(set(v) - set(S.RETREAT_PARAMS) for v in settings.values()):
            raise RuntimeError('A setting overrides an unknown script constant')
    if stage != 'dry':
        gpu_room(2)
    root, manifest = prepare('mech-retreat-' + stage, POLICY, repo, protocol=PROTOCOL, authorization=AUTHORIZATION, family='mechanism')
    body = R.write_plan(CENSUS_ROOT, root, stage, settings)
    manifest.update(stage='mechanism-retreat-' + stage, partition='short' if stage == 'dry' else 'gengpu', mode=stage, shards=body['shards'],
                    runs=len(body['branches']), runs_per_arm={a: sum(u['retreat_arm'] == a for u in body['branches']) for a in R.ARMS},
                    census_root=str(CENSUS_ROOT), prefix_root=body['prefix_root'], params=body['params'], settings=body.get('settings'),
                    inits=body['inits'], smoke_root=None if smoke_root is None else str(smoke_root),
                    seed='per run: the census prefix seed, shared by every arm of a state; the script is deterministic')
    write(root/'CAMPAIGN.json', manifest)
    jobs = {}
    try:
        if stage == 'dry':
            jobs['dry'] = cpu_job(root, 'cb_mech_retreat_dry', [], 'retreat_dry.sbatch')
        elif stage == 'smoke':
            jobs['smoke'] = gpu_job(root, 'cb_mech_retreat', '0-1', HOURS['smoke'], [], CB_MECH_SHARDS=body['shards'])
            jobs['summary'] = cpu_job(root, 'cb_mech_retreat_summary', after(jobs['smoke']), 'retreat_summary.sbatch', CB_STATE_CHECK=STATE_CHECK['smoke'])
        else:
            jobs['gate'] = cpu_job(root, 'cb_mech_retreat_gate', [], 'retreat_summary.sbatch', log='slurm_gate_%j.log', CB_RETREAT_GATE_SMOKE=smoke_root)
            jobs['formal'] = gpu_job(root, 'cb_mech_retreat', '0-%d%%2' % (body['shards'] - 1), HOURS['formal'], after(jobs['gate']),
                                     CB_MECH_SHARDS=body['shards'])
            jobs['summary'] = cpu_job(root, 'cb_mech_retreat_summary', after(jobs['formal']), 'retreat_summary.sbatch', CB_STATE_CHECK=STATE_CHECK['formal'])
    except BaseException:
        if jobs:
            command('scancel', *jobs.values())
        raise
    pipeline = dict(roots={stage: str(root)}, jobs=jobs); info = dict(manifest, pipeline=pipeline)
    write(root/'RECEIPT.json', info); write(root/'PIPELINE.json', pipeline)
    local = REPO/'results/mechanism'/root.name; local.mkdir(parents=True, exist_ok=False); write(local/'RECEIPT.json', info)
    print(pipeline, flush=True)
    return pipeline


if __name__ == '__main__':
    arguments = sys.argv[1:]
    if not arguments or arguments[0] not in ('dry', 'smoke', 'formal') or len(arguments) != dict(smoke=1, formal=2).get(arguments[0], len(arguments)) \
            or len(arguments) > 2:
        raise SystemExit(__doc__)
    main(*arguments)
