"""Published launcher for the post-training step (docs/generative_verifier/HIL_TRAIN_PROTOCOL.md). Run from the repo root on
the login node with the openpi environment:
  /projects/p33100/siosio/envs/openpi/bin/python experiments/mechanism/launch_train.py export-smoke
  /projects/p33100/siosio/envs/openpi/bin/python experiments/mechanism/launch_train.py export
  /projects/p33100/siosio/envs/openpi/bin/python experiments/mechanism/launch_train.py train NAME TEACHER_ROOT
export-smoke  a handful of teacher episodes: export (one GPU) -> conversion (CPU). Checks the path; nothing from it is used.
export        the teacher data: export of every training episode (two GPUs) -> conversion (CPU).
train         one training run NAME (a key of SETTINGS) on the dataset of a finished formal export. NAME smoke is a few
              steps that measure memory and speed; its checkpoints are not used.
Every run root is a fresh snapshot of the clean, published commit. pi0.5 alone, stored runs of the finished census and
retreat batches, no API.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from verifier_dataset.inventory import ASSETS, read, write  # noqa: E402
from verifier_dataset.v5_launch import command, submit  # noqa: E402
from verifier_dataset.hil_suite_launch import checkpoint_path, gpu_room, prepare  # noqa: E402
import teacher as T  # noqa: E402

PROTOCOL = 'docs/generative_verifier/HIL_TRAIN_PROTOCOL.md'
CENSUS_ROOT = ASSETS/'mechanism/20261110T062732Z_mech-census-formal_pi05_libero_d1f4b0269a0f'
RETREAT_ROOT = ASSETS/'mechanism/20261114T220647Z_mech-retreat-formal_pi05_libero_69398de28780'
LIBERO = dict(repo_id='physical-intelligence/libero', root='/projects/p33100/siosio/huggingface_cache/lerobot/physical-intelligence/libero',
              episodes=1693, frames=273465)
AUTHORIZATION = ('2026-11-15: Claude reported the revised training design (HIL_METHOD_PLAN.md section 2.4) and listed three things to decide: the '
                 'cheap first pass (downloading the LIBERO training data, replaying and rendering the stored teacher runs, a training smoke, a '
                 'small T1 and its evaluation on the 360 held-out states), the two new scripted motions, and new prefixes for untrained phases '
                 'and layouts. The user replied "开始吧 尽量做出正面的结果 都可以写可以跑". pi0.5 alone, fine-tuned from the pi05_libero checkpoint on '
                 'stored runs of unchanged standard LIBERO-Goal and on LIBERO\'s public demonstrations; no verifier, no paid API, SafeLIBERO '
                 'untouched, at most two GPUs.')
POLICY = 'pi05_libero'
MECH = 'experiments/mechanism/'
NEW_SHARE = .75                         # of the training examples; the rest are LIBERO's own demonstrations (ReSteer mixes 3 : 1)
# One entry per training run. Everything but name, base_checkpoint and datasets, which the launcher fills in.
SETTINGS = dict(
    smoke=dict(batch_size=32, steps=40, peak_lr=2.5e-5, warmup_steps=10, ema_decay=None, save_interval=20, seed=0, num_workers=8),
)
HOURS = dict(smoke='01', formal='04')


def env(root, **extra):
    return dict(os.environ, CB_FD_ROOT=str(root), CB_POLICY_CONFIG=POLICY, **{k: str(v) for k, v in extra.items()})


def after(*jobs):
    return ['--dependency=' + ','.join('afterok:%s' % j for j in jobs), '--kill-on-invalid-dep=yes']


def job(root, name, script, deps=(), array=None, hours=None, **extra):
    options = ['--job-name=' + name, *deps, '--output=' + str(root/('slurm_%A_%a.log' if array else 'slurm_%j.log'))]
    if array:
        options.append('--array=' + array)
    if hours:
        options.append('--time=%s:00:00' % hours)
    return submit(env(root, **extra), *options, str(root/'source'/MECH/script))


def libero_ready():
    root = Path(LIBERO['root']); info = read(root/'meta/info.json')
    if info['total_episodes'] != LIBERO['episodes'] or info['total_frames'] != LIBERO['frames']:
        raise RuntimeError('Unexpected LIBERO dataset')
    missing = [k for k in range(LIBERO['episodes']) if not (root/info['data_path'].format(episode_chunk=k // info['chunks_size'], episode_index=k)).is_file()]
    if missing:
        raise RuntimeError('LIBERO dataset incomplete: %d episode files missing' % len(missing))


def teacher_root(repo, mode):
    root, manifest = prepare('mech-teacher-' + mode, POLICY, repo, protocol=PROTOCOL, authorization=AUTHORIZATION, family='mechanism')
    body = T.write_plan(CENSUS_ROOT, RETREAT_ROOT, root, mode)
    manifest.update(stage='mechanism-teacher-' + mode, partition='gengpu', mode=mode, shards=body['shards'], workers=body['workers'],
                    episodes=len(body['episodes']), counts=body['counts'], census_root=str(CENSUS_ROOT), retreat_root=str(RETREAT_ROOT),
                    prefix_root=body['prefix_root'], held_out=body['held_out'], dev_init=body['dev_init'],
                    seed='none: stored runs are replayed; the export has no random draw')
    write(root/'CAMPAIGN.json', manifest)
    return root, manifest, body


def train_root(repo, name, dataset, settings):
    root, manifest = prepare('mech-train-' + name, POLICY, repo, protocol=PROTOCOL, authorization=AUTHORIZATION, family='mechanism')
    body = dict(settings, name=name, base_checkpoint=checkpoint_path(POLICY),
                datasets=[dict(repo_id=LIBERO['repo_id'], root=LIBERO['root'], share=1 - NEW_SHARE), dict(dataset, share=NEW_SHARE)])
    write(root/'TRAIN.json', body)
    manifest.update(stage='mechanism-train-' + name, partition='gengpu', train=body, seed=settings['seed'])
    write(root/'CAMPAIGN.json', manifest)
    return root, manifest


def finished_export(root):
    root = Path(root); done = read(root/'DATASET.json'); plan = read(root/'PLAN.json')
    if not done['passed'] or done['mode'] != 'formal' or plan.get('experiment') != 'mechanism-teacher':
        raise RuntimeError('Training needs a finished formal export')
    return dict(repo_id=done['repo_id'], root=done['path'])


def main(stage, *arguments):
    repo = Path.cwd().resolve()
    active = command('squeue', '-h', '-u', os.environ['USER'], '-o', '%j')
    if any(n.startswith('cb_') for n in active.splitlines()):
        raise RuntimeError('Finish existing CrashBench jobs before publishing/syncing and launching')
    jobs = {}
    try:
        if stage in ('export-smoke', 'export'):
            mode = 'smoke' if stage == 'export-smoke' else 'formal'
            gpu_room(T.SHARDS[mode])
            root, manifest, body = teacher_root(repo, mode)
            jobs['export'] = job(root, 'cb_mech_teacher', 'teacher_export.sbatch', array='0-%d' % (body['shards'] - 1), hours=HOURS[mode],
                                 CB_TEACHER_SHARDS=body['shards'], CB_TEACHER_WORKERS=body['workers'])
            jobs['convert'] = job(root, 'cb_mech_teacher_convert', 'teacher_convert.sbatch', after(jobs['export']))
        else:
            name, teacher = arguments
            if name not in SETTINGS:
                raise RuntimeError('Unknown training run: ' + name)
            libero_ready(); gpu_room(1)
            root, manifest = train_root(repo, name, finished_export(Path(teacher).resolve()), SETTINGS[name])
            jobs['train'] = job(root, 'cb_mech_train', 'train_switch.sbatch', hours=HOURS.get(name))
    except BaseException:
        if jobs:
            command('scancel', *jobs.values())
        raise
    pipeline = dict(roots={stage: str(root)}, jobs=jobs); info = dict(manifest, pipeline=pipeline)
    write(root/'RECEIPT.json', info); write(root/'PIPELINE.json', pipeline)
    local = REPO/'results/mechanism'/root.name; local.mkdir(parents=True, exist_ok=False); write(local/'RECEIPT.json', info)
    print(pipeline, flush=True)


if __name__ == '__main__':
    arguments = sys.argv[1:]
    if not arguments or arguments[0] not in ('export-smoke', 'export', 'train') or len(arguments) != (3 if arguments[0] == 'train' else 1):
        raise SystemExit(__doc__)
    main(*arguments)
