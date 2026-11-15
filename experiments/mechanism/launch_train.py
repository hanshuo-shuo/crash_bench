"""Published launcher for the post-training step (docs/generative_verifier/HIL_TRAIN_PROTOCOL.md). Run from the repo root on
the login node with the openpi environment:
  /projects/p33100/siosio/envs/openpi/bin/python experiments/mechanism/launch_train.py pipeline NAME
  /projects/p33100/siosio/envs/openpi/bin/python experiments/mechanism/launch_train.py export
  /projects/p33100/siosio/envs/openpi/bin/python experiments/mechanism/launch_train.py train NAME TEACHER_ROOT
  /projects/p33100/siosio/envs/openpi/bin/python experiments/mechanism/launch_train.py eval SPLIT TRAIN_ROOT STEP [BASE_ROOT]
pipeline  one chain: export -> conversion -> training run NAME -> the development evaluation of each saved checkpoint.
          Each job starts only if the one before it succeeded.
export    the teacher data alone: export of every training episode (two GPUs) -> conversion (CPU).
train     one training run NAME (a key of SETTINGS) on the dataset of a finished export.
eval      one checkpoint (STEP, a key of TRAIN_ROOT/TRAINED.json's checkpoints) on the states of SPLIT (dev or test): the
          fine-tuned model on two GPUs, then the keep units on the base checkpoint (skipped when BASE_ROOT, a finished base
          run of the same split, is given), then the summary. test needs the protocol's readings frozen and runs once per
          training run.
Every run root is a fresh snapshot of the clean, published commit. pi0.5 alone, stored runs of the finished census and
retreat batches, no API. At most two GPUs are in use at any time.
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
import switch_eval as E  # noqa: E402

PROTOCOL = 'docs/generative_verifier/HIL_TRAIN_PROTOCOL.md'
FROZEN = '第 5 节状态：已冻结'
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
# One entry per training run: everything but name, base_checkpoint and datasets, which the launcher fills in.
SETTINGS = dict(
    t1=dict(batch_sizes=[64, 32], steps=2000, peak_lr=2.5e-5, warmup_steps=100, ema_decay=None, save_interval=1000, seed=0, num_workers=10),
)
HOURS = dict(export='04', train='12', dev='03', test='06')
STATE_CHECK = 400


def env(root, **extra):
    return dict(os.environ, CB_FD_ROOT=str(root), CB_POLICY_CONFIG=POLICY, **{k: str(v) for k, v in extra.items()})


def after(*jobs):
    return ['--dependency=' + ','.join('afterok:%s' % j for j in jobs), '--kill-on-invalid-dep=yes'] if jobs else []


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


def checkpoints(train, name):
    """Where the training run NAME in root TRAIN saves: step -> folder (openpi: checkpoint_base_dir/config name/exp_name/step)."""
    s = SETTINGS[name]
    return {str(k): train/'checkpoints'/POLICY/name/str(k) for k in sorted(set(range(s['save_interval'], s['steps'], s['save_interval'])) | {s['steps'] - 1})}


def fresh(stage, repo):
    return prepare(stage, POLICY, repo, protocol=PROTOCOL, authorization=AUTHORIZATION, family='mechanism')


def teacher_root(repo):
    root, manifest = fresh('mech-teacher', repo)
    body = T.write_plan(CENSUS_ROOT, RETREAT_ROOT, root, 'formal')
    manifest.update(stage='mechanism-teacher', partition='gengpu', mode='formal', shards=body['shards'], workers=body['workers'],
                    episodes=len(body['episodes']), episodes_per_kind=body['episodes_per_kind'], counts=body['counts'],
                    census_root=str(CENSUS_ROOT), retreat_root=str(RETREAT_ROOT), prefix_root=body['prefix_root'], held_out=body['held_out'],
                    dev_init=body['dev_init'], seed='none: stored runs are replayed; the export has no random draw')
    write(root/'CAMPAIGN.json', manifest)
    return root, manifest, body


def train_root(repo, name, dataset):
    root, manifest = fresh('mech-train-' + name, repo)
    body = dict(SETTINGS[name], name=name, base_checkpoint=checkpoint_path(POLICY),
                datasets=[dict(repo_id=LIBERO['repo_id'], root=LIBERO['root'], share=1 - NEW_SHARE), dict(dataset, share=NEW_SHARE)])
    write(root/'TRAIN.json', body)
    manifest.update(stage='mechanism-train-' + name, partition='gengpu', train=body, seed=body['seed'])
    write(root/'CAMPAIGN.json', manifest)
    return root, manifest


def finished_export(root):
    root = Path(root); done = read(root/'DATASET.json'); plan = read(root/'PLAN.json')
    if not done['passed'] or done['mode'] != 'formal' or plan.get('experiment') != 'mechanism-teacher':
        raise RuntimeError('Training needs a finished formal export')
    return dict(repo_id=done['repo_id'], root=done['path'])


def finished_checkpoint(train, step):
    train = Path(train).resolve(); trained = read(train/'TRAINED.json'); name = trained['settings']['name']
    if not trained['passed'] or step not in trained['checkpoints'] or name not in SETTINGS:
        raise RuntimeError('Evaluation needs a checkpoint of a finished training run')
    checkpoint = Path(trained['checkpoints'][step])
    if not (checkpoint/'params').exists() or not (checkpoint/'assets'/'physical-intelligence/libero').exists():
        raise RuntimeError('Checkpoint incomplete: %s' % checkpoint)
    return train, name, checkpoint


def base_root(repo, split):
    root, manifest = fresh('mech-eval-%s-base' % split, repo)
    body = E.write_plan(CENSUS_ROOT, RETREAT_ROOT, root, split, 'base', checkpoint_path(POLICY), ('keep',), shards=1)
    manifest.update(stage='mechanism-eval-%s-base' % split, partition='gengpu', mode=split, shards=1, runs=len(body['branches']),
                    seed='per run: the census prefix seed')
    write(root/'CAMPAIGN.json', manifest)
    return root, manifest


def finished_base(base, split):
    base = Path(base).resolve(); plan = read(base/'PLAN.json')
    if plan['mode'] != split or plan['kinds'] != ['keep'] or not read(base/'SHARD_branch_0_COMPLETE.json')['passed']:
        raise RuntimeError('BASE_ROOT is not a finished base run of this split')
    return base


def eval_root(repo, split, name, train, step, checkpoint, base):
    if split == 'test':
        if FROZEN not in (repo/PROTOCOL).read_text():
            raise RuntimeError('Freeze the readings (protocol section 5) before the test evaluation')
        for old in sorted((ASSETS/'mechanism').glob('*_mech-eval-test-%s-*' % name)):
            raise RuntimeError('This training run already has a test evaluation: %s' % old)
    root, manifest = fresh('mech-eval-%s-%s-%s' % (split, name, step), repo)
    body = E.write_plan(CENSUS_ROOT, RETREAT_ROOT, root, split, 'ft', checkpoint, base_root=base)
    manifest.update(stage='mechanism-eval-%s-%s-%s' % (split, name, step), partition='gengpu', mode=split, shards=body['shards'], runs=len(body['branches']),
                    units_per_kind=body['units_per_kind'], policy_checkpoint=str(checkpoint), train_root=str(train), train_step=step,
                    base_root=str(base), census_root=str(CENSUS_ROOT), retreat_root=str(RETREAT_ROOT), seed='per run: the census prefix seed')
    write(root/'CAMPAIGN.json', manifest)
    return root, manifest


def submit_export(root, body, jobs):
    jobs['export'] = job(root, 'cb_mech_teacher', 'teacher_export.sbatch', array='0-%d' % (body['shards'] - 1), hours=HOURS['export'],
                         CB_TEACHER_SHARDS=body['shards'], CB_TEACHER_WORKERS=body['workers'])
    jobs['convert'] = job(root, 'cb_mech_teacher_convert', 'teacher_convert.sbatch', after(jobs['export']))


def submit_eval(root, split, checkpoint, base, run_base, deps, jobs, label):
    """The fine-tuned model on two GPUs, then (run_base) the base checkpoint's keep units, then the summary."""
    def runner(r, c, array, wait, shards):
        return job(r, 'cb_mech_eval', 'closed_loop.sbatch', after(*wait), array=array, hours=HOURS[split], CB_POLICY_CHECKPOINT=c, CB_MECH_SHARDS=shards)
    jobs[label] = runner(root, checkpoint, '0-%d' % (E.SHARDS - 1), deps, E.SHARDS); waits = [jobs[label]]
    if run_base:
        jobs[label + '_base'] = runner(base, checkpoint_path(POLICY), '0-0', [jobs[label]], 1); waits.append(jobs[label + '_base'])
    jobs[label + '_summary'] = job(root, 'cb_mech_eval_summary', 'switch_eval_summary.sbatch', after(*waits), CB_STATE_CHECK=STATE_CHECK)
    return waits


def main(stage, *arguments):
    repo = Path.cwd().resolve()
    active = command('squeue', '-h', '-u', os.environ['USER'], '-o', '%j')
    if any(n.startswith('cb_') for n in active.splitlines()):
        raise RuntimeError('Finish existing CrashBench jobs before publishing/syncing and launching')
    jobs = {}; roots = {}
    try:
        if stage == 'export':
            gpu_room(T.SHARDS['formal'])
            root, manifest, body = teacher_root(repo); roots['teacher'] = (root, manifest)
            submit_export(root, body, jobs)
        elif stage == 'train':
            name, teacher = arguments
            if name not in SETTINGS:
                raise RuntimeError('Unknown training run: ' + name)
            libero_ready(); gpu_room(1)
            roots['train'] = train_root(repo, name, finished_export(Path(teacher).resolve()))
            jobs['train'] = job(roots['train'][0], 'cb_mech_train', 'train_switch.sbatch', hours=HOURS['train'])
        elif stage == 'eval':
            split, train, step = arguments[:3]
            gpu_room(E.SHARDS)
            train, name, checkpoint = finished_checkpoint(train, step)
            given = finished_base(arguments[3], split) if len(arguments) > 3 else None
            if given is None:
                roots['base'] = base_root(repo, split)
            base = given if given is not None else roots['base'][0]
            roots['eval'] = eval_root(repo, split, name, train, step, checkpoint, base)
            submit_eval(roots['eval'][0], split, checkpoint, base, given is None, [], jobs, 'eval')
        else:
            (name,) = arguments
            if name not in SETTINGS:
                raise RuntimeError('Unknown training run: ' + name)
            libero_ready(); gpu_room(2)
            teacher, manifest, body = teacher_root(repo); roots['teacher'] = (teacher, manifest)
            roots['train'] = train_root(repo, name, dict(repo_id=body['repo_id'], root=str(teacher/'lerobot'/body['repo_id'])))
            train = roots['train'][0]; roots['base'] = base_root(repo, 'dev'); base = roots['base'][0]
            saved = checkpoints(train, name)
            for step, checkpoint in saved.items():
                roots['eval_' + step] = eval_root(repo, 'dev', name, train, step, checkpoint, base)
            submit_export(teacher, body, jobs)
            jobs['train'] = job(train, 'cb_mech_train', 'train_switch.sbatch', after(jobs['convert']), hours=HOURS['train'])
            waits = [jobs['train']]
            for k, (step, checkpoint) in enumerate(saved.items()):
                waits = submit_eval(roots['eval_' + step][0], 'dev', checkpoint, base, k == 0, waits, jobs, 'eval_' + step)
    except BaseException:
        if jobs:
            command('scancel', *jobs.values())
        raise
    pipeline = dict(roots={k: str(r) for k, (r, _) in roots.items()}, jobs=jobs)
    for root, manifest in roots.values():
        info = dict(manifest, pipeline=pipeline)
        write(root/'RECEIPT.json', info); write(root/'PIPELINE.json', pipeline)
        local = REPO/'results/mechanism'/root.name; local.mkdir(parents=True, exist_ok=False); write(local/'RECEIPT.json', info)
    print(pipeline, flush=True)


if __name__ == '__main__':
    arguments = sys.argv[1:]
    if not arguments or len(arguments) not in dict(pipeline=(2,), export=(1,), train=(3,), eval=(4, 5)).get(arguments[0], ()):
        raise SystemExit(__doc__)
    main(*arguments)
