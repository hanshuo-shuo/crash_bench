"""Published launcher for the post-training step (docs/generative_verifier/HIL_TRAIN_PROTOCOL.md). Run from the repo root on
the login node with the openpi environment:
  /projects/p33100/siosio/envs/openpi/bin/python experiments/mechanism/launch_train.py pipeline NAME[,NAME...] [TEACHER_ROOT]
  /projects/p33100/siosio/envs/openpi/bin/python experiments/mechanism/launch_train.py export
  /projects/p33100/siosio/envs/openpi/bin/python experiments/mechanism/launch_train.py instruct
  /projects/p33100/siosio/envs/openpi/bin/python experiments/mechanism/launch_train.py train NAME TEACHER_ROOT [INSTRUCT_ROOT]
  /projects/p33100/siosio/envs/openpi/bin/python experiments/mechanism/launch_train.py eval SPLIT TRAIN_ROOT STEP [BASE_ROOT]
pipeline  one chain: the teacher data the runs need and do not have yet (the switch export unless TEACHER_ROOT, a
          finished one, is given; the instruction export and the script baseline when a run is taught the other
          instructions) -> the training runs, side by side -> the development evaluation of each saved checkpoint, one
          after the other once every training job has ended. A job starts only if the jobs it needs succeeded; a run
          that fails does not stop the other run's evaluations. A finished base run of the same kinds is used again.
export    the switch teacher data alone: export of every training episode (two GPUs) -> conversion (CPU).
instruct  the teacher data of the other instructions (protocol section 8): the scripts on the training states, rendered (one
          GPU) -> conversion (CPU); and the same scripts without cameras on the dev and test states (CPU): the script
          baseline.
train     one training run NAME (a key of SETTINGS) on the datasets of finished exports.
eval      one checkpoint (STEP, a key of TRAIN_ROOT/TRAINED.json's checkpoints) on the states of SPLIT (dev or test): the
          fine-tuned model on two GPUs, then the per-state units on the base checkpoint (skipped when BASE_ROOT, a finished
          base run of the same split and kinds, is given), then the summary. test needs the protocol's readings frozen and
          runs once per training run.
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
# One entry per training run: everything but name, base_checkpoint and datasets, which the launcher fills in.
T1 = dict(batch_sizes=[64, 32], steps=2000, peak_lr=2.5e-5, warmup_steps=100, ema_decay=None, save_interval=1000, seed=0, num_workers=10)
T1B = dict(T1, batch_sizes=[32], steps=6000, save_interval=2000)   # t1 for three times as long (64 did not fit in t1's probe)
SETTINGS = dict(t1=T1, t1b=T1B, t2=dict(T1B))     # t2 differs from t1b in its data alone
# Share of the training examples drawn from each teacher dataset; the rest are LIBERO's own demonstrations (ReSteer mixes 3 : 1).
DATA = dict(t1=dict(switch=.75), t1b=dict(switch=.75), t2=dict(switch=.5, instruct=.25))
TEACHER_MODE = dict(switch='formal', instruct='instruct')
KINDS = dict(t1=E.KINDS, t1b=E.KINDS, t2=E.KINDS + tuple(E.INSTRUCT))
HOURS = dict(export='04', train='12', dev='05', test='06', baseline='02')   # a run to the full budget takes about 30 s
STATE_CHECK = 400


def env(root, **extra):
    return dict(os.environ, CB_FD_ROOT=str(root), CB_POLICY_CONFIG=POLICY, **{k: str(v) for k, v in extra.items()})


def after(*jobs, ended=()):
    """Start after `jobs` succeeded and the jobs in `ended` are over, whatever their result."""
    needs = ['afterok:%s' % j for j in jobs] + ['afterany:%s' % j for j in ended if j not in jobs]
    return ['--dependency=' + ','.join(needs), '--kill-on-invalid-dep=yes'] if needs else []


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


STAGE = {'formal': 'mech-teacher', 'instruct': 'mech-teacher-instruct', 'instruct-baseline': 'mech-teacher-baseline'}


def teacher_root(repo, mode='formal'):
    root, manifest = fresh(STAGE[mode], repo)
    body = T.write_plan(CENSUS_ROOT, RETREAT_ROOT, root, mode)
    manifest.update(stage=STAGE[mode].replace('mech-', 'mechanism-'), partition='gengpu' if body['render'] else 'short', mode=mode,
                    shards=body['shards'], workers=body['workers'],
                    episodes=len(body['episodes']), episodes_per_kind=body['episodes_per_kind'], counts=body['counts'],
                    census_root=str(CENSUS_ROOT), retreat_root=str(RETREAT_ROOT), prefix_root=body['prefix_root'], held_out=body['held_out'],
                    dev_init=body['dev_init'], seed='none: stored runs are replayed; the export has no random draw')
    write(root/'CAMPAIGN.json', manifest)
    return root, manifest, body


def datasets(name, found):
    """The training mix of run NAME; found[kind] is dict(repo_id, root) of that teacher dataset."""
    shares = DATA[name]
    return ([dict(repo_id=LIBERO['repo_id'], root=LIBERO['root'], share=round(1 - sum(shares.values()), 6))]
            + [dict(found[kind], share=share) for kind, share in shares.items()])


def train_root(repo, name, found):
    root, manifest = fresh('mech-train-' + name, repo)
    body = dict(SETTINGS[name], name=name, base_checkpoint=checkpoint_path(POLICY), datasets=datasets(name, found))
    write(root/'TRAIN.json', body)
    manifest.update(stage='mechanism-train-' + name, partition='gengpu', train=body, seed=body['seed'])
    write(root/'CAMPAIGN.json', manifest)
    return root, manifest


def finished_export(root, mode='formal'):
    root = Path(root).resolve(); done = read(root/'DATASET.json'); plan = read(root/'PLAN.json')
    if not done['passed'] or done['mode'] != mode or plan.get('experiment') != 'mechanism-teacher':
        raise RuntimeError('Training needs a finished %s export' % mode)
    return dict(repo_id=done['repo_id'], root=done['path'])


def planned_export(root, body):
    return dict(repo_id=body['repo_id'], root=str(root/'lerobot'/body['repo_id']))


def finished_baseline():
    """The newest finished script baseline (launch_train.py instruct)."""
    for root in sorted((ASSETS/'mechanism').glob('*_%s_*' % STAGE['instruct-baseline']), reverse=True):
        if (root/'BASELINE.json').exists() and read(root/'BASELINE.json')['passed']:
            return root
    raise RuntimeError('Evaluating the other instructions needs a finished script baseline')


def finished_checkpoint(train, step):
    train = Path(train).resolve(); trained = read(train/'TRAINED.json'); name = trained['settings']['name']
    if not trained['passed'] or step not in trained['checkpoints'] or name not in SETTINGS:
        raise RuntimeError('Evaluation needs a checkpoint of a finished training run')
    checkpoint = Path(trained['checkpoints'][step])
    if not (checkpoint/'params').exists() or not (checkpoint/'assets'/'physical-intelligence/libero').exists():
        raise RuntimeError('Checkpoint incomplete: %s' % checkpoint)
    return train, name, checkpoint


def per_state(name):
    return [k for k in KINDS[name] if k in E.PER_STATE]


def base_root(repo, split, name):
    root, manifest = fresh('mech-eval-%s-base' % split, repo)
    body = E.write_plan(CENSUS_ROOT, RETREAT_ROOT, root, split, 'base', checkpoint_path(POLICY), tuple(per_state(name)), shards=1)
    manifest.update(stage='mechanism-eval-%s-base' % split, partition='gengpu', mode=split, shards=1, runs=len(body['branches']),
                    seed='per run: the census prefix seed')
    write(root/'CAMPAIGN.json', manifest)
    return root, manifest


def finished_base(base, split, name):
    base = Path(base).resolve(); plan = read(base/'PLAN.json')
    if plan['mode'] != split or plan['kinds'] != per_state(name) or not read(base/'SHARD_branch_0_COMPLETE.json')['passed']:
        raise RuntimeError('BASE_ROOT is not a finished base run of this split and these kinds')
    return base


def newest_base(split, name):
    """The newest finished base run of this split and the per-state kinds of run NAME, if there is one."""
    for root in sorted((ASSETS/'mechanism').glob('*_mech-eval-%s-base_*' % split), reverse=True):
        try:
            return finished_base(root, split, name)
        except (RuntimeError, OSError, KeyError):
            continue
    return None


def eval_root(repo, split, name, train, step, checkpoint, base, script=None):
    if split == 'test':
        if FROZEN not in (repo/PROTOCOL).read_text():
            raise RuntimeError('Freeze the readings (protocol section 5) before the test evaluation')
        for old in sorted((ASSETS/'mechanism').glob('*_mech-eval-test-%s-*' % name)):
            raise RuntimeError('This training run already has a test evaluation: %s' % old)
    root, manifest = fresh('mech-eval-%s-%s-%s' % (split, name, step), repo)
    body = E.write_plan(CENSUS_ROOT, RETREAT_ROOT, root, split, 'ft', checkpoint, KINDS[name], base_root=base, script_root=script)
    manifest.update(stage='mechanism-eval-%s-%s-%s' % (split, name, step), partition='gengpu', mode=split, shards=body['shards'], runs=len(body['branches']),
                    units_per_kind=body['units_per_kind'], policy_checkpoint=str(checkpoint), train_root=str(train), train_step=step,
                    base_root=str(base), script_root=None if script is None else str(script), census_root=str(CENSUS_ROOT), retreat_root=str(RETREAT_ROOT), seed='per run: the census prefix seed')
    write(root/'CAMPAIGN.json', manifest)
    return root, manifest


def submit_export(root, body, jobs, label='export', ended=()):
    jobs[label] = job(root, 'cb_mech_teacher', 'teacher_export.sbatch', after(ended=ended), array='0-%d' % (body['shards'] - 1), hours=HOURS['export'],
                      CB_TEACHER_SHARDS=body['shards'], CB_TEACHER_WORKERS=body['workers'])
    jobs[label + '_convert'] = job(root, 'cb_mech_teacher_convert', 'teacher_convert.sbatch', after(jobs[label]))
    return jobs[label + '_convert']


def submit_baseline(root, body, jobs):
    jobs['baseline'] = job(root, 'cb_mech_teacher_baseline', 'teacher_baseline.sbatch', hours=HOURS['baseline'], CB_TEACHER_WORKERS=body['workers'])
    return jobs['baseline']


def instruct_roots(repo, roots, jobs, ended=()):
    """The instruction teacher data (one GPU, once the GPU jobs in `ended` are over, then CPU) and the script baseline
    (CPU): (dataset, job to wait for, baseline root, its job)."""
    root, manifest, body = teacher_root(repo, 'instruct'); roots['instruct'] = (root, manifest)
    base, base_manifest, base_body = teacher_root(repo, 'instruct-baseline'); roots['baseline'] = (base, base_manifest)
    return planned_export(root, body), submit_export(root, body, jobs, 'instruct', ended), base, submit_baseline(base, base_body, jobs)


def submit_eval(root, split, checkpoint, base, run_base, deps, jobs, label, also=(), ended=()):
    """The fine-tuned model on two GPUs (after `deps` succeeded and the GPU jobs in `ended` are over), then (run_base) the
    base checkpoint's per-state units, then the summary, which also waits for the jobs in `also` (the script baseline).
    Returns this evaluation's GPU jobs."""
    def runner(r, c, array, wait, over, shards):
        return job(r, 'cb_mech_eval', 'closed_loop.sbatch', after(*wait, ended=over), array=array, hours=HOURS[split], CB_POLICY_CHECKPOINT=c,
                   CB_MECH_SHARDS=shards)
    jobs[label] = runner(root, checkpoint, '0-%d' % (E.SHARDS - 1), deps, ended, E.SHARDS); waits = [jobs[label]]
    if run_base:
        jobs[label + '_base'] = runner(base, checkpoint_path(POLICY), '0-0', [jobs[label]], (), 1); waits.append(jobs[label + '_base'])
    jobs[label + '_summary'] = job(root, 'cb_mech_eval_summary', 'switch_eval_summary.sbatch', after(*waits, *also), CB_STATE_CHECK=STATE_CHECK)
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
        elif stage == 'instruct':
            gpu_room(1)
            instruct_roots(repo, roots, jobs)
        elif stage == 'train':
            name = arguments[0]
            if name not in SETTINGS or len(arguments) != 1 + len(DATA[name]):
                raise RuntimeError('Training run %s needs these finished exports: %s' % (name, ', '.join(DATA.get(name, ()))))
            libero_ready(); gpu_room(1)
            found = {kind: finished_export(root, TEACHER_MODE[kind]) for kind, root in zip(DATA[name], arguments[1:])}
            roots['train'] = train_root(repo, name, found)
            jobs['train'] = job(roots['train'][0], 'cb_mech_train', 'train_switch.sbatch', hours=HOURS['train'])
        elif stage == 'eval':
            split, train, step = arguments[:3]
            gpu_room(E.SHARDS)
            train, name, checkpoint = finished_checkpoint(train, step)
            given = finished_base(arguments[3], split, name) if len(arguments) > 3 else None
            if given is None:
                roots['base'] = base_root(repo, split, name)
            base = given if given is not None else roots['base'][0]
            script = finished_baseline() if 'instruct' in DATA[name] else None
            roots['eval'] = eval_root(repo, split, name, train, step, checkpoint, base, script)
            submit_eval(roots['eval'][0], split, checkpoint, base, given is None, [], jobs, 'eval')
        else:
            names = arguments[0].split(',')
            if any(n not in SETTINGS for n in names) or len(set(names)) != len(names) or len(names) > 2:
                raise RuntimeError('Unknown, repeated or more than two training runs: ' + arguments[0])
            libero_ready(); gpu_room(2)
            found = {}; data = {}; script = None; also = []
            if len(arguments) > 1:
                found['switch'] = finished_export(arguments[1]); data['switch'] = []
            else:
                teacher, manifest, body = teacher_root(repo); roots['teacher'] = (teacher, manifest)
                found['switch'] = planned_export(teacher, body); data['switch'] = [submit_export(teacher, body, jobs)]
            if any('instruct' in DATA[n] for n in names):
                found['instruct'], wait, script, baseline = instruct_roots(repo, roots, jobs, [jobs[k] for k in ('export',) if k in jobs])
                data['instruct'] = [wait]; also = [baseline]
            for name in names:   # one GPU each; the switch export (two GPUs) is over before the instruction export (one) starts
                roots['train_' + name] = train_root(repo, name, {kind: found[kind] for kind in DATA[name]})
                jobs['train_' + name] = job(roots['train_' + name][0], 'cb_mech_train', 'train_switch.sbatch',
                                            after(*[j for kind in DATA[name] for j in data[kind]]), hours=HOURS['train'])
            held = [jobs['train_' + n] for n in names]   # the GPU jobs an evaluation waits out
            for name in names:
                train = roots['train_' + name][0]; given = newest_base('dev', name); instructed = 'instruct' in DATA[name]
                if given is None:
                    roots['base_' + name] = base_root(repo, 'dev', name)
                base = given if given is not None else roots['base_' + name][0]
                for k, (step, checkpoint) in enumerate(checkpoints(train, name).items()):
                    label = 'eval_%s_%s' % (name, step)
                    roots[label] = eval_root(repo, 'dev', name, train, step, checkpoint, base, script if instructed else None)
                    held = submit_eval(roots[label][0], 'dev', checkpoint, base, given is None and k == 0, [jobs['train_' + name]], jobs, label,
                                       also if instructed else (), held)
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
    if not arguments or len(arguments) not in dict(pipeline=(2, 3), export=(1,), instruct=(1,), train=(3, 4), eval=(4, 5)).get(arguments[0], ()):
        raise SystemExit(__doc__)
    main(*arguments)
