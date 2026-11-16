"""Teacher data for the post-training step (docs/generative_verifier/HIL_TRAIN_PROTOCOL.md): which stored runs teach the
mid-task switch, and their conversion to the training format.

Every mid-task state of the census (old task A, initial layout, switch step, new task B) has up to three stored runs with
prompt B from the switch step on: the retreat to the start pose (home) and the release-and-lift (release) of the retreat
batch, and the untouched policy (native) of the census. The teacher of a state is the first of ORDER that finished B.

Besides these switch episodes, every prefix of the training split (task A done from the start under its own prompt) is a
keep episode: the same mid-task pictures with the instruction unchanged, where the action is to carry on.

Split, by state:
  test   the old task is one of HELD_OUT: never in training, evaluated once with frozen readings;
  dev    initial layout DEV_INIT of the other old tasks: never in training, used to develop;
  train  the rest; only these are exported.

The other mid-task instructions (STOP, RETURN, CANCEL; protocol section 8) have no stored runs to learn from. Their teacher
is a script (hil_suite.Instruct) run at every switch step of the training split's prefixes; a run is kept when it does what
the instruction asks by the suite's own rule. The same script run without cameras on the dev and test states is the
"keyword and script" baseline of those states.

  teacher.py plan CENSUS_ROOT RETREAT_ROOT OUT_ROOT [--mode MODE]   writes OUT_ROOT/PLAN.json (read-only on the two batches);
                                                         MODE: formal (switch), smoke, instruct, instruct-baseline
  teacher.py convert ROOT                                checks the export and writes ROOT/lerobot/REPO_ID -> ROOT/DATASET.json
  teacher.py baseline ROOT                               an instruct-baseline root's runs -> ROOT/BASELINE.json
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))

ORDER = ('home', 'native', 'release')   # the retreat when it finished B, otherwise a stored success without it
HELD_OUT = (4, 7)                       # old tasks: put the bowl on top of the cabinet; turn on the stove
DEV_INIT = 9
REPO_ID = 'crashbench/switch_teacher'
INSTRUCT_REPO_ID = 'crashbench/instruct_teacher'
FPS = 10                                # as physical-intelligence/libero
SHARDS = {'smoke': 1, 'formal': 2, 'instruct': 1, 'instruct-baseline': 1}
WORKERS = {'smoke': 2, 'formal': 6, 'instruct': 6, 'instruct-baseline': 8}
INSTRUCT_SPLITS = {'instruct': ('train',), 'instruct-baseline': ('dev', 'test')}
SMOKE_INIT, SMOKE_SHIFT = 0, 3          # smoke: initial layout 0 with B = (A + 3) mod 10, as the census smoke


def read(path):
    with open(path) as stream:
        return json.load(stream)


def write(path, data):
    temporary = Path(str(path) + '.tmp'); temporary.write_text(json.dumps(data, indent=1)); temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def lines(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(x) for x in stream]


# ---------------------------------------------------------------- selection (pure)
def split(task, init, held_out=HELD_OUT, dev_init=DEV_INIT):
    return 'test' if task in held_out else 'dev' if init == dev_init else 'train'


def select(units, done, folders):
    """units: the census's native units of the mid-task states; done[arm][sw]: whether that arm's run finished B;
    folders[arm]: where that arm's runs are stored (FOLDER/SW__ARM). One row per state."""
    out = []
    for u in sorted(units, key=lambda u: u['sw']):
        arm = next((a for a in ORDER if done[a][u['sw']]), None)
        out.append(dict(id=u['sw'], sw=u['sw'], kind='switch', arm=arm, source=None if arm is None else str(Path(folders[arm])/('%s__%s' % (u['sw'], arm))),
                        split=split(u['task'], u['init']), prefix=u['prefix'], suite=u['suite'], task=u['task'], other=u['other'],
                        init=u['init'], phase=u['phase'], checkpoint=u['checkpoint'], prompt=u['prompt'], pair=u['pair'],
                        same_object=u['covariates']['same_object']))
    return out


def keep_episodes(prefixes, prefix_root, wait):
    """One episode per prefix of the training split: task A under its own prompt from the end of the wait steps."""
    return [dict(id=p['id'] + '_KEEP', sw=None, kind='keep', arm='prefix', source=str(Path(prefix_root)/'prefix'/p['id']), split='train',
                 prefix=p['id'], suite=p['suite'], task=p['task'], other=None, init=p['init'], phase=0, checkpoint=wait, prompt=p['prompt'],
                 pair='%d->%d' % (p['task'], p['task']), same_object=None)
            for p in sorted(prefixes, key=lambda p: p['id']) if split(p['task'], p['init']) == 'train']


def instruct_episodes(units, prompts, retreat_root, splits, conditions, text):
    """One scripted episode per (switch step of a prefix in the given splits, instruction). units: the census's native units
    of the mid-task states; prompts[task]: the task's own prompt; text(condition, prompt_a): the instruction's wording."""
    states = {}
    for u in sorted(units, key=lambda u: u['sw']):
        states.setdefault((u['prefix'], u['checkpoint']), u)
    return [dict(id='%s_s%03d_%s' % (prefix, at, c), sw=None, kind='script', condition=c, arm='script', source=None,
                 home_source=str(Path(retreat_root)/'branch'/(u['sw'] + '__home')) if c == 'RETURN' else None, split=split(u['task'], u['init']),
                 prefix=prefix, suite=u['suite'], task=u['task'], other=None, init=u['init'], phase=u['phase'], checkpoint=at,
                 prompt=text(c, prompts[u['task']]), pair='%d->%s' % (u['task'], c), same_object=None)
            for (prefix, at), u in sorted(states.items()) if split(u['task'], u['init']) in splits for c in conditions]


def counts(states):
    return {s: dict(states=sum(x['split'] == s for x in states), **Counter(x['arm'] or 'none' for x in states if x['split'] == s))
            for s in ('train', 'dev', 'test')}


# ---------------------------------------------------------------- plan
def plan(census_root, retreat_root, mode='formal'):
    import retreat as R
    from verifier_dataset import hil_suite as S
    census_root = Path(census_root); retreat_root = Path(retreat_root)
    census_plan = R.finished_census(census_root); retreat_plan = read(retreat_root/'PLAN.json')
    summary = read(retreat_root/'SUMMARY.json')
    if retreat_plan.get('experiment') != 'mechanism-retreat' or retreat_plan['mode'] != 'formal' or summary['tool'] != 'tool_ok' \
            or Path(retreat_plan['census_root']) != census_root:
        raise RuntimeError('The teacher data needs the finished formal retreat batch on this census')
    for n, digest in read(retreat_root/'HASHES.json').items():
        if sha(retreat_root/n) != digest:
            raise RuntimeError('Retreat run output changed: ' + n)
    units = [u for u in census_plan['branches'] if u['arm']['name'] == 'native' and u['phase']]
    done = dict(native={r['sw']: r['b_done'] for r in lines(census_root/'RUNS.jsonl.gz') if r['arm'] == 'native' and r['phase']})
    for arm in ('home', 'release'):
        done[arm] = {r['sw']: r['b_done'] for r in lines(retreat_root/'RUNS.jsonl.gz') if r['arm'] == arm}
    if any(set(done[a]) != {u['sw'] for u in units} for a in ORDER):
        raise RuntimeError('The two batches do not cover the same states')
    states = select(units, done, dict(native=census_root/'branch', home=retreat_root/'branch', release=retreat_root/'branch'))
    if mode in INSTRUCT_SPLITS:
        prompts = {p['task']: p['prompt'] for p in census_plan['prefix_order']}
        episodes = instruct_episodes(units, prompts, retreat_root, INSTRUCT_SPLITS[mode], S.INTERVENTIONS,
                                     lambda c, a: dict(STOP=S.STOP, RETURN=S.RETURN, CANCEL=S.CANCEL % a)[c])
    else:
        episodes = [s for s in states if s['split'] == 'train' and s['arm'] is not None
                    and (mode == 'formal' or (s['init'] == SMOKE_INIT and s['other'] == (s['task'] + SMOKE_SHIFT) % 10))]
        episodes += [e for e in keep_episodes(census_plan['prefix_order'], census_plan['prefix_root'], S.WAIT)
                     if mode == 'formal' or e['init'] == SMOKE_INIT]
    return dict(experiment='mechanism-teacher', mode=mode, order=list(ORDER), held_out=list(HELD_OUT), dev_init=DEV_INIT,
                repo_id=INSTRUCT_REPO_ID if mode in INSTRUCT_SPLITS else REPO_ID, render=mode != 'instruct-baseline',
                shards=SHARDS[mode], workers=WORKERS[mode], prefix_root=census_plan['prefix_root'], census_root=str(census_root),
                retreat_root=str(retreat_root), census_hashes_sha256=sha(census_root/'HASHES.json'),
                retreat_hashes_sha256=sha(retreat_root/'HASHES.json'), counts=counts(states),
                episodes_per_kind=dict(Counter(e['kind'] for e in episodes)), states=states, episodes=episodes)


def write_plan(census_root, retreat_root, out_root, mode='formal'):
    out_root = Path(out_root)
    if (out_root/'PLAN.json').exists():
        raise RuntimeError('Plan target already has a plan: %s' % out_root)
    body = plan(census_root, retreat_root, mode); write(out_root/'PLAN.json', body)
    print(len(body['episodes']), 'episodes', body['episodes_per_kind'], body['counts'])
    return body


# ---------------------------------------------------------------- export check and conversion
def export_reading(plan_, metas):
    """Tool check of the export: every episode replayed with equal physics; a stored run re-rendered the stored policy input
    and a scripted RETURN ended in the stored home run's physics state (the exporter fails an episode otherwise). How far
    the images are from exactly equal is reported. A scripted run that did not do what its instruction asks is not a
    failure of the tool: it is counted (invalid) and left out of the dataset."""
    missing = [e['id'] for e in plan_['episodes'] if e['id'] not in metas]
    failed = [m['id'] for m in metas.values() if not m['passed']]
    ok = [m for m in metas.values() if m['passed'] and m.get('valid', True)]
    checked = [m for m in ok if 'input_check' in m]
    linf = max((max(m['input_check'][k]['linf'] for k in ('image', 'wrist_image')) for m in checked), default=None)
    unequal = sum(any(m['input_check'][k]['linf'] > 0 for k in ('image', 'wrist_image')) for m in checked)
    return dict(episodes=len(plan_['episodes']), exported=len(ok), missing=len(missing), failed=len(failed), failed_ids=failed[:20],
                invalid=sum(m['passed'] and not m.get('valid', True) for m in metas.values()),
                frames=sum(m['frames'] for m in ok), scripted_frames=sum(m['scripted'] for m in ok),
                clipped_steps=sum(m['clipped'] for m in ok), image_linf_max=linf, image_not_identical=unequal,
                passed=not missing and not failed)


def baseline_table(episodes, metas):
    """The script as a baseline: per split and instruction, how many runs did what was asked (valid) and how many pass the
    suite's rule alone (follow)."""
    out = {}
    for e in episodes:
        m = metas[e['id']]; o = m['outcome']
        follow = dict(STOP=o['stopped'], RETURN=o['home'], CANCEL=not o['a_done'])[e['condition']]
        cell = out.setdefault(e['split'], {}).setdefault(e['condition'], dict(runs=0, valid=0, follow=0, steps=[]))
        cell['runs'] += 1; cell['valid'] += bool(m['valid']); cell['follow'] += bool(follow); cell['steps'].append(m['script']['steps'])
    for cells in out.values():
        for cell in cells.values():
            cell['steps_median'] = sorted(cell.pop('steps'))[cell['runs'] // 2]
    return out


def baseline(root):
    root = Path(root); plan_ = read(root/'PLAN.json')
    if plan_['mode'] != 'instruct-baseline':
        raise RuntimeError('Not a script-baseline root')
    metas = {p.stem: read(p) for p in sorted((root/'episode').glob('*.json'))}
    reading = export_reading(plan_, metas)
    if not reading['passed']:
        write(root/'BASELINE.json', dict(passed=False, export=reading)); raise SystemExit('The runs did not pass: %s' % reading)
    table = baseline_table(plan_['episodes'], metas)
    write(root/'BASELINE.json', dict(passed=True, export=reading, table=table,
                                     rows=[dict(id=e['id'], split=e['split'], condition=e['condition'], task=e['task'], init=e['init'], phase=e['phase'],
                                                valid=metas[e['id']]['valid'], outcome=metas[e['id']]['outcome'], script=metas[e['id']]['script'])
                                           for e in plan_['episodes']],
                                     code_commit=os.environ.get('CB_CODE_COMMIT'), slurm_job=os.environ.get('SLURM_JOB_ID')))
    print(json.dumps(table), flush=True)


def features():
    image = dict(dtype='image', shape=(256, 256, 3), names=['height', 'width', 'channel'])
    return dict(image=image, wrist_image=dict(image), state=dict(dtype='float32', shape=(8,), names=['state']),
                actions=dict(dtype='float32', shape=(7,), names=['actions']))


def convert(root):
    """The exported episodes as a LeRobot dataset with the features of physical-intelligence/libero."""
    import numpy as np
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    root = Path(root); plan_ = read(root/'PLAN.json')
    metas = {p.stem: read(p) for p in sorted((root/'episode').glob('*.json'))}
    reading = export_reading(plan_, metas)
    if not reading['passed'] or not plan_.get('render', True):
        write(root/'DATASET.json', dict(passed=False, export=reading)); raise SystemExit('The export did not pass: %s' % reading)
    episodes = [e for e in plan_['episodes'] if metas[e['id']].get('valid', True)]
    target = root/'lerobot'/plan_['repo_id']
    if target.exists():
        raise RuntimeError('Dataset target exists: %s' % target)
    data = LeRobotDataset.create(repo_id=plan_['repo_id'], root=target, robot_type='panda', fps=FPS, features=features(),
                                 image_writer_threads=4, image_writer_processes=2)
    frames = 0
    for e in episodes:
        path = root/'episode'/(e['id'] + '.npz')
        if sha(path) != metas[e['id']]['npz_sha256']:
            raise RuntimeError('Exported episode changed: ' + e['id'])
        with np.load(path) as stored:   # each array once: indexing the archive decompresses the whole array every time
            z = {k: stored[k] for k in ('image', 'wrist_image', 'state', 'actions')}
        for k in range(len(z['actions'])):
            data.add_frame(dict(image=z['image'][k], wrist_image=z['wrist_image'][k], state=z['state'][k], actions=z['actions'][k], task=e['prompt']))
        data.save_episode(); frames += len(z['actions'])
    if frames != reading['frames'] or data.meta.total_episodes != len(episodes) or data.meta.total_frames != frames:
        raise RuntimeError('Converted dataset does not match the export')
    write(root/'DATASET.json', dict(passed=True, mode=plan_['mode'], repo_id=plan_['repo_id'], path=str(target), episodes=len(episodes), frames=frames,
                                    tasks=sorted({e['prompt'] for e in episodes}), by_arm=dict(Counter(e['arm'] for e in episodes)),
                                    by_kind=dict(Counter(e['kind'] for e in episodes)),
                                    by_condition=dict(Counter(e['condition'] for e in episodes if e['kind'] == 'script')),
                                    export=reading, code_commit=os.environ.get('CB_CODE_COMMIT'), slurm_job=os.environ.get('SLURM_JOB_ID')))
    print(json.dumps(reading), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('plan'); a.add_argument('census_root'); a.add_argument('retreat_root'); a.add_argument('out_root'); a.add_argument('--mode', default='formal', choices=sorted(SHARDS))
    for name in ('convert', 'baseline'):
        a = sub.add_parser(name); a.add_argument('root')
    args = p.parse_args()
    if args.command == 'plan':
        write_plan(args.census_root, args.retreat_root, args.out_root, args.mode)
    else:
        dict(convert=convert, baseline=baseline)[args.command](args.root)
