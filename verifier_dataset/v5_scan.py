"""V5 step 1 CPU geometry scan; protocol in docs/generative_verifier/V5_STEP1_PROTOCOL.md.

extract ROOT SHARD reads one suite-level shard of official initial states in the simulator.
summarize ROOT joins the shards with the V4 source trajectories and writes the readouts.
Nothing is rendered, no policy or API is called and no dataset row is produced.
"""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import time
from .inventory import ASSETS, read, sha, write
from . import v5_geometry as g

SUITES = ['safelibero_spatial', 'safelibero_object', 'safelibero_goal', 'safelibero_long']
SHARDS = [(suite, level) for suite in SUITES for level in ('I', 'II')]
V4_ROOTS = [ASSETS/'verifier_dataset'/name for name in
            ('20261015T013004Z_constraint-batch_96a99b51efaa', '20261015T013009Z_constraint-batch_96a99b51efaa')]
SETTLE = 20
EXPECTED_STATES = 1600
EXPECTED_SOURCES = 67


def identity(suite, level, task, episode):
    return '%s/%s/task%d/episode%d' % (suite, level, task, episode)


def scene_objects(env, object_points):
    import numpy as np
    out = {}
    for name, body in sorted(env.env.obj_body_id.items()):
        try:
            points, _ = object_points(env, name)
            lo, hi, bounds = points.min(0), points.max(0), True
        except RuntimeError:
            lo = hi = np.asarray(env.sim.data.body_xpos[body]); bounds = False
        out[name] = dict(lo=lo.tolist(), hi=hi.tolist(), center=((lo + hi) / 2).tolist(), collision_bounds=bounds)
    near = [o for o in out.values() if abs(o['center'][0]) < 1 and abs(o['center'][1]) < 1]
    surface = float(np.median([o['lo'][2] for o in near]))
    for o in out.values():
        o['active'] = (abs(o['center'][0]) < 1 and abs(o['center'][1]) < 1
                       and surface - .06 <= o['lo'][2] <= surface + .40)
    return out, surface


def goal_region(env, objects, name):
    if name in objects:
        return dict(kind='object', name=name, lo=objects[name]['lo'], hi=objects[name]['hi'])
    m, d = env.sim.model, env.sim.data
    try:
        position = d.site_xpos[m.site_name2id(name)].tolist()
        return dict(kind='site', name=name, lo=position, hi=position)
    except (ValueError, KeyError):
        pass
    parents = [n for n in objects if name.startswith(n + '_')]
    if not parents:
        raise RuntimeError('Unresolved goal ' + name)
    parent = max(parents, key=len)
    return dict(kind='parent', name=parent, lo=objects[parent]['lo'], hi=objects[parent]['hi'])


def extract(root, index):
    suite_name, level = SHARDS[index]
    folder = root/'shards'/str(index); folder.mkdir(parents=True)
    from .replay import configure
    configure(folder, rendering=False)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    from geometry import active_obstacle, object_points
    states = []
    for task_id in range(4):
        suite = benchmark.get_benchmark_dict()[suite_name](safety_level=level); task = suite.get_task(task_id)
        env = ControlEnv(bddl_file_name=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file,
                         use_camera_obs=False, has_offscreen_renderer=False, camera_depths=False)
        try:
            inits = suite.get_task_init_states(task_id)
            for episode in range(len(inits)):
                started = time.monotonic()
                env.seed(7); env.reset(); obs = env.set_init_state(inits[episode])
                for _ in range(SETTLE):
                    obs, _, _, _ = env.step([0.]*6 + [-1.])
                objects, surface = scene_objects(env, object_points)
                atoms = [[str(a[0]).lower()] + list(a[1:]) for a in env.env.parsed_problem['goal_state']]
                atoms = [a for a in atoms if len(a) == 3 and a[0] in ('on', 'in')]
                try:
                    obstacle = active_obstacle(env, obs)
                except RuntimeError:
                    obstacle = None
                states.append(dict(source=identity(suite_name, level, task_id, episode), family=suite_name,
                                   task='%s/%s/task%d' % (suite_name, level, task_id), language=task.language,
                                   goal_atoms=atoms, goals={a[2]: goal_region(env, objects, a[2]) for a in atoms},
                                   objects=objects, surface=surface, obstacle=obstacle,
                                   seconds=round(time.monotonic() - started, 3)))
        finally:
            env.close()
    write(folder/'STATES.json', states)
    write(folder/'EXTRACT_COMPLETE.json', dict(passed=True, shard=index, suite=suite_name, level=level,
        states=len(states), states_sha256=sha(folder/'STATES.json'), settle_steps=SETTLE,
        slurm_job=os.environ.get('SLURM_JOB_ID'), code_commit=os.environ.get('CB_CODE_COMMIT')))


def initial_rows(state):
    objects, atoms = state['objects'], state['goal_atoms']
    excluded = {a[1] for a in atoms} | {state['goals'][a[2]]['name'] for a in atoms}
    names = g.anchors(objects, state['language'], excluded)
    pairs = [(g.point(objects[a[1]]['center']), (state['goals'][a[2]]['lo'], state['goals'][a[2]]['hi'])) for a in atoms]
    rows = []
    for rule in g.rules(objects, names) if pairs else []:
        separated, clearance = g.label(rule, pairs)
        rows.append(dict(source=state['source'], unit=0, family=state['family'], task=state['task'],
                         rule=rule['key'], template=rule['template'], label=separated, clearance=clearance))
    return rows, names


def checkpoint_rows(by_source):
    rows, inputs = [], {}
    for base in V4_ROOTS:
        for case in sorted(base.glob('case_*')):
            student = read(case/'INPUT.json')
            scene = read(Path(student['evidence']['witness_root'])/'SUMMARY.json')['source']['scene']
            source = identity(scene['suite'], scene['level'], scene['task'], scene['episode'])
            goal = read(case/'execution/GEOMETRY.json')['goal_definition']['goal']
            path = case/'execution/SAMPLES.jsonl.gz'; inputs[str(path)] = sha(path)
            with gzip.open(path, 'rt') as stream:
                samples = [json.loads(line) for line in stream]
            state = by_source[source]; objects = state['objects']; target, receiver = goal[1], goal[2]
            region = state['goals'].get(receiver)
            excluded = {target, receiver} | ({region['name']} if region else set())
            rules = g.rules(objects, g.anchors(objects, student['student']['task'], excluded))
            steps = max(s['step'] for s in samples)
            at = {s['step']: s for s in samples if s['phase'] == 'endpoint'}
            at[0] = next(s for s in samples if s['phase'] == 'initial')
            for t in range(0, max(0, steps - 10), 5):
                now = at[t]; start = g.point(now['native_point'])
                center = now['native_goal_state']['receiver_position']
                goal_box = g.shifted((region['lo'], region['hi']), center) if region else g.point(center)
                suffix = [g.point(s['native_point']) for s in samples if s['step'] > t]
                for rule in rules:
                    separated, clearance = g.label(rule, [(start, goal_box)])
                    crossed = g.crosses(rule, start, suffix)
                    row = dict(source=source, unit=t, family=scene['suite'], task=state['task'] + '|' + student['student']['task'],
                               rule=rule['key'], template=rule['template'], label=separated, clearance=clearance,
                               time_bin=min(g.TIME_BINS - 1, int(g.TIME_BINS * t / steps)), suffix_crosses=crossed)
                    if not separated:
                        row['witnessed'] = not crossed
                    rows.append(row)
    return rows, inputs


def dump(path, rows):
    with gzip.open(path, 'wt') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')


def summarize(root):
    states = []
    for index in range(len(SHARDS)):
        folder = root/'shards'/str(index); done = read(folder/'EXTRACT_COMPLETE.json')
        if not done['passed'] or sha(folder/'STATES.json') != done['states_sha256']:
            raise RuntimeError('Shard %d incomplete or changed' % index)
        states += read(folder/'STATES.json')
    by_source = {s['source']: s for s in states}
    if len(states) != EXPECTED_STATES or len(by_source) != EXPECTED_STATES:
        raise RuntimeError('Expected every official initial state exactly once')
    initial, anchor_counts, per_family = [], Counter(), {f: [] for f in SUITES}
    for state in states:
        rows, names = initial_rows(state); initial += rows
        anchor_counts.update(g.category(n) for n in names); per_family[state['family']].append(len(names))
    checkpoint, inputs = checkpoint_rows(by_source)
    if len({r['source'] for r in checkpoint}) != EXPECTED_SOURCES:
        raise RuntimeError('Expected all V4 source trajectories')
    dump(root/'INITIAL_ROWS.jsonl.gz', initial); dump(root/'CHECKPOINT_ROWS.jsonl.gz', checkpoint)
    primary = [r for r in checkpoint if r['clearance'] >= g.CLEARANCES[0]]
    summary = dict(initial=g.summarize(initial), checkpoint=g.summarize(checkpoint, timed=True),
                   anchors=dict(mean_per_state={f: sum(v) / len(v) for f, v in per_family.items()},
                                categories=dict(anchor_counts.most_common()),
                                obstacles=dict(Counter(s['obstacle'] for s in states).most_common())),
                   sanity=dict(separated_without_crossing=sum(r['label'] == 1 and not r['suffix_crosses'] for r in primary),
                               separated_rows=sum(r['label'] == 1 for r in primary),
                               unresolved_obstacles=sum(s['obstacle'] is None for s in states)),
                   v4_samples=inputs)
    summary['readings'] = g.readings(summary['initial'], summary['checkpoint'])
    write(root/'SUMMARY.json', summary)
    write(root/'V5_SCAN_COMPLETE.json', dict(passed=True, readings=summary['readings'], initial_rows=len(initial),
        checkpoint_rows=len(checkpoint), slurm_job=os.environ.get('SLURM_JOB_ID'),
        code_commit=os.environ.get('CB_CODE_COMMIT'), api_calls=0, dataset_rows=0))
    write(root/'HASHES.json', {n: sha(root/n) for n in
        ('INITIAL_ROWS.jsonl.gz', 'CHECKPOINT_ROWS.jsonl.gz', 'SUMMARY.json', 'V5_SCAN_COMPLETE.json')})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['extract', 'summarize'])
    p.add_argument('root', type=Path); p.add_argument('shard', type=int, nargs='?')
    a = p.parse_args()
    try:
        extract(a.root, a.shard) if a.mode == 'extract' else summarize(a.root)
    except BaseException as error:
        name = 'STOP_shard_%s.json' % a.shard if a.mode == 'extract' else 'STOP_summarize.json'
        write(a.root/name, dict(error=type(error).__name__, reason=str(error)))
        raise
