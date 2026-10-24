"""V5 counterfactual layouts: relocate untouched anchors, replay the source tape, keep valid witnesses.

Protocol: docs/generative_verifier/V5_LAYOUT_PROTOCOL.md. Moved anchors change the
physics hash, so the counterfactual replay itself is the witness: it must reach the
native goal, the moved anchors may touch only the table or floor and must stay within
1 cm, and neither the robot nor the target may touch the SafeLIBERO obstacle.

ROOT --shard K runs every fourth layout; ROOT --summarize aggregates; ROOT
--layout-index I runs one layout in a fresh process.
"""
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from .inventory import read, sha, write
from . import v5_geometry as g

SHARDS = 4
LAYOUTS_PER_SOURCE = 4
SOURCES_PER_FAMILY = 2
REGION_MARGIN = .10
OBJECT_GAP = .02
PATH_GAP = .06
GOAL_GAP = .04
TRIES = 200
MAX_DRIFT = .01
RESTING_LIFT = .05


def radius(box):
    lo, hi = box
    return math.hypot(hi[0] - lo[0], hi[1] - lo[1]) / 2


def rect_distance(p, lo, hi):
    return math.hypot(max(lo[0] - p[0], 0., p[0] - hi[0]), max(lo[1] - p[1], 0., p[1] - hi[1]))


def sample_layout(objects, movable, path, goal_box, target, seed):
    """New xy centres for every movable anchor; anchors with no admissible spot stay put."""
    rng = random.Random(int(seed[:16], 16))
    active = [n for n, o in objects.items() if o['active']]
    xs = [objects[n]['center'][0] for n in active]; ys = [objects[n]['center'][1] for n in active]
    x0, x1, y0, y1 = min(xs) - REGION_MARGIN, max(xs) + REGION_MARGIN, min(ys) - REGION_MARGIN, max(ys) + REGION_MARGIN
    centres = {n: objects[n]['center'][:2] for n in active}
    reach = radius((objects[target]['lo'], objects[target]['hi']))
    moved, stayed = {}, []
    for name in sorted(movable):
        r = radius((objects[name]['lo'], objects[name]['hi']))
        others = [(c, radius((objects[n]['lo'], objects[n]['hi']))) for n, c in centres.items() if n != name]
        for _ in range(TRIES):
            p = [rng.uniform(x0, x1), rng.uniform(y0, y1)]
            if all(math.hypot(p[0] - c[0], p[1] - c[1]) >= r + rc + OBJECT_GAP for c, rc in others) \
                    and all(math.hypot(p[0] - q[0], p[1] - q[1]) >= r + reach + PATH_GAP for q in path) \
                    and rect_distance(p, goal_box[0], goal_box[1]) >= r + GOAL_GAP:
                moved[name] = p; centres[name] = p
                break
        else:
            stayed.append(name)
    return dict(moved=moved, stayed=stayed, region=[[x0, y0], [x1, y1]])


def thin(path, step=.01):
    """Keep path points at least step apart in xy (the keep-out test does not need every substep)."""
    out = []
    for p in path:
        if not out or math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) >= step:
            out.append(p[:2])
    return out


def run_layout(root, index):
    import numpy as np
    from .paired_scene import make_env
    from .replay import assert_trace
    from .v5_scan import goal_region, scene_objects
    from .v5_render import body_geoms
    from .v5_trace import rows_from_trace
    case = root/('layout_%03d' % index); spec = read(case/'TRACE_INPUT.json'); plan = read(case/'LAYOUT_SPEC.json')
    folder = case/'execution'; folder.mkdir(); env = original = None
    try:
        env, physical_hash, update_marker, source, context = make_env(case, folder, render=False)
        from geometry import object_points
        m, d = env.sim.model, env.sim.data; geom = m.geom_name2id(spec['material_geom'])
        objects, surface = scene_objects(env, object_points)
        region = goal_region(env, objects, spec['goal'][2])
        support = {x for x in range(m.ngeom) if int(m.geom_bodyid[x]) == 0 or any(
            k in (m.body_id2name(int(m.geom_bodyid[x])) or '').lower() for k in ('table', 'floor'))}
        owner = {}
        for name, body in env.env.obj_body_id.items():
            for x in body_geoms(m, int(body)):
                owner[x] = name
        robot = {x for x in range(m.ngeom) if (m.body_id2name(int(m.geom_bodyid[x])) or '').startswith(('robot0_', 'gripper0_'))}
        def contacts():
            return [(int(c.geom1), int(c.geom2)) for c in d.contact[:d.ncon] if c.dist <= 0]
        resting = {n for n in objects if all(
            (owner.get(a) != n or b in support) and (owner.get(b) != n or a in support) for a, b in contacts())}
        names = g.anchors(objects, spec['student_task'], {spec['target']} | g.goal_objects(objects, spec['goal'][2]))
        free = {n for n in names if (n + '_joint0') in m.joint_names and int(m.jnt_type[m.joint_name2id(n + '_joint0')]) == 0}
        movable = sorted(n for n in free if n in resting)
        with gzip.open(Path(plan['c0_case'])/'TRACE.json.gz', 'rt') as stream:
            source_path = json.load(stream)['path']
        layout = sample_layout(objects, movable, thin([p[1:] for p in source_path]), (region['lo'], region['hi']),
                               spec['target'], plan['seed'])
        for name, (x, y) in layout['moved'].items():
            j = m.joint_name2id(name + '_joint0'); adr, dof = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
            d.qpos[adr] += x - objects[name]['center'][0]; d.qpos[adr + 1] += y - objects[name]['center'][1]
            d.qvel[dof:dof + 6] = 0.
        env.sim.forward()
        placed = {n: np.asarray(d.body_xpos[int(env.env.obj_body_id[n])]).copy() for n in layout['moved']}
        obstacle = [n for n in objects if 'obstacle' in n and objects[n]['active']]
        protected = {x for x, n in owner.items() if n in obstacle}
        target_geoms = {x for x, n in owner.items() if n == spec['target']}
        violations, path, snaps, current, drift = [], [], [], [0], [0.]
        def check():
            for a, b in contacts():
                for x, y in ((a, b), (b, a)):
                    if owner.get(x) in layout['moved'] and y not in support and owner.get(y) != owner.get(x):
                        violations.append(dict(step=current[0], kind='moved_anchor_contact', anchor=owner[x], other=m.geom_id2name(y)))
                    if x in protected and (y in robot or y in target_geoms):
                        violations.append(dict(step=current[0], kind='obstacle_contact', other=m.geom_id2name(y)))
            for n, p in placed.items():
                shift = float(np.linalg.norm(np.asarray(d.body_xpos[int(env.env.obj_body_id[n])]) - p))
                drift[0] = max(drift[0], shift)
                if shift > MAX_DRIFT:
                    violations.append(dict(step=current[0], kind='anchor_drift', anchor=n, metres=shift))
        def snapshot(step):
            objs, surf = scene_objects(env, object_points)
            snaps.append(dict(step=step, objects=objs, surface=surf, material=np.asarray(d.geom_xpos[geom]).tolist(),
                              goal=goal_region(env, objs, spec['goal'][2])))
        original = env.sim.step
        def monitored(*args, **kwargs):
            result = original(*args, **kwargs)
            path.append([current[0]] + np.asarray(d.geom_xpos[geom]).tolist()); check()
            return result
        tape_path = Path(source['root'])/'steps.jsonl'
        if sha(tape_path) != source['tape_sha256']:
            raise RuntimeError('Original source tape changed')
        tape = [json.loads(x) for x in tape_path.read_text().splitlines()]
        assert_trace(tape, source['actions'])
        wanted = set(range(0, max(0, len(tape) - 10), 5))
        check(); snapshot(0); env.sim.step = monitored
        low, high = map(np.asarray, env.env.action_spec)
        for item in tape:
            current[0] = item['step']
            obs, _, done, _ = env.step(np.clip(item['applied'], low, high).tolist())
            update_marker(obs)
            if item['step'] in wanted:
                snapshot(item['step'])
            if len(violations) > 20:
                break
        success = bool(env.env._eval_predicate(spec['goal']))
        deviation = max((math.dist(a[1:], b[1:]) for a, b in zip(path, source_path)), default=None)
        valid = success and not violations
        result = dict(layout, movable=movable, valid=valid, success=success, violations=violations[:20],
                      maximum_anchor_drift=drift[0], maximum_path_deviation=deviation, obstacle=obstacle)
        write(case/'LAYOUT.json', result)
        rows = []
        if valid:
            trace = dict(spec, source='%s#%d' % (spec['source'], plan['layout']), actions=len(tape), checkpoints=snaps, path=path)
            rows = rows_from_trace(trace)
            for row in rows:
                row.update(base_source=spec['source'], layout=plan['layout'])
        with gzip.open(case/'ROWS.jsonl.gz', 'wt') as stream:
            stream.writelines(json.dumps(r) + '\n' for r in rows)
        write(case/'COMPLETE.json', dict(passed=True, valid=valid, rows=len(rows), slurm_job=os.environ.get('SLURM_JOB_ID')))
    finally:
        if env is not None:
            if original is not None:
                delattr(env.sim, 'step')
            env.close()


def run(root, shard):
    for index, entry in enumerate(read(root/'LAYOUTS.json')):
        if index % SHARDS != shard:
            continue
        case = root/('layout_%03d' % index); case.mkdir()
        row = read(Path(entry['spec']['v4_case'])/'INPUT.json'); write(case/'INPUT.json', row)
        write(case/'TRACE_INPUT.json', entry['spec']); write(case/'LAYOUT_SPEC.json', entry['plan'])
        checkpoint = Path(row['evidence']['witness_root'])/'checkpoint_000'
        if row['evidence']['checkpoint_step'] != 0 or sha(checkpoint/'context.json') != row['evidence']['checkpoint_context_sha256']:
            raise RuntimeError('Original canonical checkpoint changed')
        shutil.copyfile(checkpoint/'context.json', case/'CANONICAL_CONTEXT.json')
        shutil.copyfile(checkpoint/'physics.npz', case/'CANONICAL_PHYSICS.npz')
        with (case/'worker.log').open('w') as log:
            subprocess.run([sys.executable, '-u', '-m', 'verifier_dataset.v5_layouts', str(root), '--layout-index', str(index)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    write(root/('SHARD_%d_COMPLETE.json' % shard), dict(passed=True, shard=shard, slurm_job=os.environ.get('SLURM_JOB_ID')))


def readouts(layouts, rows):
    """Preregistered smoke readouts R1-R5 and the reading."""
    from collections import Counter, defaultdict
    valid = [x for x in layouts if x['valid']]
    kept = [r for r in rows if r['clearance'] >= g.required_clearance(g.CLEARANCES[0], r['lift'])]
    units = defaultdict(set)
    for r in kept:
        if r['lift'] < RESTING_LIFT:
            units[(r['source'], r['unit'])].add(r['label'])
    resting = sum(labels == {0, 1} for labels in units.values())
    out = dict(layouts=len(layouts), valid_layouts=len(valid), validity=len(valid) / len(layouts) if layouts else 0.,
               invalid_reasons=dict(Counter(v['kind'] for x in layouts for v in x['violations'][:1])),
               unsuccessful=sum(not x['success'] for x in layouts),
               rows=len(rows), kept_rows=len(kept), resting_twins=resting,
               resting_twins_per_valid_layout=resting / len(valid) if valid else 0.,
               text_only_bound=g.text_bound(kept, g.text_key), strict_twins=g.readouts(kept, g.time_key)['twin_units'],
               text_balanced=g.readouts(kept, g.text_key)['balanced_rows'],
               base_sources_with_twins=len({r['base_source'] for r in kept if (r['source'], r['unit']) in
                                            {u for u, labels in units.items() if labels == {0, 1}}}),
               moved_per_layout=sum(len(x['moved']) for x in layouts) / len(layouts) if layouts else 0.,
               maximum_path_deviation=max((x['maximum_path_deviation'] or 0. for x in valid), default=None))
    ok = out['validity'] >= .5 and out['resting_twins_per_valid_layout'] >= 1 and (out['text_only_bound'] or 1.) <= .80
    out['reading'] = 'layout_design_viable' if ok else 'layout_design_insufficient'
    return out


def summarize(root):
    entries = read(root/'LAYOUTS.json'); layouts, rows = [], []
    for index in range(len(entries)):
        case = root/('layout_%03d' % index)
        if not read(case/'COMPLETE.json')['passed']:
            raise RuntimeError('Layout %d incomplete' % index)
        layouts.append(read(case/'LAYOUT.json'))
        with gzip.open(case/'ROWS.jsonl.gz', 'rt') as stream:
            rows += [json.loads(x) for x in stream]
    with gzip.open(root/'LAYOUT_ROWS.jsonl.gz', 'wt') as stream:
        stream.writelines(json.dumps(r) + '\n' for r in rows)
    summary = readouts(layouts, rows)
    write(root/'LAYOUT_SUMMARY.json', summary)
    write(root/'LAYOUT_COMPLETE.json', dict(passed=True, reading=summary['reading'], api_calls=0, dataset_rows=0,
        code_commit=os.environ.get('CB_CODE_COMMIT'), slurm_job=os.environ.get('SLURM_JOB_ID')))
    write(root/'HASHES.json', {n: sha(root/n) for n in ('LAYOUTS.json', 'LAYOUT_ROWS.jsonl.gz', 'LAYOUT_SUMMARY.json', 'LAYOUT_COMPLETE.json')})


def plan(c0_root):
    """Two C0 sources per family by digest, four seeded layouts each."""
    cases = read(c0_root/'CASES.json'); chosen = []
    for family in sorted({c['family'] for c in cases}):
        pool = sorted((c for c in cases if c['family'] == family), key=lambda c: g.digest(c['source'], 'v5-layout-smoke'))
        chosen += pool[:SOURCES_PER_FAMILY]
    index = {c['source']: i for i, c in enumerate(cases)}
    return [dict(spec=c, plan=dict(layout=k, seed=g.digest(c['source'], k, 'v5-layout'),
                                   c0_case=str(c0_root/('case_%03d' % index[c['source']]))))
            for c in chosen for k in range(LAYOUTS_PER_SOURCE)]


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path)
    p.add_argument('--layout-index', type=int); p.add_argument('--shard', type=int); p.add_argument('--summarize', action='store_true')
    a = p.parse_args()
    try:
        if a.layout_index is not None:
            run_layout(a.root, a.layout_index)
        elif a.summarize:
            summarize(a.root)
        else:
            run(a.root, a.shard)
    except BaseException as error:
        name = ('STOP_layout_%s.json' % a.layout_index if a.layout_index is not None else
                ('STOP_summarize.json' if a.summarize else 'STOP_shard_%s.json' % a.shard))
        write(a.root/name, dict(error=type(error).__name__, reason=str(error)))
        raise
