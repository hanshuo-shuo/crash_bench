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
SCREEN_LAYOUTS = 2
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


DESIGN_ATTEMPTS = 50


def design_layout(objects, movable, path, goal_box, target, seed, accept):
    """Try seeded random layouts until accept() holds; otherwise keep the last one and say so."""
    layout = None
    for attempt in range(DESIGN_ATTEMPTS):
        layout = sample_layout(objects, movable, path, goal_box, target, g.digest(seed, attempt, 'design'))
        if accept(layout):
            return dict(layout, designed=True, attempts=attempt + 1)
    return dict(layout, designed=False, attempts=DESIGN_ATTEMPTS)


def resting_twin(objects, stacks, layout, names, start, goal_box, lift):
    """Does the layout give, at this resting checkpoint, one separating and one non-separating rule
    that names a moved anchor and clears the lift-aware margin?"""
    moved = dict(objects)
    for rep, (x, y) in layout['moved'].items():
        o = objects[rep]; dx, dy = x - o['center'][0], y - o['center'][1]
        moved[rep] = dict(o, center=[x, y, o['center'][2]], lo=[o['lo'][0] + dx, o['lo'][1] + dy, o['lo'][2]],
                          hi=[o['hi'][0] + dx, o['hi'][1] + dy, o['hi'][2]])
    labels = set()
    for rule in g.rules(moved, [n for n in names if n in moved]):
        if rule['template'] != 'line' or not set(rule['anchors']) & set(layout['moved']):
            continue
        separated, clearance = g.label(rule, [(start, goal_box)])
        if clearance >= g.required_clearance(g.CLEARANCES[0], lift):
            labels.add(separated)
    return labels == {0, 1}


def movable_stacks(objects, contacts, owner, support, robot, free_joint, anchors, fixed):
    """Stacks resting on the support that contain an anchor; each moves rigidly as one unit.

    A stack is a connected set of objects in mutual contact (an obstacle on its base, a
    bowl on a box). It is movable when every member has a free joint, none is the target
    or part of the goal, nothing in it touches the robot, and its only outside contact is
    the table or floor. The representative is the stack's first anchor in name order.
    """
    from collections import defaultdict
    touching = defaultdict(set); blocked = set()
    for a, b in contacts:
        oa, ob = owner.get(a), owner.get(b)
        for o, other, geom in ((oa, ob, b), (ob, oa, a)):
            if o is None:
                continue
            if geom in robot:
                blocked.add(o)
            elif other is not None and other != o:
                touching[o].add(other)
            elif other is None and geom not in support:
                blocked.add(o)
    out, seen = {}, set()
    for name in sorted(anchors):
        if name in seen or not objects[name]['active']:
            continue
        stack, frontier = set(), [name]
        while frontier:
            n = frontier.pop()
            if n not in stack:
                stack.add(n); frontier += sorted(touching[n] - stack)
        seen |= stack
        if all(free_joint(n) for n in stack) and not stack & (fixed | blocked):
            out[name] = sorted(stack)
    return out


def merge_stacks(objects, stacks):
    """Replace each stack by its representative with the union box, so sampling treats it as one object."""
    merged = {n: o for n, o in objects.items() if not any(n in members and n != rep for rep, members in stacks.items())}
    for rep, members in stacks.items():
        lo = [min(objects[n]['lo'][i] for n in members) for i in range(3)]
        hi = [max(objects[n]['hi'][i] for n in members) for i in range(3)]
        merged[rep] = dict(objects[rep], lo=lo, hi=hi, center=[(a + b) / 2 for a, b in zip(lo, hi)])
    return merged


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
        stacks = movable_stacks(objects, contacts(), owner, support, robot, free_joint=lambda n: (n + '_joint0') in m.joint_names
                                and int(m.jnt_type[m.joint_name2id(n + '_joint0')]) == 0,
                                anchors=g.anchors(objects, spec['student_task'], {spec['target']} | g.goal_objects(objects, spec['goal'][2])),
                                fixed={spec['target']} | g.goal_objects(objects, spec['goal'][2]))
        merged = merge_stacks(objects, stacks)
        with gzip.open(Path(plan['c0_case'])/'TRACE.json.gz', 'rt') as stream:
            source_path = json.load(stream)['path']
        names = g.anchors(objects, spec['student_task'], {spec['target']} | g.goal_objects(objects, spec['goal'][2]))
        start = g.point(np.asarray(d.geom_xpos[geom]).tolist()); lift0 = max(0., float(d.geom_xpos[geom][2]) - surface)
        layout = design_layout(merged, sorted(stacks), thin([p[1:] for p in source_path]), (region['lo'], region['hi']),
                               spec['target'], plan['seed'],
                               lambda lay: resting_twin(merged, stacks, lay, names, start, (region['lo'], region['hi']), lift0))
        movable = sorted(stacks)
        members = {n for rep in layout['moved'] for n in stacks[rep]}
        for rep, (x, y) in layout['moved'].items():
            dx, dy = x - merged[rep]['center'][0], y - merged[rep]['center'][1]
            for name in stacks[rep]:
                j = m.joint_name2id(name + '_joint0'); adr, dof = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
                d.qpos[adr] += dx; d.qpos[adr + 1] += dy; d.qvel[dof:dof + 6] = 0.
        layout['stacks'] = {rep: stacks[rep] for rep in layout['moved']}
        env.sim.forward()
        placed = {n: np.asarray(d.body_xpos[int(env.env.obj_body_id[n])]).copy() for n in members}
        obstacle = [n for n in objects if 'obstacle' in n and objects[n]['active']]
        protected = {x for x, n in owner.items() if n in obstacle}
        target_geoms = {x for x, n in owner.items() if n == spec['target']}
        stack_of = {n: rep for rep in layout['moved'] for n in stacks[rep]}
        violations, path, snaps, current, drift = [], [], [], [0], [0.]
        def check():
            for a, b in contacts():
                for x, y in ((a, b), (b, a)):
                    if owner.get(x) in members and y not in support and owner.get(y) not in stacks.get(stack_of.get(owner.get(x)), ()):
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
            rows = [r for r in rows_from_trace(trace) if set(r['anchors']) & members]
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
               designed=sum(x.get('designed', False) for x in layouts),
               maximum_path_deviation=max((x['maximum_path_deviation'] or 0. for x in valid), default=None))
    twin_units = {u for u, labels in units.items() if labels == {0, 1}}
    family = {r['source']: r['family'] for r in kept}
    out['families_with_twins'] = len({family[u[0]] for u in twin_units})
    ok = (out['validity'] >= .5 and out['resting_twins_per_valid_layout'] >= 1 and (out['text_only_bound'] or 1.) <= .80
          and out['base_sources_with_twins'] >= 6 and out['families_with_twins'] >= 3)
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
    if entries and entries[0]['plan'].get('mode') == 'screen':
        split = {x['source_id']: x['split'] for x in read(Path(__file__).resolve().parent/'frozen_sources_67.json')['sources']}
        trace_root = Path(read(root/'CAMPAIGN.json')['trace_root'])
        with gzip.open(trace_root/'TRACE_ROWS.jsonl.gz', 'rt') as stream:
            natural = [json.loads(x) for x in stream]
        transfer = shortcut_transfer(natural, rows, split)
        summary.update(capability(layouts, rows, split), shortcut_transfer=transfer,
                       screen_reading='test_resists_natural_shortcut' if transfer['accuracy'] is not None
                       and transfer['accuracy'] <= .60 else 'shortcut_survives')
    write(root/'LAYOUT_SUMMARY.json', summary)
    write(root/'LAYOUT_COMPLETE.json', dict(passed=True, reading=summary.get('screen_reading', summary['reading']), api_calls=0, dataset_rows=0,
        code_commit=os.environ.get('CB_CODE_COMMIT'), slurm_job=os.environ.get('SLURM_JOB_ID')))
    write(root/'HASHES.json', {n: sha(root/n) for n in ('LAYOUTS.json', 'LAYOUT_ROWS.jsonl.gz', 'LAYOUT_SUMMARY.json', 'LAYOUT_COMPLETE.json')})


def plan(c0_root, mode='smoke'):
    """smoke: two C0 sources per family by digest, four layouts each. screen: every C0 source, two layouts each."""
    cases = read(c0_root/'CASES.json'); chosen = []
    if mode == 'screen':
        chosen, count, salt = list(cases), SCREEN_LAYOUTS, 'v5-layout-screen'
    else:
        for family in sorted({c['family'] for c in cases}):
            pool = sorted((c for c in cases if c['family'] == family), key=lambda c: g.digest(c['source'], 'v5-layout-smoke'))
            chosen += pool[:SOURCES_PER_FAMILY]
        count, salt = LAYOUTS_PER_SOURCE, 'v5-layout-v3'
    index = {c['source']: i for i, c in enumerate(cases)}
    return [dict(spec=c, plan=dict(layout=k, seed=g.digest(c['source'], k, salt), mode=mode,
                                   c0_case=str(c0_root/('case_%03d' % index[c['source']]))))
            for c in chosen for k in range(count)]


def resting_twin_units(rows):
    from collections import defaultdict
    units = defaultdict(set)
    for r in rows:
        if r['lift'] < RESTING_LIFT and r['clearance'] >= g.required_clearance(g.CLEARANCES[0], r['lift']):
            units[(r['source'], r['unit'])].add(r['label'])
    return {u for u, labels in units.items() if labels == {0, 1}}


def shortcut_transfer(natural, counterfactual, split):
    """Accuracy of the best text-and-progress lookup learned from training-source natural rows
    on held-out counterfactual resting-twin rows. Fallbacks: (task, rule), then the global majority."""
    from collections import Counter, defaultdict
    keep = lambda r: r['clearance'] >= g.required_clearance(g.CLEARANCES[0], r['lift'])
    train = [r for r in natural if keep(r) and split[r['source']] == 'train']
    tables = [defaultdict(Counter), defaultdict(Counter)]
    for r in train:
        tables[0][g.time_key(r)][r['label']] += 1; tables[1][g.text_key(r)][r['label']] += 1
    overall = Counter(r['label'] for r in train).most_common(1)[0][0] if train else 0
    twins = resting_twin_units(counterfactual)
    held = [r for r in counterfactual if keep(r) and (r['source'], r['unit']) in twins and split[r['base_source']] != 'train']
    def guess(r):
        for table, key in ((tables[0], g.time_key(r)), (tables[1], g.text_key(r))):
            if table.get(key):
                return table[key].most_common(1)[0][0], table is tables[0]
        return overall, False
    guesses = [guess(r) for r in held]
    correct = sum(label == r['label'] for (label, _), r in zip(guesses, held))
    return dict(rows=len(held), accuracy=correct / len(held) if held else None,
                matched_text_time=sum(exact for _, exact in guesses), training_rows=len(train))


def capability(layouts, rows, split):
    from collections import defaultdict
    twins = resting_twin_units(rows); base = {r['source']: r['base_source'] for r in rows}
    task = {r['base_source']: r['task'].split('|')[0] for r in rows}
    sources = sorted({base[u[0]] for u in twins})
    compositions = defaultdict(list)
    for s in sources:
        compositions[task[s]].append(s)
    return dict(capable_sources=sources, capable_compositions={t: sorted(v) for t, v in sorted(compositions.items())},
                capable_by_split={k: sum(split[s] == k for s in sources) for k in ('train', 'validation', 'test')},
                resting_twin_units=len(twins))


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
