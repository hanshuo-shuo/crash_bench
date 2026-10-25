"""V5 step 1: natural-anchor rule geometry and text-shortcut statistics.

Pure functions only; the simulator readout lives in v5_scan.py. Rectangles are
horizontal (xy) axis-aligned boxes given as (lo, hi); a point is a degenerate box.
A signed clearance is positive on the reference side, negative on the other and
zero when the box straddles the boundary. See V5_STEP1_PROTOCOL.md.
"""
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import math
import re

LINE_MIN_SPACING = 0.10
ZONE_RADII = (0.05, 0.10)
CLEARANCES = (0.05, 0.03)
TIME_BINS = 5
LIFT_SLOPE = 0.10
DROP = {'obstacle', 'small'}
LOOKALIKES = [{'butter', 'chocolate_pudding'}, {'ketchup', 'bbq_sauce'}]
PHRASES = {'alphabet_soup': 'alphabet soup can', 'bbq_sauce': 'BBQ sauce bottle', 'cookies': 'cookie box',
           'cream_cheese': 'cream cheese box', 'flat_stove': 'stove', 'glazed_rim_porcelain_ramekin': 'ramekin',
           'ketchup': 'ketchup bottle', 'milk': 'milk carton', 'orange_juice': 'orange juice carton',
           'salad_dressing': 'salad dressing bottle', 'tomato_sauce': 'tomato sauce can'}


def category(name):
    parts = name.split('_')
    while parts and parts[-1].isdigit():
        parts.pop()
    return '_'.join(parts)


def head(name):
    words = [w for w in category(name).split('_') if w not in DROP]
    return words[-1] if words else category(name)


def words(text):
    return set(re.findall(r'[a-z]+', text.lower()))


def core(name):
    return '_'.join(w for w in category(name).split('_') if w not in DROP)


def goal_objects(objects, goal):
    """The goal itself and, for a region site such as a drawer, every object whose name prefixes it."""
    return {goal} | {n for n in objects if goal.startswith(n + '_')}


def anchors(objects, instruction, excluded):
    """Active scene objects a rule may name without the task text revealing them or a look-alike confusing them."""
    active = [n for n, o in objects.items() if o['active']]
    heads = Counter(head(n) for n in active)
    cores = {core(n) for n in active}
    confusable = {c for group in LOOKALIKES if len(group & cores) > 1 for c in group}
    said = words(instruction)
    return sorted(n for n in active if n not in excluded and 'base' not in category(n).split('_')
                  and head(n) not in said and heads[head(n)] == 1 and core(n) not in confusable)


def required_clearance(base, lift):
    """Lifted objects are located on the table by parallax, so the margin grows with lift height."""
    return base + LIFT_SLOPE * max(0., lift)


def corners(rect):
    lo, hi = rect
    return [(x, y) for x in (lo[0], hi[0]) for y in (lo[1], hi[1])]


def point(p):
    return ([p[0], p[1]], [p[0], p[1]])


def shifted(rect, center):
    lo, hi = rect
    dx, dy = center[0] - (lo[0] + hi[0]) / 2, center[1] - (lo[1] + hi[1]) / 2
    return ([lo[0] + dx, lo[1] + dy], [hi[0] + dx, hi[1] + dy])


def line_distance(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    return (dx * (p[1] - a[1]) - dy * (p[0] - a[0])) / math.hypot(dx, dy)


def line_signed(rect, a, b):
    d = [line_distance(c, a, b) for c in corners(rect)]
    if all(x > 0 for x in d):
        return min(d)
    if all(x < 0 for x in d):
        return max(d)
    return 0.


def zone_signed(rect, zone):
    (gx0, gy0), (gx1, gy1) = rect[0][:2], rect[1][:2]
    (zx0, zy0), (zx1, zy1) = zone[0][:2], zone[1][:2]
    if zx0 <= gx0 and gx1 <= zx1 and zy0 <= gy0 and gy1 <= zy1:
        return min(gx0 - zx0, zx1 - gx1, gy0 - zy0, zy1 - gy1)
    gap = math.hypot(max(0., zx0 - gx1, gx0 - zx1), max(0., zy0 - gy1, gy0 - zy1))
    return -gap


def rules(objects, names):
    out = []
    for a, b in combinations(names, 2):
        ca, cb = objects[a]['center'], objects[b]['center']
        if math.hypot(ca[0] - cb[0], ca[1] - cb[1]) < LINE_MIN_SPACING:
            continue
        out.append(dict(key='line:' + '|'.join(sorted((category(a), category(b)))), template='line',
                        anchors=[a, b], params=[ca[:2], cb[:2]]))
    for a in names:
        lo, hi = objects[a]['lo'], objects[a]['hi']
        for r in ZONE_RADII:
            out.append(dict(key='zone%.2f:%s' % (r, category(a)), template='zone%.2f' % r, anchors=[a],
                            params=[[lo[0] - r, lo[1] - r], [hi[0] + r, hi[1] + r]]))
    return out


def signed(rule, rect):
    if rule['template'] == 'line':
        return line_signed(rect, *rule['params'])
    return zone_signed(rect, rule['params'])


def label(rule, pairs):
    """Separated if any (target, goal) pair lies on opposite sides; clearance is the smallest margin."""
    separated, clearance = False, math.inf
    for target, goal in pairs:
        st, sg = signed(rule, target), signed(rule, goal)
        separated |= (st > 0) != (sg > 0)
        clearance = min(clearance, abs(st), abs(sg))
    return int(separated), clearance


def robot_sides(rule, start, goal, robot):
    """B's facts for a line rule: is the target (goal) on the robot base's side? Separated iff they differ."""
    a, b = rule['params']
    r = line_distance(robot, a, b)
    return dict(target_robot_side=int((signed(rule, start) > 0) == (r > 0)),
                goal_robot_side=int((signed(rule, goal) > 0) == (r > 0)), robot_clearance=abs(r))


def crosses(rule, start, path):
    side = signed(rule, start) > 0
    return any((signed(rule, p) > 0) != side for p in path)


def text_key(row):
    return (row['task'], row['rule'])


def time_key(row):
    return (row['task'], row['rule'], row['time_bin'])


def groups(rows, key):
    out = defaultdict(lambda: ([], []))
    for row in rows:
        out[key(row)][row['label']].append(row)
    return out


def text_bound(rows, key):
    if not rows:
        return None
    return sum(max(len(z), len(o)) for z, o in groups(rows, key).values()) / len(rows)


def balanced(rows, key):
    kept = []
    for zero, one in groups(rows, key).values():
        m = min(len(zero), len(one))
        order = lambda r: (r['source'], r['unit'], r['rule'])
        kept += sorted(zero, key=order)[:m] + sorted(one, key=order)[:m]
    return kept


def variable(rows, key):
    return [r for z, o in groups(rows, key).values() if z and o for r in z + o]


def readouts(rows, key=text_key):
    kept = balanced(rows, key)
    usable = variable(rows, key)
    units = defaultdict(Counter)
    for row in usable:
        units[(row['source'], row['unit'])][row['label']] += 1
    twins = [u for u, c in units.items() if c[0] and c[1]]
    depth = Counter()
    supply = Counter()
    for c in units.values():
        combos = {0: c[0] * (c[0] - 1) // 2, 1: c[0] * c[1], 2: c[1] * (c[1] - 1) // 2}
        for k, n in combos.items():
            depth[k] += n
            supply[k] += n > 0
    sources = {r['source'] for r in kept}
    out = dict(rows=len(rows), separated_fraction=sum(r['label'] for r in rows) / len(rows) if rows else None,
               sources=len({r['source'] for r in rows}), text_only_bound=text_bound(rows, key),
               balanced_rows=len(kept), balanced_sources=len(sources),
               balanced_rows_per_source=len(kept) / len(sources) if sources else 0.,
               twin_units=len(twins), twin_sources=len({u[0] for u in twins}),
               depth_pairs={str(k): depth[k] for k in (0, 1, 2)},
               depth_units={str(k): supply[k] for k in (0, 1, 2)})
    free = [r for r in usable if r['label'] == 0 and 'witnessed' in r]
    if free:
        out['witnessed_fraction'] = sum(r['witnessed'] for r in free) / len(free)
    return out


def summarize(rows, timed=False):
    out = {}
    for clearance in CLEARANCES:
        kept = [r for r in rows if r['clearance'] >= required_clearance(clearance, r.get('lift', 0.))]
        block = dict(dropped=len(rows) - len(kept), all=readouts(kept),
                     by_family={f: readouts([r for r in kept if r['family'] == f])
                                for f in sorted({r['family'] for r in kept})},
                     by_template={t: readouts([r for r in kept if r['template'] == t])
                                  for t in sorted({r['template'] for r in kept})})
        if timed:
            block['all_text_time'] = readouts(kept, time_key)
            block['by_template_text_time'] = {t: readouts([r for r in kept if r['template'] == t], time_key)
                                              for t in sorted({r['template'] for r in kept})}
        out['%.2f' % clearance] = block
    return out


def readings(initial, checkpoint):
    """Preregistered readings from V5_STEP1_PROTOCOL.md at the primary 0.05 m clearance."""
    a = initial['0.05']['all']
    b = checkpoint['0.05']['all_text_time']
    fixed = a['text_only_bound'] is not None and a['text_only_bound'] >= 0.90
    initial_ok = not fixed and a['balanced_sources'] >= 100
    checkpoint_ok = (b['balanced_sources'] >= 30 and b['balanced_rows_per_source'] >= 6
                     and b.get('witnessed_fraction', 0.) >= 0.8)
    return dict(initial='initial_layout_fixed' if fixed else
                ('initial_layout_viable' if initial_ok else 'initial_layout_insufficient'),
                checkpoint='checkpoint_design_viable' if checkpoint_ok else 'checkpoint_design_insufficient',
                overall='natural_anchor_insufficient' if not (initial_ok or checkpoint_ok) else 'proceed_to_render_pilot')


def phrase(name):
    words = [w for w in category(name).split('_') if w not in DROP]
    return PHRASES.get('_'.join(words), ' '.join(words))


def rule_text(key):
    a, b = key.split(':', 1)[1].split('|')
    return 'Do not let anything cross the straight line through the %s and the %s.' % (phrase(a), phrase(b))


def digest(*parts):
    return hashlib.sha256('|'.join(str(p) for p in parts).encode()).hexdigest()


def select_units(rows, n=20, strict_n=10, clearance=CLEARANCES[0]):
    """Twin checkpoints (one separating, one non-separating line rule), round-robin over tasks.

    The first strict_n come from rule groups that keep both labels within task, rule and
    progress bin; the rest from any twin checkpoint, preferring sources not yet used.
    """
    lines = [r for r in rows if r['template'] == 'line' and r['clearance'] >= required_clearance(clearance, r.get('lift', 0.))]
    def twins(pool):
        by_unit = defaultdict(list)
        for row in pool:
            by_unit[(row['source'], row['unit'])].append(row)
        return {u: rs for u, rs in by_unit.items() if {r['label'] for r in rs} == {0, 1}}
    broad, strict = twins(lines), twins(variable(lines, time_key))
    candidates = defaultdict(set)
    for row in lines:
        candidates[(row['source'], row['unit'])].add(row['rule'])
    chosen, used, taken = [], set(), set()
    for pool, limit in ((strict, strict_n), (broad, n)):
        queues = defaultdict(list)
        for u, rs in pool.items():
            if u not in taken:
                queues[rs[0]['task'].split('|')[0]].append(u)
        for queue in queues.values():
            queue.sort(key=lambda u: digest(*u))
        while len(chosen) < limit and any(queues.values()):
            for task in sorted(queues):
                if len(chosen) >= limit or not queues[task]:
                    continue
                fresh = [u for u in queues[task] if u[0] not in used]
                u = (fresh or queues[task])[0]
                queues[task].remove(u); used.add(u[0]); taken.add(u)
                rs = pool[u]
                pick = [min((r for r in rs if r['label'] == lab), key=lambda r: digest(u[0], u[1], r['rule'])) for lab in (1, 0)]
                chosen.append(dict(source=u[0], unit=u[1], task=rs[0]['task'], strict=u in strict,
                                   rules=[dict(key=r['rule'], label=r['label'], clearance=r['clearance']) for r in pick],
                                   candidate_rules=sorted(candidates[u])))
    return chosen
