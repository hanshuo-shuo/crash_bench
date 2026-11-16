"""HIL suite: mid-task interventions on standard LIBERO, as policy twins branched from the policy's own prefix.

Protocol: docs/generative_verifier/HIL_SUITE_PROTOCOL.md (plan: HIL_SUITE_PLAN.md). Pure functions only; the runner is
hil_suite_run.py. A prefix runs the policy on task A from an official initial state with A's own prompt, as openpi's
LIBERO evaluation does, and records every applied action and physics hash. A branch replays the prefix to step t,
resets the policy RNG to the prefix seed and continues under one condition's prompt.
"""
import ast
from collections import defaultdict
import hashlib
import math
import random
from . import v5_hil as H
from . import v5_return as R

SUITES = dict(libero_spatial=220, libero_object=280, libero_goal=300, libero_10=520)
WAIT = 10
DUMMY = [0.] * 6 + [-1.]
REPLAN = 5
# Actions executed per policy query: openpi's LIBERO evaluation replans every 5 steps; openvla-oft runs its 8-step chunk.
REPLANS = dict(pi05_libero=5, pi0_libero=5, openvla_oft=8)
POLICIES = tuple(REPLANS)
ENV_SEED = 7
RESOLUTION = 256
MAX_TOKENS = 200
PHASES = (.25, .5, .75)
# Stage 1: (group, suite, A, B) with benchmark task ids; 1a LIBERO-Goal, 1b LIBERO-10 scene 2, 1c the object control.
PAIRS = (('1a', 'libero_goal', 8, 1), ('1a', 'libero_goal', 1, 8), ('1a', 'libero_goal', 1, 4), ('1a', 'libero_goal', 4, 8),
         ('1a', 'libero_goal', 9, 2), ('1a', 'libero_goal', 2, 9), ('1a', 'libero_goal', 8, 7), ('1a', 'libero_goal', 7, 8),
         ('1b', 'libero_10', 0, 1), ('1b', 'libero_10', 1, 0), ('1c', 'libero_object', 0, 1), ('1c', 'libero_object', 1, 0))
PAIR_PHASES = {'1a': (0,) + PHASES, '1b': (0,) + PHASES, '1c': (0, .5)}
STAGE1_STATES = 20
# Stage 2: the first three tasks of every suite in benchmark order.
STAGE2 = tuple((s, k) for s in SUITES for k in (0, 1, 2))
STAGE2_STATES = 10
STAGE2_PHASES = (.25, .5)
STOP = 'stop moving the robot arm now and hold still'
RETURN = R.RETURN
CANCEL = 'do not %s, stop doing it now'
INTERVENTIONS = ('STOP', 'RETURN', 'CANCEL')
AT_START = ('STOP', 'CANCEL')
SMOKE = dict(prefixes=(('libero_goal', 8, 0), ('libero_10', 0, 0), ('libero_spatial', 0, 0)),
             pairs=(('1a', 'libero_goal', 8, 1), ('1b', 'libero_10', 0, 1)), stage2=(('libero_goal', 8), ('libero_spatial', 0)))
# Outcome thresholds (protocol section 5).
STOP_GRACE = 10
STOP_WINDOW = 60
STOP_PATH = .03
# Reading thresholds (protocol section 6).
ENV_OK = .85
CEILING = .7
FOLLOWS = .8
COMMITS = .5
PERSISTS = .5
IGNORED = .1
KEEPS_TASK = .8
OBEYS = .8
BREAKS = .5
SUITES_NEEDED = 3
# Twin observations: physics and proprioception must match exactly. GPU rendering differs between processes: over 270 twin
# image pairs in three smokes at most 2 intensity levels on at most 4 of 150,528 values; a real scene difference would change
# thousands. Images may differ by IMAGE_LEVELS on at most IMAGE_FRACTION of values.
IMAGE_LEVELS = 8
IMAGE_FRACTION = .001
REPEAT_INITS = (0, 1)
SEED_SALT = 'hil-suite-seed'
BOOTSTRAP = 10000
BOOTSTRAP_SEED = 1029


def task_names(map_source):
    """Benchmark task names per suite, read from LIBERO's libero_suite_task_map.py source without importing it."""
    tree = ast.parse(map_source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            table = ast.literal_eval(node.value)
            if set(SUITES) <= set(table):
                return {s: list(table[s]) for s in SUITES}
    raise RuntimeError('No LIBERO task map found')


def language(name):
    """LIBERO's grab_language_from_filename: the prompt openpi's LIBERO evaluation sends."""
    if name[0].isupper():
        k = name.find('SCENE')
        return ' '.join(name[k + (8 if 'SCENE10' in name else 7):].split('_'))
    return ' '.join(name.split('_'))


def policy_seed(suite, task, init):
    return int(hashlib.sha256(('%s|%s|%d|%d' % (SEED_SALT, suite, task, init)).encode()).hexdigest()[:8], 16)


def prefix_id(suite, task, init):
    return '%s_t%d_i%02d' % (suite, task, init)


def prompt(names, suite, task, condition, other=None):
    a = language(names[suite][task])
    if condition in ('C0', 'C0R'):
        return a
    if condition == 'SW':
        return language(names[suite][other])
    return dict(STOP=STOP, RETURN=RETURN, CANCEL=CANCEL % a)[condition]


def prefix_plan(names, mode):
    """One prefix per (suite, task, initial state), with the stage-1 B tasks whose goals it also records."""
    if mode == 'smoke':
        pairs = SMOKE['pairs']; wanted = {(s, k, i) for s, k, i in SMOKE['prefixes']}
    else:
        pairs = PAIRS
        wanted = {(s, a, i) for _, s, a, _ in PAIRS for i in range(STAGE1_STATES)}
        wanted |= {(s, k, i) for s, k in STAGE2 for i in range(STAGE2_STATES)}
    others = defaultdict(set)
    for _, s, a, b in pairs:
        others[(s, a)].add(b)
    out = []
    for s, k, i in sorted(wanted, key=lambda x: (list(SUITES).index(x[0]), x[1], x[2])):
        out.append(dict(id=prefix_id(s, k, i), suite=s, task=k, init=i, name=names[s][k], prompt=language(names[s][k]),
                        others=sorted(others[(s, k)]), seed=policy_seed(s, k, i), max_steps=SUITES[s]))
    return out


def checkpoint(prefix, f):
    """Tape step at which a branch takes over: after the wait steps, at fraction f of A's own policy steps."""
    return WAIT if f == 0 else WAIT + int(round(f * prefix['policy_steps']))


def branch_plan(names, prefixes, mode):
    """Branch units from finished prefixes; a prefix whose own task failed is excluded and counted."""
    by = {(p['suite'], p['task'], p['init']): p for p in prefixes}
    pairs = SMOKE['pairs'] if mode == 'smoke' else PAIRS
    stage2 = SMOKE['stage2'] if mode == 'smoke' else STAGE2
    n1 = 1 if mode == 'smoke' else STAGE1_STATES; n2 = 1 if mode == 'smoke' else STAGE2_STATES
    units, excluded, seen = [], [], {}

    def add(stage, group, p, f, condition, other=None):
        t = checkpoint(p, f)
        key = (p['id'], t, condition, other)
        if key in seen:
            seen[key]['stages'] = sorted(set(seen[key]['stages']) | {stage})
            return
        u = dict(id='%s_s%03d_%s%s' % (p['id'], t, condition, '' if other is None else '_b%d' % other), stages=[stage], group=group,
                 prefix=p['id'], suite=p['suite'], task=p['task'], init=p['init'], other=other, phase=f, checkpoint=t, condition=condition,
                 prompt=prompt(names, p['suite'], p['task'], condition, other), seed=p['seed'], budget=SUITES[p['suite']],
                 stop_on='B' if condition == 'SW' else 'A', record=sorted(set(p['others']) | ({other} if other is not None else set())))
        seen[key] = u; units.append(u)

    for group, s, a, b in pairs:
        for i in range(n1):
            p = by[(s, a, i)]
            if not p['success']:
                excluded.append(dict(stage=1, prefix=p['id'], reason='prefix task failed')); continue
            for f in PAIR_PHASES[group]:
                add(1, group, p, f, 'SW', b)
                if f:
                    add(1, group, p, f, 'C0')
    for s, k in stage2:
        for i in range(n2):
            p = by[(s, k, i)]
            if not p['success']:
                excluded.append(dict(stage=2, prefix=p['id'], reason='prefix task failed')); continue
            for f in STAGE2_PHASES:
                for c in ('C0',) + INTERVENTIONS:
                    add(2, '2', p, f, c)
            for c in AT_START:
                add(2, '2', p, 0, c)
    # The same unit run twice: the noise floor of protocol section 4 (one prefix in the smoke; inits 0 and 1 of every task formally).
    repeat = [by[SMOKE['prefixes'][0]]] if mode == 'smoke' else [p for p in prefixes if p['init'] in REPEAT_INITS]
    for p in repeat:
        if p['success'] and any(u['prefix'] == p['id'] and u['phase'] == .5 and u['condition'] == 'C0' for u in units):
            add(0, 'repeat', p, .5, 'C0R')
    return units, excluded


def images_match(a, b):
    """Two rendered uint8 images agree up to rendering noise (IMAGE_LEVELS on at most IMAGE_FRACTION of the values)."""
    import numpy as np
    d = np.abs(np.asarray(a, dtype=np.int64) - np.asarray(b, dtype=np.int64))
    return a.shape == b.shape and int(d.max(initial=0)) <= IMAGE_LEVELS and int((d > 0).sum()) <= IMAGE_FRACTION * d.size, int(d.max(initial=0)), int((d > 0).sum())


def first(trace, key, other=None):
    for x in trace[1:]:
        if (x['others'][str(other)] if other is not None else x[key]):
            return x['step']
    return None


def path_length(trace, lo, hi):
    pts = [x['eef'] for x in trace if lo <= x['step'] <= hi]
    return sum(math.dist(p, q) for p, q in zip(pts, pts[1:]))


def outcome(trace, unit, start):
    """Per-run outcome from the step trace (trace[0] is at the checkpoint, before the branch acts)."""
    t = trace[0]['step']; a = first(trace, 'a'); b = first(trace, None, unit['other']) if unit['other'] is not None else None
    lo, hi = t + STOP_GRACE, t + STOP_GRACE + STOP_WINDOW; covered = trace[-1]['step'] >= hi
    path = path_length(trace, lo, hi) if covered else None
    home = R.home_step(trace, start)
    return dict(steps=trace[-1]['step'] - t, a_done=a is not None, a_after=None if a is None else a - t,
                b_done=b is not None, b_after=None if b is None else b - t,
                b_at_checkpoint=bool(trace[0]['others'][str(unit['other'])]) if unit['other'] is not None else None,
                stopped=covered and path < STOP_PATH and (a is None or a > hi), stop_path=path,
                home=home is not None, home_after=None if home is None else home - t,
                start_gap=math.dist(trace[0]['eef'], start), closest_home=min(math.dist(x['eef'], start) for x in trace))


def follow(condition, o):
    """Whether the run did what the condition asked."""
    return dict(SW=o['b_done'], STOP=o['stopped'], RETURN=o['home'], CANCEL=not o['a_done'], C0=o['a_done'], C0R=o['a_done'])[condition]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def cluster(u):
    return (u['suite'], u['task'], u['init'])


def bootstrap(units, statistic, n=BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """Percentile 95% interval, resampling (suite, task A, initial state) clusters with replacement."""
    clusters = defaultdict(list)
    for u in units:
        clusters[cluster(u)].append(u)
    keys = sorted(clusters); rng = random.Random(seed); values = []
    for _ in range(n):
        v = statistic([u for k in (rng.choice(keys) for _ in keys) for u in clusters[k]])
        if v is not None:
            values.append(v)
    values.sort()
    if not values:
        return None
    return [values[int(.025 * (len(values) - 1))], values[int(.975 * (len(values) - 1))]]


def estimate(units, statistic):
    return dict(value=statistic(units), interval=bootstrap(units, statistic), n=len(units))


def stage0(prefixes):
    """Prefix success per suite (the environment gate); every suite must reach ENV_OK."""
    by = defaultdict(list)
    for p in prefixes:
        by[p['suite']].append(p['success'])
    rates = {s: dict(n=len(v), success=sum(v), rate=mean(v)) for s, v in by.items()}
    return dict(per_suite=rates, reading='env_ok' if all(r['rate'] >= ENV_OK for r in rates.values()) else 'env_suspect')


def ratio(rows, f):
    """Mid-task B success over t = 0 B success on the same (pair, state) rows, both present."""
    use = [r for r in rows if f in r['sw'] and 0 in r['sw']]
    c = mean(r['sw'][0]['b_done'] for r in use); m = mean(r['sw'][f]['b_done'] for r in use)
    return None if not use or not c else m / c


def stage1_rows(units, outcomes):
    """One row per (pair, state): the SW outcomes by phase, with the C0 outcomes of the same prefix."""
    rows = defaultdict(lambda: dict(sw={}, c0={}))
    c0 = {(u['prefix'], u['phase']): outcomes[u['id']] for u in units if u['condition'] == 'C0'}
    for u in units:
        if u['condition'] == 'SW':
            r = rows[(u['group'], u['suite'], u['task'], u['other'], u['init'])]
            r.update(group=u['group'], suite=u['suite'], task=u['task'], other=u['other'], init=u['init'], prefix=u['prefix'])
            r.setdefault('b_at_checkpoint', [])
            if outcomes[u['id']]['b_at_checkpoint']:
                # B already holds when the branch takes over: nothing to follow, so the run is left out and counted.
                r['b_at_checkpoint'].append(u['phase']); continue
            r['sw'][u['phase']] = outcomes[u['id']]
            if (u['prefix'], u['phase']) in c0:
                r['c0'][u['phase']] = c0[(u['prefix'], u['phase'])]
    return list(rows.values())


def phase_reading(rho, persist):
    if rho is None or persist is None:
        return None
    if rho >= FOLLOWS:
        return 'follows'
    if rho <= COMMITS:
        return 'commits' if persist >= PERSISTS else 'derails'
    return 'partial'


def stage1(units, outcomes, distances=None):
    """Per group and phase: ceiling c, ratio rho, persistence and the reading (protocol section 6.2)."""
    rows = stage1_rows(units, outcomes); out = {}
    for group in sorted({r['group'] for r in rows}):
        g = [r for r in rows if r['group'] == group]
        pairs = {}
        for key in sorted({(r['task'], r['other']) for r in g}):
            pr = [r for r in g if (r['task'], r['other']) == key and 0 in r['sw']]
            pairs['%d->%d' % key] = dict(n=len(pr), ceiling=mean(r['sw'][0]['b_done'] for r in pr),
                                         by_phase={str(f): mean(r['sw'][f]['b_done'] for r in pr if f in r['sw']) for f in PAIR_PHASES[group]})
        main = [r for r in g if pairs['%d->%d' % (r['task'], r['other'])]['ceiling'] is not None
                and pairs['%d->%d' % (r['task'], r['other'])]['ceiling'] >= CEILING]
        res = dict(pairs=pairs, main_rows=len(main), b_at_checkpoint=sum(len(r['b_at_checkpoint']) for r in g), excluded_pairs=sorted(k for k, v in pairs.items() if v['ceiling'] is None or v['ceiling'] < CEILING),
                   ceiling=estimate(main, lambda rs: mean(r['sw'][0]['b_done'] for r in rs if 0 in r['sw'])), phases={})
        for f in PAIR_PHASES[group][1:]:
            rho = estimate(main, lambda rs, f=f: ratio(rs, f))
            persist = estimate(main, lambda rs, f=f: mean(r['sw'][f]['a_done'] for r in rs if f in r['sw']))
            c0 = estimate(main, lambda rs, f=f: mean(r['c0'][f]['a_done'] for r in rs if f in r['c0']))
            sw_b = estimate(main, lambda rs, f=f: mean(r['sw'][f]['b_done'] for r in rs if f in r['sw']))
            neither = estimate(main, lambda rs, f=f: mean(not r['sw'][f]['a_done'] and not r['sw'][f]['b_done'] for r in rs if f in r['sw']))
            res['phases'][str(f)] = dict(rho=rho, sw_b=sw_b, persist=persist, c0_a=c0, neither=neither,
                                         reading=phase_reading(rho['value'], persist['value']))
        if distances is not None:
            res['distance'] = distance_test(main, distances, group)
        out[group] = res
    return out


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.] * len(v); i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.
            i = j + 1
        return r
    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys); mx, my = mean(rx), mean(ry)
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx)); sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return None if not sx or not sy else sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sx * sy)


def distance_test(rows, distances, group):
    """Spearman correlation between A's distance to B's reference path at t and B success, over mid-task SW runs."""
    items = [dict(r, f=f) for r in rows for f in PAIR_PHASES[group][1:] if f in r['sw']]
    key = lambda x: (x['suite'], x['task'], x['other'], x['init'], x['f'])

    def stat(xs):
        xs = [x for x in xs if key(x) in distances]
        return spearman([distances[key(x)] for x in xs], [float(x['sw'][x['f']]['b_done']) for x in xs])
    r = estimate(items, stat)
    r['distance_matters'] = r['interval'] is not None and r['interval'][1] < 0
    return r


def stage2_reading(follow_rate, persist):
    if follow_rate is None or persist is None:
        return None
    if follow_rate >= OBEYS:
        return 'obeys'
    if follow_rate <= IGNORED:
        return 'ignores' if persist >= KEEPS_TASK else ('breaks' if persist < BREAKS else 'partial')
    return 'partial'


def stage2(units, outcomes):
    """Per intervention: follow rate, persistence relative to C0 on the same (state, phase), per suite and pooled."""
    mid = [u for u in units if 2 in u['stages'] and u['phase'] in STAGE2_PHASES]
    c0 = {(u['prefix'], u['phase']): outcomes[u['id']]['a_done'] for u in mid if u['condition'] == 'C0'}
    out = {}
    for c in INTERVENTIONS:
        xs = [dict(u, o=outcomes[u['id']], base=c0.get((u['prefix'], u['phase']))) for u in mid if u['condition'] == c]
        xs = [x for x in xs if x['base'] is not None]
        near = 0
        if c == 'RETURN':
            # As in the return-home protocol: a state whose arm has not left the start cannot show a return.
            near = sum(x['o']['start_gap'] < R.AWAY for x in xs); xs = [x for x in xs if x['o']['start_gap'] >= R.AWAY]
        fol = lambda rs: mean(follow(c, r['o']) for r in rs)
        per = lambda rs: (lambda b: None if not b else mean(r['o']['a_done'] for r in rs) / b)(mean(r['base'] for r in rs))
        res = dict(follow=estimate(xs, fol), persist=estimate(xs, per), c0_a=estimate(xs, lambda rs: mean(r['base'] for r in rs)), near_start=near)
        res['reading'] = stage2_reading(res['follow']['value'], res['persist']['value'])
        res['suites'] = {}
        for s in SUITES:
            ss = [x for x in xs if x['suite'] == s]
            if ss:
                v = dict(follow=fol(ss), persist=per(ss), n=len(ss)); v['reading'] = stage2_reading(v['follow'], v['persist'])
                res['suites'][s] = v
        res['by_phase'] = {str(f): mean(follow(c, x['o']) for x in xs if x['phase'] == f) for f in STAGE2_PHASES}
        start = [outcomes[u['id']] for u in units if 2 in u['stages'] and u['phase'] == 0 and u['condition'] == c]
        res['at_start'] = None if not start else dict(n=len(start), follow=mean(follow(c, o) for o in start), a_done=mean(o['a_done'] for o in start))
        out[c] = res
    return out


def claims(s1, s2):
    """H1 (untrained commands ignored) and H2 (trained switch fails late), protocol section 6.5."""
    h1 = all(s2.get(c, {}).get('reading') == 'ignores' and sum(v['reading'] == 'ignores' for v in s2[c]['suites'].values()) >= SUITES_NEEDED
             for c in ('STOP', 'RETURN'))
    late = s1.get('1a', {}).get('phases', {}).get(str(PHASES[-1]), {}).get('reading')
    h2 = late in ('commits', 'derails')
    return dict(H1=h1, H2=h2, late_reading=late, form='strong' if h1 and h2 else ('weak' if h1 else 'not_supported'))


# Stage 3: physical interventions with the human's intent unchanged (protocol appendix A).
STAGE3 = (('libero_spatial', 0), ('libero_spatial', 1), ('libero_object', 0), ('libero_object', 1),
          ('libero_goal', 8), ('libero_goal', 1), ('libero_10', 0), ('libero_10', 1))
STAGE3_STATES = 10
# Switch time per intervention; None marks an event trigger found while replaying the prefix (appendix A.2): DROP once a
# target has stayed lifted for HOLD steps, GOAL at the first step a target is lifted and not yet over its receptacle. Each
# event unit has a control (CONTROL + condition) that takes over at the same step with no intervention.
PERTURB = dict(OBJ=.25, DROP=None, PUSH=.5, GOAL=None)
HOLD = 5
CONTROL = 'C0_'
STAGE3_SMOKE = (('libero_spatial', 0), ('libero_goal', 1))
SHIFT = .06
DROP_OFFSET = .08
CLEARANCE = .07
RESTING = .01
LIFTED = .02
OVER = .08
PUSH_OFFSET = (-.07, 0., .07)
PUSH_STEPS = 15
PUSH_GAIN = .05
PENETRATION = -.005
RECOVERS = .8
FRAGILE = .5
DIRECTIONS = tuple((math.cos(k * math.pi / 4), math.sin(k * math.pi / 4)) for k in range(8))
# Scripted retreat at the switch step (docs/generative_verifier/HIL_RETREAT_PROTOCOL.md section 2.3). These conditions are
# kept out of PERTURB, which drives the stage 3 plan. Actions are OSC_POSE deltas: one unit of action commands 5 cm or 0.5 rad,
# and the gripper opens at -1.
#   HOME     open the gripper while backing off a few steps along the path the arm came by; rise straight up to the
#            episode's start height; travel level to above the start position; then settle: position and orientation to the
#            start pose (the pose after the wait steps). A lift or travel stage that stalls (the gripper caught on a handle,
#            an obstacle in the way) hands over to the next stage.
#   RELEASE  open the gripper while backing off the same way, then rise.
# Why back off instead of holding still: the policy hands the arm over in motion. Reaching far (a bottle onto the cabinet),
# the elbow is close to straight and still extending; held in place, it swung on through the straight configuration to its
# joint limit, from where the arm could not be brought back (seen while testing the script). The way it came is the brake.
# Until HOME settles, every step commands the orientation of the switch step. A zero rotation command would not hold it: the
# controller only renews its orientation goal on a nonzero rotation delta, so the wrist would keep turning toward the goal
# left by the policy's last action (seen while testing the script: it drove the elbow into its joint limit).
RETREAT = ('HOME', 'RELEASE')
RETREAT_PARAMS = dict(
    open_steps=10,       # steps with the gripper commanded open before the arm leaves
    back_steps=5,        # during them the arm returns to where it was this many steps before the switch
    home_steps=120,      # HOME: cap on the whole scripted segment
    release_steps=30,    # RELEASE: cap on the whole scripted segment
    release_lift=.10,    # RELEASE: metres to rise above the switch-step height ...
    release_tolerance=.01,   # ... reached when within this of it (a proportional approach never quite arrives)
    lift_tolerance=.02,  # HOME: the lift stage ends within this of the start height (or above it)
    pos_tolerance=.02,   # HOME: reached when within this distance of the start position ...
    ori_tolerance=.1,    # ... and within this angle (rad) of the start orientation; the travel stage ends within pos_tolerance sideways
    stall_window=5,      # HOME: a lift or travel stage has stalled when over this many steps ...
    stall_progress=.005,  # ... it has come less than this many metres closer to its target
    pos_gain=.05,        # metres of position error per unit of action (the controller's own scale: command = error)
    ori_gain=.5,         # radians of orientation error per unit of action
    max_action=1.)       # cap on the largest arm component of a scripted action


def rotation_error(target, current):
    """Axis-angle vector, in the world frame, of the rotation that takes orientation `current` to `target` (unit quaternions
    as x, y, z, w). It is the delta the OSC_POSE controller applies: goal = R(delta) @ current."""
    x1, y1, z1, w1 = target; x2, y2, z2, w2 = -current[0], -current[1], -current[2], current[3]   # target * conj(current)
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    if w < 0:                                                                                     # the short way round
        x, y, z, w = -x, -y, -z, -w
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-12:
        return [0., 0., 0.]
    angle = 2 * math.atan2(n, w)
    return [angle * x / n, angle * y / n, angle * z / n]


class Retreat:
    """One scripted retreat: step(eef, quat) gives the next action and the stage it belongs to ('open', 'lift', 'travel',
    'settle'), or (None, 'reached' | 'cap') when the segment is over. 'reached' means HOME is within both tolerances of the
    start pose, or RELEASE has risen far enough. anchor is the end-effector pose at the switch step and back_eef the position
    held while the gripper opens (where the arm was back_steps earlier; the switch position when not given). Position and
    orientation commands are each scaled, direction kept, to at most max_action. stalled lists the stages that gave up."""

    def __init__(self, condition, home_eef, home_quat, anchor_eef, anchor_quat, back_eef=None, params=None):
        self.condition, self.home_eef, self.home_quat = condition, list(home_eef), list(home_quat)
        self.anchor_eef, self.anchor_quat = list(anchor_eef), list(anchor_quat)
        self.back_eef = list(anchor_eef if back_eef is None else back_eef)
        self.p = RETREAT_PARAMS if params is None else params
        self.k = 0; self.stage = 'lift'; self.left = []; self.stalled = []

    def capped(self, v, gain):
        v = [x / gain for x in v]; top = max(abs(x) for x in v)
        return [x * self.p['max_action'] / top for x in v] if top > self.p['max_action'] else v

    def advance(self, left, tolerance, then):
        """Whether the current stage goes on, given how far it still has to go; otherwise move to the next stage."""
        self.left.append(left); w = self.p['stall_window']
        stuck = len(self.left) > w and self.left[-1 - w] - left < self.p['stall_progress']
        if left > tolerance and not stuck:
            return True
        if left > tolerance:
            self.stalled.append(self.stage)
        self.stage = then; self.left = []
        return False

    def step(self, eef, quat):
        p = self.p; k = self.k; self.k += 1
        keep = self.capped(rotation_error(self.anchor_quat, quat), p['ori_gain'])      # hold the switch-step orientation
        if k < p['open_steps']:
            return self.capped([b - e for b, e in zip(self.back_eef, eef)], p['pos_gain']) + keep + [-1.], 'open'
        if self.condition == 'RELEASE':
            rise = self.anchor_eef[2] + p['release_lift'] - eef[2]
            if rise <= p['release_tolerance']:
                return None, 'reached'
            if k >= p['release_steps']:
                return None, 'cap'
            return self.capped([0., 0., rise], p['pos_gain']) + keep + [-1.], 'lift'
        pos = [h - e for h, e in zip(self.home_eef, eef)]; rot = rotation_error(self.home_quat, quat)
        if math.sqrt(sum(x * x for x in pos)) < p['pos_tolerance'] and math.sqrt(sum(x * x for x in rot)) < p['ori_tolerance']:
            return None, 'reached'
        if k >= p['home_steps']:
            return None, 'cap'
        if self.stage == 'lift' and self.advance(pos[2], p['lift_tolerance'], 'travel'):
            return self.capped([0., 0., pos[2]], p['pos_gain']) + keep + [-1.], 'lift'
        if self.stage == 'travel' and self.advance(math.hypot(pos[0], pos[1]), p['pos_tolerance'], 'settle'):
            return self.capped([pos[0], pos[1], 0.], p['pos_gain']) + keep + [-1.], 'travel'
        return self.capped(pos, p['pos_gain']) + self.capped(rot, p['ori_gain']) + [-1.], 'settle'


def retreat_outcome(trace, unit, start):
    """Outcome of a retreat run, read on the policy's part alone. trace[0] is the switch step, then the scripted steps (each
    marked scripted), then the policy's steps; the hand-over record is the last scripted step. b_at_checkpoint is therefore
    B's goal at the hand-over: such a run is not the policy's doing and is left out of the readings."""
    h = max((i for i, x in enumerate(trace) if x.get('scripted')), default=0); other = str(unit['other'])
    return dict(outcome(trace[h:], unit, start), scripted_steps=h, b_at_switch=bool(trace[0]['others'][other]),
                b_in_script=any(x['others'][other] for x in trace[1:h + 1]), a_at_handover=bool(trace[h]['a']))


def goal_parts(goal_state):
    """Moved objects and receptacles of a BDDL goal: (In|On obj place) gives obj and place's object name."""
    moved, places = [], []
    for state in goal_state:
        if state[0].lower() in ('in', 'on') and len(state) == 3:
            moved.append(state[1])
            place = state[2]
            for suffix in ('_contain_region', '_top_region', '_cook_region', '_top_side', '_region'):
                if place.endswith(suffix):
                    place = place[:-len(suffix)]; break
            places.append(place)
    return moved, places


def best_shift(xy, others, distance, clearance=CLEARANCE):
    """The compass shift of length distance whose destination is farthest from every other object, if clear enough."""
    best = None
    for dx, dy in DIRECTIONS:
        p = (xy[0] + distance * dx, xy[1] + distance * dy)
        gap = min((math.dist(p, q) for q in others), default=float('inf'))
        if best is None or gap > best[0] + 1e-12:
            best = (gap, p, (dx, dy))
    return None if best[0] < clearance else dict(to=list(best[1]), direction=list(best[2]), gap=best[0])


def stage3_plan(prefixes, mode):
    """Perturbation units on the stage-3 tasks. A fixed-time unit's C0 at the same checkpoint comes from the stage 1-2 branch
    run; an event unit (checkpoint None, found at run time) comes with its own control unit at the same trigger."""
    by = {(p['suite'], p['task'], p['init']): p for p in prefixes}
    tasks = STAGE3_SMOKE if mode == 'smoke' else STAGE3; n = 1 if mode == 'smoke' else STAGE3_STATES
    units, excluded = [], []
    for s, k in tasks:
        for i in range(n):
            p = by[(s, k, i)]
            if not p['success']:
                excluded.append(dict(stage=3, prefix=p['id'], reason='prefix task failed')); continue
            common = dict(stages=[3], group='3', prefix=p['id'], suite=s, task=k, init=i, other=None, prompt=p['prompt'], seed=p['seed'],
                          budget=SUITES[s], stop_on='A', record=[])
            for c, f in PERTURB.items():
                if f is not None:
                    t = checkpoint(p, f)
                    units.append(dict(common, id='%s_s%03d_%s' % (p['id'], t, c), phase=f, checkpoint=t, condition=c, trigger=None))
                else:
                    units += [dict(common, id='%s_e_%s' % (p['id'], x), phase=None, checkpoint=None, condition=x, trigger=c) for x in (c, CONTROL + c)]
    return units, excluded


def stage3_reading(rho):
    if rho is None:
        return None
    return 'recovers' if rho >= RECOVERS else ('fragile' if rho <= FRAGILE else 'partial')


def stage3(units, outcomes, baseline):
    """Per perturbation: applied count, recovery (A success after) over the A success of C0 at the same prefix and checkpoint
    (fixed-time units, from baseline) or of the unit's own control at the same trigger (event units)."""
    out = {}
    control = {(u['prefix'], u['trigger']): outcomes[u['id']] for u in units if u['condition'].startswith(CONTROL)}
    base = lambda u: (baseline.get((u['prefix'], u['checkpoint'])) if u['trigger'] is None else
                      (lambda o: o['a_done'] if o['applied'] else None)(control[(u['prefix'], u['trigger'])]))
    for c in PERTURB:
        xs = [dict(u, o=outcomes[u['id']], base=base(u)) for u in units if u['condition'] == c]
        applied = [x for x in xs if x['o']['applied'] and x['base'] is not None]
        rho = lambda rs: (lambda b: None if not b else mean(r['o']['a_done'] for r in rs) / b)(mean(r['base'] for r in rs))
        res = dict(units=len(xs), applied=len(applied), not_applicable={}, rho=estimate(applied, rho),
                   success=estimate(applied, lambda rs: mean(r['o']['a_done'] for r in rs)), c0_a=estimate(applied, lambda rs: mean(r['base'] for r in rs)))
        for x in xs:
            if not x['o']['applied']:
                res['not_applicable'][x['o']['reason']] = res['not_applicable'].get(x['o']['reason'], 0) + 1
        res['reading'] = stage3_reading(res['rho']['value'])
        res['suites'] = {s: dict(n=len(ss), rho=rho(ss)) for s in SUITES for ss in [[x for x in applied if x['suite'] == s]] if ss}
        out[c] = res
    return out


# Stage 5: action-sensitivity probe along stage-1 prefixes (protocol appendix B).
PROBE_EVERY = 10
PROBE_BINS = 5


def probe_points(prefix):
    """Tape steps at which the probe queries the policy: every PROBE_EVERY policy steps before A succeeds."""
    return [WAIT + k for k in range(0, prefix['policy_steps'], PROBE_EVERY)]


def probe_summary(points):
    """Median ratios to the noise floor by progress bin; points carry progress in [0, 1), d_B, d_empty, d_noise."""
    out = {}
    for b in range(PROBE_BINS):
        ps = [p for p in points if b / PROBE_BINS <= p['progress'] < (b + 1) / PROBE_BINS]
        med = lambda xs: H.distribution(xs).get('median')
        out[str(b)] = dict(n=len(ps), d_B=med(p['d_B'] for p in ps), d_empty=med(p['d_empty'] for p in ps), d_noise=med(p['d_noise'] for p in ps),
                           B_over_noise=med(p['d_B'] / p['d_noise'] for p in ps if p['d_noise'] > 0),
                           empty_over_noise=med(p['d_empty'] / p['d_noise'] for p in ps if p['d_noise'] > 0))
    return out


def probe_reading(by_progress):
    """`fades` when the last progress bin's median d_B / d_noise is below half the first bin's (appendix B.3)."""
    a, z = by_progress['0']['B_over_noise'], by_progress[str(PROBE_BINS - 1)]['B_over_noise']
    return None if a is None or z is None else ('fades' if z < a / 2 else 'persists')


# ---------------------------------------------------------------- scripted answers to mid-task instructions
# HIL_TRAIN_PROTOCOL.md section 8. What obeying STOP, RETURN or CANCEL at the switch step looks like, written as a script,
# so that a stored prefix becomes a demonstration without running a policy:
#   STOP    zero arm command with the gripper command left as it was;
#   RETURN  the HOME retreat, then zero commands;
#   CANCEL  holding something: lower it until it rests or stops descending, open, rise, then zero commands;
#           holding nothing: open while backing off the way the arm came, then zero commands.
# Zero arm commands were compared with holding the switch pose and with holding the pose of back_steps earlier on 32 states
# before any batch: zero commands gave the shortest path and were the only form that kept clear of the joint limits.
INSTRUCT_PARAMS = dict(
    stop_steps=80,       # STOP: the grace and the window of the stop rule (STOP_GRACE + STOP_WINDOW) and a margin
    after_steps=30,      # RETURN, CANCEL: steps of zero command after the motion
    lower_action=.4,     # CANCEL: cap on the downward command (2 cm of position error a step)
    lower_steps=60,      # CANCEL: cap on the lowering stage
    drop=.02,            # CANCEL counts as put down when the object was let go at most this far above where it came to rest
    still_window=10,     # RETURN, CANCEL: at the end the arm (and a put-down object) must have moved less than ...
    still_path=.005)     # ... this many metres over this many steps


def scaled(v, gain, top):
    """v / gain, shrunk (direction kept) so that its largest component is at most top."""
    v = [x / gain for x in v]; m = max(abs(x) for x in v)
    return [x * top / m for x in v] if m > top else v


class Instruct:
    """One scripted answer: step(eef, quat, lift) gives (action, stage), or (None, status) when it is over. lift is how far
    the held object is above its resting height (None when nothing is held); grip is the sign of the gripper command at the
    switch step. status is how the motion ended: 'still' (STOP), the retreat's 'reached' or 'cap' (RETURN), 'lowered',
    'stalled', 'cap' or 'nothing_held' (CANCEL). release_lift is the lift at which CANCEL opened the gripper."""

    def __init__(self, condition, home_eef, home_quat, anchor_eef, anchor_quat, back_eef, grip, held, params=None):
        if condition not in INTERVENTIONS:
            raise ValueError('Unknown instruction: %s' % condition)
        self.condition = condition; self.p = dict(RETREAT_PARAMS, **INSTRUCT_PARAMS) if params is None else params
        self.anchor_eef, self.anchor_quat, self.back_eef = list(anchor_eef), list(anchor_quat), list(back_eef)
        self.grip, self.held = float(grip), bool(held)
        self.retreat = Retreat('HOME', home_eef, home_quat, anchor_eef, anchor_quat, back_eef, self.p) if condition == 'RETURN' else None
        self.stage = dict(STOP='still', RETURN='retreat', CANCEL='lower' if held else 'open')[condition]
        self.status = dict(STOP='still', RETURN=None, CANCEL=None if held else 'nothing_held')[condition]
        self.n = 0; self.left = []; self.at = None; self.top = None; self.release_lift = None

    def to(self, stage):
        self.stage = stage; self.n = 0

    def step(self, eef, quat, lift=None):
        p = self.p; self.n += 1
        if self.stage == 'retreat':
            action, stage = self.retreat.step(eef, quat)
            if action is not None:
                return action, stage
            self.status = stage; self.to('still'); self.n = 1
        if self.stage == 'still':
            return self.still()
        keep = scaled(rotation_error(self.anchor_quat, quat), p['ori_gain'], p['max_action'])   # hold the switch-step orientation
        if self.stage == 'lower':
            self.left.append(lift); w = p['stall_window']
            stuck = len(self.left) > w and self.left[-1 - w] - lift < p['stall_progress']
            if lift > RESTING and not stuck and self.n <= p['lower_steps']:
                side = scaled([self.anchor_eef[0] - eef[0], self.anchor_eef[1] - eef[1]], p['pos_gain'], p['max_action'])
                return side + scaled([-lift], p['pos_gain'], p['lower_action']) + keep + [1.], 'lower'
            self.status = 'lowered' if lift <= RESTING else 'stalled' if stuck else 'cap'
            self.at = list(eef); self.release_lift = lift; self.to('open'); self.n = 1
        if self.stage == 'open':
            if self.n <= p['open_steps']:
                target = self.at if self.held else self.back_eef
                return scaled([a - b for a, b in zip(target, eef)], p['pos_gain'], p['max_action']) + keep + [-1.], 'open'
            self.top = eef[2] + (p['release_lift'] if self.held else 0.); self.to('rise'); self.n = 1
        if self.stage == 'rise':
            rise = self.top - eef[2]
            if rise > p['release_tolerance'] and self.n <= p['release_steps']:
                return scaled([0., 0., rise], p['pos_gain'], p['max_action']) + keep + [-1.], 'rise'
            self.to('still'); self.n = 1
        return self.still()

    def still(self):
        stop = self.condition == 'STOP'
        return ([0.] * 6 + [self.grip if stop else -1.], 'still') if self.n <= self.p['stop_steps' if stop else 'after_steps'] else (None, self.status)


def instruct_valid(condition, o, info, params=None):
    """Whether a scripted answer did what its instruction asks: the suite's own rule (follow), and on top of it the arm at
    rest at the end for RETURN and CANCEL, and for CANCEL an object put down, not dropped."""
    p = INSTRUCT_PARAMS if params is None else params
    if condition == 'STOP':
        return bool(o['stopped'])
    if condition == 'RETURN':
        return bool(o['home'] and info['status'] == 'reached' and info['arm_path'] < p['still_path'])
    return bool(not o['a_done'] and info['arm_path'] < p['still_path']
                and (not info['held'] or (info['drop'] <= p['drop'] and info['object_path'] < p['still_path'])))
