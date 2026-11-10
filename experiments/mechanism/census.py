"""Task-pair census of the mechanism pilot (docs/generative_verifier/HIL_MECHANISM_CENSUS.md): plan and summary.

Every ordered pair (A, B) of the ten LIBERO-Goal tasks (one shared scene) on the official initial states 0-9. Fresh prefixes
run each task A with its own prompt (verifier_dataset/hil_suite_run.py, prefix phase, every other Goal task's goal recorded
at every step). A branch replays A's prefix to the switch step (the wait steps, or half / three quarters of A's own policy
steps), then continues under prompt B with the prefix seed (the HIL suite stage 1 SW condition) under one arm of
serve_steer.py: native, or amp2 (the closed-loop repair, fixed beforehand: layers 12-14, image tokens, B - A at lam 2).
f = 0 runs native only (the ceiling and the start of the image-route fade); f = .5 and .75 run both arms.

  census.py prefixes NAMES_JSON OUT_JSON                         the prefix plan (10 tasks x inits 0-9)
  census.py plan PREFIX_ROOT SMOKE_ROOT FORMAL_ROOT STAGE1_ROOT  branch plans of a finished prefix run (read-only on it)
  census.py summarize ROOT [--require-tool-ok]                   tool check, readings and tables -> ROOT/SUMMARY.json
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import closed_loop as C  # noqa: E402
from verifier_dataset import hil_suite as S  # noqa: E402

SUITE = 'libero_goal'
TASKS = tuple(range(10))
INITS = tuple(range(10))
SMOKE_INIT, SMOKE_SHIFT = 0, 3       # smoke: init 0, B = (A + 3) mod 10 (every task once as A and once as B)
PHASES = (0, .5, .75)
LAYERS = C.LAYERS
ARMS = dict(native=('native', 1.), amp2=('amp', 2.))
PHASE_ARMS = {0: ('native',), .5: ('native', 'amp2'), .75: ('native', 'amp2')}
SHARDS = dict(smoke=2, formal=6)
PILOT = ('1->8', '7->8', '2->9')     # pairs whose open-loop data chose the window
# The object each task's BDDL goal moves or operates (test_census checks it against the BDDL files).
MOVED = {0: 'cabinet_middle_drawer', 1: 'bowl', 2: 'wine_bottle', 3: 'bowl', 4: 'bowl', 5: 'plate', 6: 'cream_cheese', 7: 'stove',
         8: 'bowl', 9: 'wine_bottle'}
CLOSED = .05            # gripper aperture (left - right finger joint) below which the gripper counts as closed (open ~.08)
# Readings (protocol section 5).
MIN_CELL = 5            # states a cell needs to be read
CEILING_AT = .5         # a pair is eligible when B from the start (f = 0, native) succeeds at least this often
FAILURE_AT, SUCCESS_AT = .5, .9
COMMON, RARE = .25, .10
GAIN, HARM = C.GAIN, C.HARM
RIDGE = 1.
DRAWS, SEED = 2000, 0


def pair_label(a, b):
    return '%d->%d' % (a, b)


# ---------------------------------------------------------------- plans
def prefix_plan(names):
    out = []
    for k in TASKS:
        for i in INITS:
            out.append(dict(id=S.prefix_id(SUITE, k, i), suite=SUITE, task=k, init=i, name=names[SUITE][k], prompt=S.language(names[SUITE][k]),
                            others=[b for b in TASKS if b != k], seed=S.policy_seed(SUITE, k, i), max_steps=S.SUITES[SUITE]))
    return out


def sw_id(prefix, t, b):
    return '%s_s%03d_SW_b%d' % (prefix['id'], t, b)


def branch_units(prefixes, tapes, mode):
    """(units, excluded): every SW unit of the census on successful prefixes, once per arm of its phase. A unit whose B goal
    already holds at the switch step on the prefix tape is excluded and counted, as is every unit of a failed prefix."""
    prompts = {p['task']: p['prompt'] for p in prefixes}
    if set(prompts) != set(TASKS) or len(set(prompts.values())) != len(TASKS):
        raise RuntimeError('Every Goal task needs its own prompt')
    units, excluded = [], []
    for p in sorted(prefixes, key=lambda p: (p['task'], p['init'])):
        if mode == 'smoke' and p['init'] != SMOKE_INIT:
            continue
        others = [(p['task'] + SMOKE_SHIFT) % 10] if mode == 'smoke' else [b for b in TASKS if b != p['task']]
        if not p['success']:
            excluded += [dict(prefix=p['id'], other=b, reason='prefix task failed') for b in others]
            continue
        tape = tapes[p['id']]
        for b in others:
            for f in PHASES:
                t = S.checkpoint(p, f); step = tape[t]
                if step['step'] != t or not S.WAIT <= t < tape[-1]['step']:
                    raise RuntimeError('Checkpoint outside the prefix tape: %s %s' % (p['id'], f))
                if step['others'][str(b)]:
                    excluded.append(dict(prefix=p['id'], other=b, phase=f, reason='B holds at the switch step')); continue
                sw = sw_id(p, t, b); aperture = float(step['gripper'][0] - step['gripper'][1])
                covariates = dict(aperture=aperture, gripper_closed=aperture < CLOSED, same_object=MOVED[p['task']] == MOVED[b])
                for name in PHASE_ARMS[f]:
                    kind, lam = ARMS[name]
                    arm = dict(name=name, kind=kind, lam=lam, layers=list(LAYERS), prompt_a=p['prompt'], direction_seed=C.direction_seed(sw))
                    units.append(dict(id=sw + '__' + name, sw=sw, group='1a', stages=[1], suite=SUITE, task=p['task'], other=b, init=p['init'],
                                      phase=f, checkpoint=t, condition='SW', prompt=prompts[b], seed=p['seed'], budget=S.SUITES[SUITE],
                                      stop_on='B', record=[b], prefix=p['id'], pair=pair_label(p['task'], b), arm=arm, covariates=covariates))
    return units, excluded


def finished_prefix(prefix_root):
    done = C.read(prefix_root/'PREFIX_COMPLETE.json')
    if not done['passed'] or done['mode'] != 'formal':
        raise RuntimeError('The census needs a finished formal prefix run')
    if C.read(prefix_root/'STAGE0.json')['reading'] != 'env_ok':
        raise RuntimeError('Prefix success read env_suspect: stop and check the environment')
    hashes = C.read(prefix_root/'PREFIX_HASHES.json')
    for n, digest in hashes.items():
        if C.sha(prefix_root/n) != digest:
            raise RuntimeError('Prefix run output changed: ' + n)
    return hashes


def load_tapes(prefix_root, prefixes):
    from verifier_dataset import hil_suite_run as H
    return {p['id']: H.load(Path(prefix_root)/'prefix'/p['id'])[1] for p in prefixes}


def plan(prefix_root, mode, stage1_root, tapes=None):
    prefix_root = Path(prefix_root); hashes = finished_prefix(prefix_root)
    prefixes = C.read(prefix_root/'PREFIXES.json')
    if {(p['task'], p['init']) for p in prefixes} != {(k, i) for k in TASKS for i in INITS} or any(p['suite'] != SUITE for p in prefixes):
        raise RuntimeError('The prefix run is not the census prefix plan')
    tapes = load_tapes(prefix_root, prefixes) if tapes is None else tapes
    chosen, excluded = branch_units(prefixes, tapes, mode)
    used = {u['prefix'] for u in chosen}
    order = [p for p in sorted(C.read(prefix_root/'PLAN.json')['prefixes'], key=lambda p: (p['task'], p['init'])) if p['id'] in used]
    return dict(mode=mode, experiment='mechanism-census', shards=SHARDS[mode], prefix_root=str(prefix_root), prefix_hashes=hashes,
                prefix_order=order, branches=chosen, excluded=excluded, layers=LAYERS, arms={k: list(v) for k, v in ARMS.items()},
                phase_arms={str(k): list(v) for k, v in PHASE_ARMS.items()}, inits=[SMOKE_INIT] if mode == 'smoke' else list(INITS),
                stage1_root=str(stage1_root), stage1_hashes_sha256=C.sha(Path(stage1_root)/'HASHES.json'))


def write_plans(prefix_root, smoke_root, formal_root, stage1_root):
    prefixes = C.read(Path(prefix_root)/'PREFIXES.json'); tapes = load_tapes(prefix_root, prefixes)
    for root, mode in ((smoke_root, 'smoke'), (formal_root, 'formal')):
        root = Path(root)
        if (root/'PLAN.json').exists() or not (root/'CAMPAIGN.json').exists():
            raise RuntimeError('Plan target must be a prepared root without a plan: %s' % root)
        body = plan(prefix_root, mode, stage1_root, tapes)
        C.write(root/'PLAN.json', body)
        print(mode, len(body['branches']), 'units', len(body['prefix_order']), 'prefixes', len(body['excluded']), 'excluded')


# ---------------------------------------------------------------- statistics (pure)
def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return None
    rank = lambda v: np.argsort(np.argsort(v, kind='stable'), kind='stable').astype(float)
    def avg_rank(v):  # ties get their mean rank
        r = rank(v); out = r.copy()
        for value in np.unique(v):
            m = v == value; out[m] = r[m].mean()
        return out
    a, b = avg_rank(x), avg_rank(y)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def resample(groups, rng):
    """groups: pair -> {cell: [rows]}. Pairs drawn with replacement, then rows within each cell of a drawn pair."""
    keys = sorted(groups); out = []
    for k in rng.integers(0, len(keys), len(keys)):
        cells = groups[keys[k]]
        out.append({c: [rows[j] for j in rng.integers(0, len(rows), len(rows))] for c, rows in cells.items()})
    return out


def flatten(sample):
    """Rows of a resample; a pair drawn twice keeps two distinct labels, so its cells count twice."""
    return [dict(s, pair='%s#%d' % (s['pair'], j)) for j, g in enumerate(sample) for v in g.values() for s in v]


def interval(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return None, None
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def by_pair(rows, key):
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        out[r['pair']][key(r)].append(r)
    return {p: dict(v) for p, v in out.items()}


def paired_diff(states, x, y, cells, draws=None, seed=SEED):
    """Mean over states of b_done[x] - b_done[y] on the given (pair, phase) cells, with a 95% interval from a two-level
    bootstrap: pairs with replacement, then states within each cell."""
    draws = DRAWS if draws is None else draws
    rows = [dict(pair=s['pair'], cell=(s['pair'], s['phase']), d=int(s[x]) - int(s[y]), x=int(s[x]), y=int(s[y]))
            for s in states if (s['pair'], s['phase']) in cells and s.get(x) is not None and s.get(y) is not None]
    if not rows:
        return None
    groups = by_pair(rows, lambda r: r['cell']); rng = np.random.default_rng(seed)
    boot = [np.mean([r['d'] for g in resample(groups, rng) for v in g.values() for r in v]) for _ in range(draws)]
    lo, hi = interval(boot)
    return dict(n=len(rows), cells=len({r['cell'] for r in rows}), pairs=len(groups), mean=float(np.mean([r['d'] for r in rows])), lo=lo, hi=hi,
                x_rate=float(np.mean([r['x'] for r in rows])), y_rate=float(np.mean([r['y'] for r in rows])),
                x_only=sum(r['d'] > 0 for r in rows), y_only=sum(r['d'] < 0 for r in rows))


def logistic(X, y, ridge=RIDGE, iterations=100):
    """L2-penalized logistic regression (intercept in column 0, not penalized) by Newton's method."""
    X = np.asarray(X, float); y = np.asarray(y, float); w = np.zeros(X.shape[1])
    penalty = np.eye(X.shape[1]) * ridge; penalty[0, 0] = 0
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        step = np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]) + penalty, X.T @ (p - y) + penalty @ w)
        w -= step
        if np.abs(step).max() < 1e-9:
            break
    return w


def auc(score, y):
    score = np.asarray(score, float); y = np.asarray(y, bool)
    pos, neg = score[y], score[~y]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None]).sum() + .5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


# ---------------------------------------------------------------- readings (pure, on per-state rows)
def states(runs):
    """One row per SW state: native / amp2 b_done, native's A completion, covariates, request-1 image gap; gamma = the gap
    over the same (pair, init)'s f = 0 gap."""
    out = {}
    for r in runs:
        s = out.setdefault(r['sw'], dict(sw=r['sw'], pair=r['pair'], task=r['task'], other=r['other'], init=r['init'], phase=r['phase'],
                                         covariates=r['covariates']))
        s[r['arm']] = r['b_done']
        if r['arm'] == 'native':
            s['a_done'] = r['a_done']; s['gap'] = r['gap_first']
    start = {(s['pair'], s['init']): s.get('gap') for s in out.values() if s['phase'] == 0}
    for s in out.values():
        g0 = start.get((s['pair'], s['init']))
        s['gamma'] = s['gap'] / g0 if s['phase'] and s.get('gap') is not None and g0 else None
    return list(out.values())


def cell_table(rows):
    """(pair, phase) -> dict(n, native rate, amp2 rate, gamma median, covariate shares)."""
    cells = defaultdict(list)
    for s in rows:
        if 'native' in s:
            cells[(s['pair'], s['phase'])].append(s)
    out = {}
    for key, ss in cells.items():
        amp = [s['amp2'] for s in ss if 'amp2' in s]; gam = [s['gamma'] for s in ss if s['gamma'] is not None]
        out[key] = dict(n=len(ss), native=float(np.mean([s['native'] for s in ss])), amp2=float(np.mean(amp)) if amp else None,
                        native_done=sum(s['native'] for s in ss), amp2_done=sum(amp) if amp else None,
                        a_done_when_failed=sum(s['a_done'] for s in ss if not s['native']),
                        gamma=float(np.median(gam)) if gam else None,
                        gripper_closed=float(np.mean([s['covariates']['gripper_closed'] for s in ss])),
                        same_object=ss[0]['covariates']['same_object'])
    return out


def classes(cells):
    """Eligible pairs (f = 0 native rate >= CEILING_AT on >= MIN_CELL states) and the class of each readable later cell."""
    eligible = {p for (p, f), c in cells.items() if f == 0 and c['n'] >= MIN_CELL and c['native'] >= CEILING_AT}
    out = {}
    for (p, f), c in cells.items():
        if f == 0:
            continue
        if p not in eligible:
            out[(p, f)] = 'ineligible'
        elif c['n'] < MIN_CELL:
            out[(p, f)] = 'thin'
        else:
            out[(p, f)] = 'failure' if c['native'] <= FAILURE_AT else 'success' if c['native'] >= SUCCESS_AT else 'middle'
    return eligible, out


def prevalence(rows, draws=None, seed=SEED):
    """S1: share of readable cells that are failure cells, per phase, with a pair-level bootstrap interval."""
    draws = DRAWS if draws is None else draws
    def shares(rs):
        cells = cell_table(rs); _, cls = classes(cells)
        out = {}
        for f in PHASES[1:]:
            read = [c for (p, g), c in cls.items() if g == f and c in ('failure', 'success', 'middle')]
            out[f] = dict(cells=len(read), failure=sum(c == 'failure' for c in read), success=sum(c == 'success' for c in read),
                          middle=sum(c == 'middle' for c in read), share=(sum(c == 'failure' for c in read) / len(read)) if read else None)
        return out
    point = shares(rows); groups = by_pair(rows, lambda s: s['phase']); rng = np.random.default_rng(seed)
    boot = [shares(flatten(resample(groups, rng))) for _ in range(draws)]
    for f in PHASES[1:]:
        point[f]['lo'], point[f]['hi'] = interval([b[f]['share'] for b in boot])
    share = point[.75]['share']
    reading = 'insufficient' if share is None else 'common' if share >= COMMON else 'rare' if share <= RARE else 'occasional'
    return dict(reading=reading, per_phase={str(f): v for f, v in point.items()})


def ratio(rows):
    """rho per phase: native B success at f over native B success at f = 0, on (pair, init) rows having both (eligible pairs)."""
    cells = cell_table(rows); eligible, _ = classes(cells)
    start = {(s['pair'], s['init']): s['native'] for s in rows if s['phase'] == 0 and 'native' in s}
    out = {}
    for f in PHASES[1:]:
        use = [(s['native'], start[(s['pair'], s['init'])]) for s in rows if s['phase'] == f and s['pair'] in eligible and 'native' in s
               and (s['pair'], s['init']) in start]
        c = np.mean([b for _, b in use]) if use else 0
        out[str(f)] = dict(n=len(use), rate=float(np.mean([a for a, _ in use])) if use else None, ceiling=float(c) if use else None,
                           rho=float(np.mean([a for a, _ in use]) / c) if use and c else None)
    return out


def restoration(rows, cells):
    """S2: amp2 - native on the given failure cells."""
    gain = paired_diff(rows, 'amp2', 'native', cells)
    if gain is None:
        reading = 'insufficient'
    elif gain['lo'] > 0:
        reading = 'restores_switch' if gain['mean'] >= GAIN else 'weakly_restores'
    elif gain['hi'] < 0:
        reading = 'harms'
    else:
        reading = 'no_restore'
    return dict(reading=reading, gain=gain)


def selectivity(rows, cells):
    """S3: amp2 - native on the given success cells."""
    d = paired_diff(rows, 'amp2', 'native', cells)
    if d is None:
        reading = 'insufficient'
    elif d['lo'] >= -HARM:
        reading = 'selective'
    elif d['hi'] < 0 and d['mean'] <= -HARM:
        reading = 'damages'
    else:
        reading = 'unclear'
    return dict(reading=reading, change=d)


PHYSICAL = ('phase75', 'same_object', 'gripper_closed', 'must_release')


def design(rows, with_gamma, center=None):
    """Feature matrix: intercept, the physical covariates (0/1), and optionally standardized log gamma."""
    cols = [np.ones(len(rows))]
    cols.append(np.asarray([s['phase'] == .75 for s in rows], float))
    cols.append(np.asarray([s['covariates']['same_object'] for s in rows], float))
    cols.append(np.asarray([s['covariates']['gripper_closed'] for s in rows], float))
    cols.append(np.asarray([s['covariates']['gripper_closed'] and not s['covariates']['same_object'] for s in rows], float))
    if with_gamma:
        lg = np.log(np.asarray([s['gamma'] for s in rows], float))
        m, sd = center if center is not None else (lg.mean(), lg.std() or 1.)
        cols.append((lg - m) / sd)
    return np.stack(cols, 1)


def explanation(rows, cells, draws=None, seed=SEED):
    """S4: (a) across readable cells, Spearman of the median gamma against the native switch rate; (b) state level, logistic
    regression of the native switch on the physical covariates with and without log gamma, pair-level bootstrap intervals."""
    draws = DRAWS if draws is None else draws
    use = [s for s in rows if s['phase'] and (s['pair'], s['phase']) in cells and 'native' in s and s['gamma'] is not None and s['gamma'] > 0]
    if len({s['pair'] for s in use}) < 5:
        return dict(across_cells=dict(reading='insufficient'), beyond_physics=dict(reading='insufficient'))
    def rho(rs):
        t = cell_table(rs); keys = [k for k in t if t[k]['gamma'] is not None]
        return spearman([t[k]['gamma'] for k in keys], [t[k]['native'] for k in keys])
    groups = by_pair(use, lambda s: s['phase']); rng = np.random.default_rng(seed)
    point = rho(use); boot = [rho(flatten(resample(groups, rng))) for _ in range(draws)]
    lo, hi = interval(boot)
    a = dict(rho=point, lo=lo, hi=hi, cells=len({(s['pair'], s['phase']) for s in use}),
             reading='insufficient' if point is None or lo is None else 'fade_tracks' if lo > 0 else 'reversed' if hi < 0 else 'unclear')
    lg = np.log(np.asarray([s['gamma'] for s in use], float)); center = (lg.mean(), lg.std() or 1.)
    y = np.asarray([s['native'] for s in use], float)
    names = ('intercept',) + PHYSICAL + ('log_gamma',)
    full = logistic(design(use, True, center), y); physical = logistic(design(use, False), y)
    rng = np.random.default_rng(seed + 1); coefs = []
    pairs = sorted({s['pair'] for s in use}); per = defaultdict(list)
    for s in use:
        per[s['pair']].append(s)
    for _ in range(draws):
        sample = [s for k in rng.integers(0, len(pairs), len(pairs)) for s in per[pairs[k]]]
        ys = np.asarray([s['native'] for s in sample], float)
        coefs.append(logistic(design(sample, True, center), ys) if 0 < ys.sum() < len(ys) else np.full(len(names), np.nan))
    coefs = np.asarray(coefs)
    table = {n: dict(coef=float(full[j]), **dict(zip(('lo', 'hi'), interval(coefs[:, j])))) for j, n in enumerate(names)}
    g = table['log_gamma']
    if g['lo'] is None:
        reading = 'insufficient'
    elif g['lo'] > 0:
        reading = 'beyond_physics'
    elif any(table[n]['lo'] is not None and (table[n]['lo'] > 0 or table[n]['hi'] < 0) for n in PHYSICAL):
        reading = 'physics_explains'
    else:
        reading = 'unclear'
    b = dict(reading=reading, states=len(use), pairs=len(pairs), coefficients=table,
             physical_only={n: float(physical[j]) for j, n in enumerate(names[:-1])},
             auc_physical=auc(design(use, False) @ physical, y), auc_full=auc(design(use, True, center) @ full, y))
    return dict(across_cells=a, beyond_physics=b)


def gain_factors(rows, cells):
    """Reported: per failure cell amp2 - native, against the cell's median gamma and covariates."""
    t = cell_table(rows); keys = [k for k in cells if k in t and t[k]['amp2'] is not None]
    gains = {k: t[k]['amp2'] - t[k]['native'] for k in keys}
    with_g = [k for k in keys if t[k]['gamma'] is not None]
    group = lambda f: {str(v): dict(cells=len(ks), mean_gain=float(np.mean([gains[k] for k in ks])))
                       for v in (False, True) for ks in [[k for k in keys if f(k) == v]] if ks}
    return dict(cells=len(keys), spearman_gamma=spearman([t[k]['gamma'] for k in with_g], [gains[k] for k in with_g]),
                spearman_native=spearman([t[k]['native'] for k in keys], [gains[k] for k in keys]),
                by_same_object=group(lambda k: t[k]['same_object']), by_gripper_closed=group(lambda k: t[k]['gripper_closed'] >= .5),
                by_phase={str(f): float(np.mean([gains[k] for k in keys if k[1] == f])) for f in PHASES[1:] if any(k[1] == f for k in keys)},
                share_gain_30=float(np.mean([gains[k] >= .3 for k in keys])) if keys else None)


def readings(runs):
    rows = states(runs); cells = cell_table(rows); eligible, cls = classes(cells)
    pick = lambda kinds, pilot: tuple(k for k, c in cls.items() if c in kinds and ((k[0] in PILOT) == pilot if pilot is not None else True))
    failure, success = pick(('failure',), False), pick(('success',), False)
    readable = pick(('failure', 'success', 'middle'), None)
    return dict(
        prevalence=prevalence(rows),                                                    # S1
        restoration=restoration(rows, failure),                                         # S2
        selectivity=selectivity(rows, success),                                         # S3
        explanation=explanation(rows, readable),                                        # S4
        reported=dict(
            rho=ratio(rows), eligible_pairs=len(eligible), pairs=len({s['pair'] for s in rows}),
            classes={c: sum(v == c for v in cls.values()) for c in ('failure', 'middle', 'success', 'thin', 'ineligible')},
            restoration_with_pilot=restoration(rows, pick(('failure',), None)),
            restoration_pilot=restoration(rows, pick(('failure',), True)),
            restoration_by_phase={str(f): restoration(rows, tuple(k for k in failure if k[1] == f)) for f in PHASES[1:]},
            middle=paired_diff(rows, 'amp2', 'native', pick(('middle',), None)),
            selectivity_with_pilot=selectivity(rows, pick(('success',), None)),
            failure_finishes_a=dict(failed=sum(1 for s in rows if (s['pair'], s['phase']) in pick(('failure',), None) and not s['native']),
                                    finished_a=sum(1 for s in rows if (s['pair'], s['phase']) in pick(('failure',), None) and not s['native'] and s['a_done'])),
            gain_factors=gain_factors(rows, pick(('failure',), None))),
        cells={'%s f=%s' % k: dict(v, cls=cls.get(k, 'start')) for k, v in sorted(cells.items())})


def tool_reading(runs, state_check):
    """C1: (reading, detail). Completion, replay against the prefix, outcome recomputation and logs are checked while loading."""
    fidelity = [r['fidelity_linf'] for r in runs]
    native = [r for r in runs if r['kind'] == 'native']; amp = [r for r in runs if r['kind'] == 'amp']
    detail = dict(runs=len(runs), physics_equal=sum(r['physics_equal'] for r in runs),
                  fidelity_median=float(np.median(fidelity)), fidelity_max=float(np.max(fidelity)),
                  native_moved=sum(r['max_delta'] > 0 or r['first_shift_linf'] > 0 for r in native),
                  scale_error_max=max([r['scale_error'] for r in amp], default=0.), steered_unmoved=sum(r['first_shift_linf'] == 0 for r in amp),
                  prompt_errors=sum(not r['prompt_ok'] for r in runs), b_at_checkpoint=sum(bool(r['b_at_checkpoint']) for r in runs),
                  state_check=dict(passed=state_check.get('passed'), runs=state_check.get('runs'), cross_checks=state_check.get('cross_checks')))
    ok = (detail['physics_equal'] == len(runs) and detail['fidelity_median'] <= C.FIDELITY_MEDIAN and detail['fidelity_max'] <= C.FIDELITY_MAX
          and detail['native_moved'] == 0 and detail['scale_error_max'] <= C.SCALE_TOLERANCE and detail['steered_unmoved'] == 0
          and detail['prompt_errors'] == 0 and detail['b_at_checkpoint'] == 0 and state_check.get('passed') is True
          and (state_check.get('cross_checks') or 0) > 0)
    return ('tool_ok' if ok else 'tool_failed'), detail


# ---------------------------------------------------------------- summary (CPU, reads the run root)
def run_row(root, u, tape):
    from verifier_dataset import hil_suite_run as H
    row, trace = H.load(Path(root)/'branch'/u['id'])
    if not C.same_outcome(S.outcome(trace, u, row['start_eef']), row['outcome']):
        raise RuntimeError('Recomputed outcome differs: ' + u['id'])
    if row['rng_reset'].get('arm') != u['arm']:
        raise RuntimeError('Arm not acknowledged: ' + u['id'])
    with open(Path(root)/'branch'/u['id']/'POLICY.jsonl') as stream:
        client = [json.loads(x) for x in stream]
    log = C.server_log(root, row['rng_reset']['run_id'], client)
    scale = u['arm']['lam'] - 1
    ratio_ = lambda e, n: e['delta_' + n] / (scale * e['diff_' + n]) if e['diff_' + n] > 0 else (1. if e['delta_' + n] == 0 else np.inf)
    o = row['outcome']
    return dict(id=u['id'], sw=u['sw'], pair=u['pair'], task=u['task'], other=u['other'], init=u['init'], phase=u['phase'], arm=u['arm']['name'],
                kind=u['arm']['kind'], b_done=o['b_done'], a_done=o['a_done'], b_after=o['b_after'], steps=o['steps'],
                b_at_checkpoint=o['b_at_checkpoint'], covariates=u['covariates'],
                physics_equal=row['physics_at_checkpoint'] == tape[u['checkpoint']]['physics'], first_input=row['first_policy_input_sha256'],
                prompt_ok=row['prompt'] == u['prompt'], fidelity_linf=log[0]['fidelity_linf'], first_shift_linf=log[0]['first_shift_linf'],
                max_delta=max(max(e['delta_k'], e['delta_v']) for e in log),
                scale_error=max(abs(ratio_(e, n) - 1) for e in log for n in 'kv') if scale else 0.,
                gap_first=log[0]['diff_v'] / log[0]['norm_v'], requests=len(log),
                seconds_per_request=float(np.median([e['seconds'] for e in log[1:]])) if len(log) > 1 else None,
                actions_outside_spec=row['actions_outside_spec'], elapsed_seconds=row['elapsed_seconds'])


def stage1_replication(plan_, runs):
    """Reported: native rates of the census against HIL suite stage 1 on the shared Goal pairs and inits (cell level)."""
    root = Path(plan_['stage1_root'])
    if C.sha(root/'HASHES.json') != plan_['stage1_hashes_sha256']:
        raise RuntimeError('Stage 1 run changed')
    s1 = C.read(root/'PLAN.json')
    with gzip.open(root/'RUNS.jsonl.gz', 'rt') as stream:
        done = {r['id']: bool(r['b_done']) for r in map(json.loads, stream)}
    inits = set(plan_['inits']); old = defaultdict(list); new = defaultdict(list)
    for u in s1['branches']:
        if u['suite'] == SUITE and u['condition'] == 'SW' and u['init'] in inits and u['phase'] in PHASES and u['id'] in done:
            old[(pair_label(u['task'], u['other']), u['phase'])].append(done[u['id']])
    for r in runs:
        if r['arm'] == 'native':
            new[(r['pair'], r['phase'])].append(r['b_done'])
    keys = sorted(set(old) & set(new))
    return {'%s f=%s' % k: dict(stage1='%d/%d' % (sum(old[k]), len(old[k])), census='%d/%d' % (sum(new[k]), len(new[k]))) for k in keys}


def summarize(root, require_tool_ok=False):
    from verifier_dataset import hil_suite_run as H
    root = Path(root); plan_ = C.read(root/'PLAN.json')
    for k in range(plan_['shards']):
        if not C.read(root/('SHARD_branch_%d_COMPLETE.json' % k))['passed']:
            raise RuntimeError('Shard %d incomplete' % k)
    prefix_root = Path(plan_['prefix_root'])
    for n, digest in plan_['prefix_hashes'].items():
        if C.sha(prefix_root/n) != digest:
            raise RuntimeError('Prefix run output changed: ' + n)
    tapes = {p['id']: H.load(prefix_root/'prefix'/p['id'])[1] for p in plan_['prefix_order']}
    runs = [run_row(root, u, tapes[u['prefix']]) for u in plan_['branches']]
    check = C.read(root/'STATE_CHECK.json') if (root/'STATE_CHECK.json').exists() else {}
    tool, detail = tool_reading(runs, check)
    twins = defaultdict(set)
    for r in runs:
        twins[r['sw']].add(r['first_input'])
    logs = sorted((root/'server').glob('*.jsonl'))
    per_arm = defaultdict(list)
    for r in runs:
        per_arm[r['arm']].append(r)
    summary = dict(
        mode=plan_['mode'], tool=tool, tool_detail=detail, runs=len(runs), layers=plan_['layers'], inits=plan_['inits'],
        readings=readings(runs) if tool == 'tool_ok' else None, descriptive_only=plan_['mode'] == 'smoke',
        excluded=dict(units=len(plan_['excluded']), by_reason={k: sum(e['reason'] == k for e in plan_['excluded']) for k in {e['reason'] for e in plan_['excluded']}}),
        twin_first_input_equal=sum(len(v) == 1 for v in twins.values()), twin_states=len(twins),
        stage1_replication=stage1_replication(plan_, runs),
        first_shift_median={a: float(np.median([r['first_shift_linf'] for r in rs])) for a, rs in per_arm.items()},
        seconds_per_run=float(np.median([r['elapsed_seconds'] for r in runs])),
        actions_outside_spec={a: sum(r['actions_outside_spec'] for r in rs) for a, rs in per_arm.items()},
        total_run_hours=sum(r['elapsed_seconds'] for r in runs) / 3600.,
        server_logs_sha256=hashlib.sha256(''.join(p.name + C.sha(p) for p in logs).encode()).hexdigest(), server_logs=len(logs),
        api_calls=0, code_commit=os.environ.get('CB_CODE_COMMIT'), slurm_job=os.environ.get('SLURM_JOB_ID'))
    with gzip.open(root/'RUNS.jsonl.gz', 'wt') as stream:
        stream.writelines(json.dumps(r) + '\n' for r in runs)
    C.write(root/'SUMMARY.json', summary)
    C.write(root/'HASHES.json', {n: C.sha(root/n) for n in ('PLAN.json', 'RUNS.jsonl.gz', 'SUMMARY.json')})
    print(json.dumps({k: summary[k] for k in ('mode', 'tool', 'tool_detail', 'excluded', 'seconds_per_run')}, indent=1))
    if summary['readings']:
        r = summary['readings']
        print(json.dumps(dict(S1=r['prevalence']['reading'], S2=r['restoration']['reading'], S3=r['selectivity']['reading'],
                              S4a=r['explanation']['across_cells']['reading'], S4b=r['explanation']['beyond_physics']['reading'])))
    if require_tool_ok and (tool != 'tool_ok' or plan_['mode'] != 'smoke'):
        raise SystemExit('Gate: the smoke tool check is %s; the formal batch does not start' % tool)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prefixes', 'plan', 'summarize']); parser.add_argument('paths', nargs='+')
    parser.add_argument('--require-tool-ok', action='store_true'); args = parser.parse_args()
    if args.action == 'prefixes':
        body = prefix_plan(C.read(args.paths[0])); C.write(args.paths[1], body); print(len(body), 'prefixes')
    elif args.action == 'plan':
        write_plans(*map(lambda p: Path(os.path.abspath(p)), args.paths))
    else:
        summarize(args.paths[0], args.require_tool_ok)
