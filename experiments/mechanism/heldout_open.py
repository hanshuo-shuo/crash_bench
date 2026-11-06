"""Held-out open-loop localization of the mechanism pilot (docs/generative_verifier/HIL_MECHANISM_HELDOUT_OPEN.md).

Inference only on the stored HIL suite stage 1 switch inputs; no simulator, no training, no API.
  full states  (the eight held-out pairs of the repair batch): the pilot's open-loop measures (run_open_loop.run_state) plus
               the open-loop push of the repair;
  steer states (the five pilot failure cells, calibration): the open-loop push only.
The open-loop push is request 1 of the closed-loop arms on the stored input, per noise seed: a (prompt A, nothing moved),
b (prompt B), amp2 (prompt B, the image keys/values of layers 12-14 moved one more B - A) and rand2 (the same norm along
the random direction of the closed-loop rand2 arm's request 1). push = E(amp2) - 1 on the a -> b path.

  heldout_open.py plan BRANCH_ROOT OUT_JSON smoke|formal        the states (read-only on stage 1)
  heldout_open.py run ROOT --shard K --shards N --checkpoint DIR
  heldout_open.py summarize ROOT [--require-tool-ok]            C1 and the readings H1-H4 -> ROOT/SUMMARY.json
"""
import argparse
from collections import defaultdict
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent/'feasibility'))
import closed_loop as C  # noqa: E402
import metrics as M  # noqa: E402
import repair as R  # noqa: E402
import run_open_loop as O  # noqa: E402
import selection as S  # noqa: E402

GOAL, TEN, OBJECT = R.GOAL, R.TEN, R.OBJECT
HELD_PAIRS = ((GOAL, 1, 4), (GOAL, 4, 8), (GOAL, 8, 1), (GOAL, 8, 7), (GOAL, 9, 2), (TEN, 0, 1), (TEN, 1, 0), (OBJECT, 1, 0))
HELD_LABELS = tuple(R.label(s, '%d->%d' % (a, b)) for s, a, b in HELD_PAIRS)
CALIBRATION = R.IN_FAILURE            # the pilot failure cells: open-loop push only
LAYERS = C.LAYERS                     # the repair's window, 12-14
WINDOW = 'L12-14_image'
SMOKE_INITS = (0, 1)
READING_INITS = M.READING_INITS       # 2-19, the repair batch's formal inits
SEEDS = 4
FAILURE_CLASSES = ('in_failure', 'commits', 'derails')
ROUTE_AT = M.PARTIAL                  # H1: .4
MIN_READABLE = 5                      # H1: readable states of a pair's f = 0 cell
MIN_PAIRS = 4                         # H1: readable pairs
MIN_CELLS = 8                         # H3: failure cells with a push
TRACKS = .5                           # H3
HALF = .5                             # H4
# H4: effective / weak by the repair batch's pooled gain over each pair's failure cells (>= closed_loop.GAIN); checked.
EFFECTIVE = (R.label(GOAL, '2->9'), R.label(GOAL, '7->8'), R.label(GOAL, '8->7'), R.label(TEN, '0->1'))
WEAK = (R.label(TEN, '1->0'), R.label(OBJECT, '1->0'))
DRAWS, SEED = 2000, 0
MECH = Path('/projects/p33100/siosio/crashbench_safelibero/mechanism')
REPAIR_ROOT = MECH/'20261105T052651Z_mech-repair-formal_pi05_libero_6199b280f028'
PILOT_ROOT = MECH/'20261104T202621Z_mech-formal_pi05_libero_4d8c25a1dc96'
read, write, sha = O.read, O.write, O.sha


# ---------------------------------------------------------------- plan (read-only on stage 1)
def pair_label(u):
    return R.label(u['suite'], '%d->%d' % (u['task'], u['other']))


def kind_of(u):
    if (u['suite'], u['task'], u['other']) in HELD_PAIRS:
        return 'full'
    if (pair_label(u), u['phase']) in CALIBRATION:
        return 'steer'
    return None


def select(branch_plan, runs, inits):
    """State rows for every SW unit of the held-out pairs and the calibration cells at `inits`, each with its A twin (the C0
    branch of the same prefix, switch step and group; the prefix itself at f = 0). Nothing is selected on outcome."""
    rows = {r['id']: r for r in runs}; twins = {}
    for u in branch_plan['branches']:
        if u['condition'] == 'C0':
            key = (u['prefix'], u['checkpoint'], u['group'])
            if key in twins:
                raise ValueError('Two C0 twins for %r' % (key,))
            twins[key] = u
    states = []
    for u in branch_plan['branches']:
        if u['condition'] != 'SW' or u['init'] not in inits or u['id'] not in rows:
            continue
        kind = kind_of(u)
        if kind is None:
            continue
        label = pair_label(u); cell = R.cell_class((label, u['phase']))
        if cell is None:
            raise ValueError('No repair-batch class for %s' % u['id'])
        twin = None if u['phase'] == 0 else twins[(u['prefix'], u['checkpoint'], u['group'])]
        if twin is not None and (twin['seed'] != u['seed'] or twin['task'] != u['task']):
            raise ValueError('Twin mismatch for %s' % u['id'])
        states.append(dict(
            id='%s_f%03d' % (u['id'], round(100 * u['phase'])), kind=kind, cell=cell, suite=u['suite'], pair=label,
            task_a=u['task'], task_b=u['other'], init=u['init'], phase=u['phase'], checkpoint=u['checkpoint'], seed=u['seed'],
            prefix=u['prefix'], sw=u['id'], twin=None if twin is None else twin['id'], prompt_b=u['prompt'],
            tokens_b=branch_plan['prompt_tokens'][u['prompt']], b_done=bool(rows[u['id']]['b_done']),
            twin_a_done=None if twin is None else bool(rows[twin['id']]['a_done']), direction_seed=C.direction_seed(u['id'])))
    return states


def counts(states):
    out = defaultdict(lambda: dict(states=0, b_done=0))
    for s in states:
        c = out['%s %s f=%s' % (s['kind'], s['pair'], s['phase'])]; c['states'] += 1; c['b_done'] += s['b_done']
    return dict(sorted(out.items()))


def plan(branch_root, out_json, mode):
    branch_root = Path(branch_root); branch_plan = read(branch_root/'PLAN.json')
    with gzip.open(branch_root/'RUNS.jsonl.gz', 'rt') as stream:
        runs = [json.loads(x) for x in stream]
    states = select(branch_plan, runs, SMOKE_INITS if mode == 'smoke' else READING_INITS)
    S.attach_files(states, branch_root, branch_plan['prefix_root'])
    for s in states:
        s['tokens_a'] = branch_plan['prompt_tokens'].get(s['prompt_a'])
    out = dict(experiment='mechanism-heldout-open', mode=mode, branch_root=str(branch_root), prefix_root=branch_plan['prefix_root'],
               held_pairs=list(HELD_LABELS), calibration=[list(c) for c in CALIBRATION], layers=LAYERS, seeds=SEEDS,
               inits=list(SMOKE_INITS if mode == 'smoke' else READING_INITS), states=states, counts=counts(states))
    write(out_json, out)
    return out


# ---------------------------------------------------------------- GPU
def push_requests(patcher, data_a, data_b, seeds, direction_seed, layers=LAYERS):
    """Request 1 of the closed-loop arms on one stored input, per seed: chunks a, b, amp2, rand2 [n, H, 7] and the
    steer stats of amp2 and rand2 [n, 6]. The sample key is the one openpi's Policy.infer draws after a reset to the seed
    (serve_steer: split(key(seed))[1]); the random direction is rand2's request 1 (fold_in(key(direction_seed), 1))."""
    import jax
    import patching as P
    mask = np.isin(np.arange(patcher.depth), layers)
    direction = jax.random.fold_in(jax.random.key(direction_seed), 1)
    chunks = dict(a=[], b=[], amp2=[], rand2=[]); stats = dict(amp2=[], rand2=[])
    for s in seeds:
        sample = jax.random.split(jax.random.key(int(s)))[1]
        chunks['a'].append(patcher.steer(dict(data_a), dict(data_b), sample, mask, 0., P.AMPLIFY, direction)[0])
        chunks['b'].append(patcher.steer(dict(data_b), dict(data_a), sample, mask, 0., P.AMPLIFY, direction)[0])
        for name, mode in (('amp2', P.AMPLIFY), ('rand2', P.RANDOM)):
            x, st = patcher.steer(dict(data_b), dict(data_a), sample, mask, 1., mode, direction)
            chunks[name].append(x); stats[name].append(np.asarray(st))
    return {k: np.stack(v) for k, v in chunks.items()}, {k: np.stack(v) for k, v in stats.items()}


def push_fields(chunks, stats, native):
    """The push of one state from push_requests' output; `native` holds openpi's Policy.infer chunks a, b on the first seed."""
    a, b = np.asarray(chunks['a'], float), np.asarray(chunks['b'], float)
    floor = M.noise_floor(a); ok = M.defined(a, b, floor); readable = int(ok.sum()) >= M.MIN_SEEDS

    def push(name):
        e = M.effect(chunks[name], a, b) - 1
        return float(e[ok].mean()) if readable else None

    def raw(name):
        shift = np.linalg.norm(M.path(chunks[name]) - M.path(b), axis=-1)
        return float(np.median(shift)) / floor if floor > 0 else None

    def scale_error(st):  # steer stats: delta_k, diff_k, norm_k, delta_v, diff_v, norm_v; at scale 1 delta = diff
        st = np.asarray(st, float)
        return float(max(np.abs(st[:, 0] / np.maximum(st[:, 1], 1e-30) - 1).max(), np.abs(st[:, 3] / np.maximum(st[:, 4], 1e-30) - 1).max()))
    ab = np.linalg.norm(M.path(b) - M.path(a), axis=-1)
    return dict(push=dict(
        floor=floor, defined=ok.tolist(), readable=readable, amp2=push('amp2'), rand2=push('rand2'), raw_amp2=raw('amp2'),
        raw_rand2=raw('rand2'), ab_floors=float(np.median(ab)) / floor if floor > 0 else None,
        a_vs_native_linf=float(np.abs(a[0] - np.asarray(native['a'])).max()), b_vs_native_linf=float(np.abs(b[0] - np.asarray(native['b'])).max()),
        scale_error=dict(amp2=scale_error(stats['amp2']), rand2=scale_error(stats['rand2'])),
        amp2_changed=bool(np.abs(np.asarray(chunks['amp2']) - b).max() > 0),
        diff_norm_k=np.asarray(stats['amp2'])[:, 1].tolist(), diff_norm_v=np.asarray(stats['amp2'])[:, 4].tolist()))


def push_arrays(chunks, stats):
    return dict(**{'push_' + k: v for k, v in chunks.items()}, **{'push_stats_' + k: v for k, v in stats.items()})


def run_steer_state(patcher, policy, state, seeds_per_state, folder, layers=LAYERS):
    """A calibration state: the open-loop push only, with the input and Policy.infer checks."""
    from serve import digest
    started = time.monotonic()
    data_a, data_b, _, pixel = O.inputs(state)
    seeds = [(state['seed'] + j) % O.SEED_SPAN for j in range(seeds_per_state)]
    native = dict(a=O.native(policy, data_a, seeds[0]), b=O.native(policy, data_b, seeds[0]))
    chunks, stats = push_requests(patcher, data_a, data_b, seeds, state['direction_seed'], layers)
    row = dict(id=state['id'], pair=state['pair'], phase=state['phase'], init=state['init'], b_done=state['b_done'], seeds=seeds,
               input_b_exact=digest(data_b) == state['b_first']['input_sha256'], a_twin_pixel_max_abs=pixel,
               native_b_vs_logged_linf=float(np.abs(native['b'] - np.asarray(state['b_first']['actions'])).max()),
               **push_fields(chunks, stats, native), **O.provenance())
    row['elapsed_seconds'] = time.monotonic() - started
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/'ACTIONS.npz', native_a=native['a'], native_b=native['b'], seeds=np.asarray(seeds, np.uint64),
                        **push_arrays(chunks, stats))
    write(folder/'ROW.json', row)
    write(folder/'HASHES.json', {p.name: sha(p) for p in sorted(folder.iterdir()) if p.name != 'HASHES.json'})
    return row


def run_full_state(patcher, policy, state, seeds_per_state, folder, layers=LAYERS):
    """A held-out state: run_open_loop.run_state (the pilot's measures, unchanged) with the open-loop push added."""
    def extra(data_a, data_b, seeds, native):
        chunks, stats = push_requests(patcher, data_a, data_b, seeds, state['direction_seed'], layers)
        return push_fields(chunks, stats, native), push_arrays(chunks, stats)
    return O.run_state(patcher, policy, state, seeds_per_state, folder, extra=extra)


def run(root, shard, shards, checkpoint):
    import jax
    from openpi.policies import policy_config
    from openpi.training import config
    import patching as P
    root = Path(root); plan_doc = read(root/'PLAN.json'); states = read(root/'STATES.json')['states']
    if sha(root/'STATES.json') != plan_doc['states_sha256']:
        raise RuntimeError('STATES.json changed')
    chosen = set(plan_doc['state_ids'])
    mine = [s for i, s in enumerate(s for s in states if s['id'] in chosen) if i % shards == shard]
    policy = policy_config.create_trained_policy(config.get_config('pi05_libero'), checkpoint)
    if policy._is_pytorch_model:
        raise RuntimeError('JAX policy required')
    patcher = P.Patcher(policy); progress = root/('PROGRESS_%d.json' % shard); done = []
    for state in mine:
        folder = root/'states'/state['id']
        if (folder/'HASHES.json').exists():
            raise RuntimeError('State already written: %s' % state['id'])
        step = run_full_state if state['kind'] == 'full' else run_steer_state
        row = step(patcher, policy, state, plan_doc['seeds'], folder); done.append(state['id'])
        write(progress, dict(done=len(done), total=len(mine), last=state['id'], last_seconds=row['elapsed_seconds']))
    write(root/('SHARD_%d_COMPLETE.json' % shard), dict(passed=True, states=done, jax_devices=[str(d) for d in jax.devices()],
                                                       **O.provenance()))


# ---------------------------------------------------------------- loading
def checked_rows(root, ids):
    rows = []
    for sid in ids:
        folder = Path(root)/'states'/sid
        for name, digest in read(folder/'HASHES.json').items():
            if sha(folder/name) != digest:
                raise RuntimeError('State output changed: %s/%s' % (sid, name))
        rows.append(read(folder/'ROW.json'))
    return rows


def load(root):
    """Rows of a finished root, each joined with its state's kind, class, pair and SW id; raises on any integrity failure."""
    root = Path(root); plan_doc = read(root/'PLAN.json')
    if sha(root/'STATES.json') != plan_doc['states_sha256']:
        raise RuntimeError('STATES.json changed')
    for k in range(plan_doc['shards']):
        if not read(root/('SHARD_%d_COMPLETE.json' % k))['passed']:
            raise RuntimeError('Shard %d did not pass' % k)
    states = {s['id']: s for s in read(root/'STATES.json')['states']}
    if set(states) != set(plan_doc['state_ids']):
        raise RuntimeError('Planned states differ from STATES.json')
    keys = ('kind', 'cell', 'pair', 'sw', 'phase', 'init')
    return plan_doc, [dict(r, **{k: states[r['id']][k] for k in keys}) for r in checked_rows(root, plan_doc['state_ids'])]


def closed_loop_runs(root=REPAIR_ROOT):
    """SW id -> {arm: b_done} for the native and amp2 runs of the repair batch's formal root (hash-checked)."""
    root = Path(root)
    if sha(root/'RUNS.jsonl.gz') != read(root/'HASHES.json')['RUNS.jsonl.gz']:
        raise RuntimeError('Repair batch runs changed')
    out = defaultdict(dict)
    with gzip.open(root/'RUNS.jsonl.gz', 'rt') as stream:
        for line in stream:
            r = json.loads(line)
            if r['arm'] in ('native', 'amp2'):
                out[r['sw']][r['arm']] = bool(r['b_done'])
    return dict(out)


def pilot_rows(root=PILOT_ROOT):
    """The pilot open-loop formal rows (hash-checked), labelled like this batch (suite-prefixed pair, SW id, repair class)."""
    root = Path(root); plan_doc = read(root/'PLAN.json'); states = {s['id']: s for s in read(root/'STATES.json')['states']}
    out = []
    for r in checked_rows(root, plan_doc['state_ids']):
        s = states[r['id']]; pair = R.label(GOAL, s['pair'])
        out.append(dict(r, pair=pair, sw=s['sw'], cell=R.cell_class((pair, s['phase'])), kind='pilot'))
    return out


# ---------------------------------------------------------------- readings (pure)
def tool_reading(rows):
    """C1: the pilot's P1 on the held-out states, the inputs, and the push program's fidelity, scale and effect."""
    full = [r for r in rows if r['kind'] == 'full']
    p1, detail = M.fidelity_reading(full) if full else ('tool_failed', {})
    native = [max(r['push']['a_vs_native_linf'], r['push']['b_vs_native_linf']) for r in rows]
    scale = max(max(r['push']['scale_error'].values()) for r in rows)
    checks = dict(pilot_p1=p1 == 'tool_ok', inputs=all(r['input_b_exact'] for r in rows),
                  push_fidelity=float(np.median(native)) <= C.FIDELITY_MEDIAN and max(native) <= C.FIDELITY_MAX,
                  scale=scale <= C.SCALE_TOLERANCE, amp2_changes=all(r['push']['amp2_changed'] for r in rows))
    return ('tool_ok' if all(checks.values()) else 'tool_failed'), dict(
        checks, p1=detail, push_native_linf_median=float(np.median(native)), push_native_linf_max=float(max(native)),
        scale_error_max=scale, states=dict(full=len(full), steer=len(rows) - len(full)))


def three_layer(names):
    return [n for n in names if n.startswith('L') and n[1:3] != n[4:6]]


def single_layer(names, group='all'):
    return [n for n in names if n.startswith('L') and n[1:3] == n[4:6] and n.endswith('_' + group)]


def t0_route(rows, inits=READING_INITS):
    """H1: per held-out pair, where B's effect enters the action expert at f = 0."""
    out = {}
    for pair in HELD_LABELS:
        group = [r for r in rows if r['kind'] == 'full' and r['pair'] == pair and r['phase'] == 0 and r['init'] in inits]
        readable = sum(r['interface'][WINDOW] is not None for r in group)
        if readable < MIN_READABLE:
            out[pair] = dict(reading='unreadable', readable=readable, states=len(group))
            continue
        names = list(group[0]['interface'])
        cell = {n: M.bootstrap_median([r['interface'][n] for r in group]) for n in names}
        windows = three_layer(names); best = max(windows, key=lambda n: cell[n]['median'])
        single = {n[1:3]: cell[n]['median'] for n in single_layer(names)}
        reading = 'image_window' if cell[WINDOW]['median'] >= ROUTE_AT else 'elsewhere' if cell[best]['median'] >= ROUTE_AT else 'diffuse'
        out[pair] = dict(reading=reading, readable=readable, states=len(group), window=cell[WINDOW], best=best, best_value=cell[best],
                         best_by_group={g: max(((n, cell[n]['median']) for n in windows if n.endswith('_' + g)), key=lambda x: x[1])
                                        for g in ('all', 'image', 'text')},
                         peak_single=max(single, key=single.get), single_all=single,
                         single_image={n[1:3]: cell[n]['median'] for n in single_layer(names, 'image')})
    readable = [v for v in out.values() if v['reading'] != 'unreadable']
    overall = ('insufficient' if len(readable) < MIN_PAIRS else
               'shared' if all(v['reading'] == 'image_window' for v in readable) else 'differs')
    return dict(reading=overall, pairs=out)


def pooled_median(groups, draws=DRAWS, seed=SEED):
    """Median of the pooled values of several cells; percentile bootstrap resampling values within each cell."""
    groups = [np.asarray(g, float) for g in groups if len(g)]
    if not groups:
        return None
    rng = np.random.default_rng(seed)
    boot = [np.median(np.concatenate([rng.choice(g, len(g)) for g in groups])) for _ in range(draws)]
    return dict(median=float(np.median(np.concatenate(groups))), lo=float(np.quantile(boot, .025)), hi=float(np.quantile(boot, .975)),
                n=int(sum(len(g) for g in groups)), cells=len(groups))


def gamma_cells(rows, group='image', inits=READING_INITS):
    """(pair, phase) -> paired gap ratios (band gap at the phase / at f = 0, same pair and init) of the held-out states."""
    full = [r for r in rows if r['kind'] == 'full' and r['init'] in inits]
    base = {(r['pair'], r['init']): M.band_gap(r, group) for r in full if r['phase'] == 0}
    cells = defaultdict(list)
    for r in full:
        b = base.get((r['pair'], r['init']))
        if r['phase'] != 0 and b is not None and b > 0:
            cells[(r['pair'], r['phase'])].append(M.band_gap(r, group) / b)
    return dict(cells)


def fade(rows, inits=READING_INITS):
    """H2: does the image-token mark fade in the held-out failure cells and stay in the success cells (pooled per class)?"""
    image, text = gamma_cells(rows, 'image', inits), gamma_cells(rows, 'text', inits)
    pooled = lambda classes: pooled_median([v for c, v in sorted(image.items()) if R.cell_class(c) in classes])
    failure, success, middle = pooled(('commits', 'derails')), pooled(('success',)), pooled(('middle',))
    if failure is None or success is None:
        reading = 'insufficient'
    elif failure['median'] <= M.FADE and failure['hi'] < 1 and success['lo'] >= M.KEEP:
        reading = 'image_route_fades'
    elif failure['lo'] >= M.KEEP:
        reading = 'image_route_kept'
    else:
        reading = 'mixed'
    cells = {'%s f=%s' % c: dict(cls=R.cell_class(c), image=M.bootstrap_median(v), text=M.bootstrap_median(text.get(c, [])))
             for c, v in sorted(image.items())}
    return dict(reading=reading, failure=failure, success=success, middle=middle, cells=cells)


def ranks(x):
    """Ranks from 1 with ties given their average rank."""
    x = np.asarray(x, float); order = np.argsort(x, kind='mergesort'); out = np.empty(len(x)); i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        out[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return out


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    if len(rx) < 2 or rx.std() == 0 or ry.std() == 0:
        return float('nan')
    return float(np.corrcoef(rx, ry)[0, 1])


def failure_cells(rows, loop, value, inits=READING_INITS):
    """(pair, phase) -> [(value(row) or None, amp2 - native b_done of the same SW unit in the repair batch)] over the failure
    cells; raises when a failure state has no native and amp2 run there."""
    cells = defaultdict(list)
    for r in rows:
        if r['cell'] not in FAILURE_CLASSES or r['init'] not in inits:
            continue
        arms = loop.get(r['sw'], {})
        if set(arms) != {'native', 'amp2'}:
            raise RuntimeError('No closed-loop native and amp2 runs for %s' % r['sw'])
        cells[(r['pair'], r['phase'])].append((value(r), int(arms['amp2']) - int(arms['native'])))
    return dict(cells)


def cell_values(items):
    values = [v for v, _ in items if v is not None]
    return (float(np.median(values)) if values else None), float(np.mean([g for _, g in items]))


def proxy(cells, draws=DRAWS, seed=SEED):
    """H3: Spearman rho over the failure cells between the cell median of an open-loop value and the closed-loop gain;
    the interval resamples states within each cell (value and gain from the same states)."""
    point = {k: cell_values(v) for k, v in sorted(cells.items())}
    usable = [k for k, (s, _) in point.items() if s is not None]
    table = {'%s f=%s' % k: dict(value=point[k][0], gain=point[k][1], states=len(cells[k]),
                                 readable=sum(v is not None for v, _ in cells[k])) for k in point}
    if len(usable) < MIN_CELLS:
        return dict(reading='insufficient', cells=table)
    rho = spearman([point[k][0] for k in usable], [point[k][1] for k in usable])
    rng = np.random.default_rng(seed); boot = []
    for _ in range(draws):
        xs, ys = [], []
        for k in usable:
            items = cells[k]; s, g = cell_values([items[i] for i in rng.integers(0, len(items), len(items))])
            if s is not None:
                xs.append(s); ys.append(g)
        boot.append(spearman(xs, ys) if len(xs) >= MIN_CELLS else np.nan)
    boot = np.asarray(boot, float); lo, hi = float(np.nanquantile(boot, .025)), float(np.nanquantile(boot, .975))
    if np.isnan(rho):
        reading = 'insufficient'
    else:
        reading = 'proxy_tracks' if rho >= TRACKS and lo > 0 else 'proxy_weak' if rho > 0 else 'proxy_fails'
    return dict(reading=reading, rho=rho, lo=lo, hi=hi, cells_used=len(usable), undefined_draws=int(np.isnan(boot).sum()), cells=table)


def weak_reason(cells, route):
    """H4: for each weak pair, the first rule that holds: window, nothing_to_push, weak_push, downstream. `cells` are the
    failure cells with the open-loop push as the value; `route` is H1."""
    pairs = sorted({p for p, _ in cells})
    gains = {p: float(np.mean([g for k, v in cells.items() if k[0] == p for _, g in v])) for p in pairs}
    effective = sorted(p for p in pairs if gains[p] >= C.GAIN); weak = sorted(p for p in pairs if gains[p] < C.GAIN)
    if effective != sorted(EFFECTIVE) or weak != sorted(WEAK):
        raise RuntimeError('Effective/weak pairs differ from the frozen lists: %r / %r' % (effective, weak))
    rate = lambda items: float(np.mean([v is not None for v, _ in items]))
    reference = [k for k in sorted(cells) if k[0] in EFFECTIVE]
    ref_rate = float(np.median([rate(cells[k]) for k in reference]))
    ref_values = [cell_values(cells[k])[0] for k in reference]
    ref_push = float(np.median([v for v in ref_values if v is not None]))
    out = {}
    for pair in WEAK:
        items = [x for k in sorted(cells) if k[0] == pair for x in cells[k]]
        h1 = route['pairs'][pair]['reading']; pushes = [v for v, _ in items if v is not None]
        push = float(np.median(pushes)) if pushes else None
        if h1 in ('elsewhere', 'diffuse'):
            reading = 'window'
        elif rate(items) < HALF * ref_rate:
            reading = 'nothing_to_push'
        elif push is None or push < HALF * ref_push:
            reading = 'weak_push'
        else:
            reading = 'downstream'
        out[pair] = dict(reading=reading, t0_route=h1, window_testable=h1 != 'unreadable', readable_rate=rate(items), push=push,
                         gain=gains[pair], states=len(items), cells=['%s f=%s' % k for k in sorted(cells) if k[0] == pair])
    return dict(pairs=out, reference=dict(cells=['%s f=%s' % k for k in reference], readable_rate=ref_rate, push=ref_push),
                gains=gains, effective=effective, weak=weak)


def describe(rows, loop=None, inits=READING_INITS):
    """Per (kind, pair, phase) cell of the reading inits: localization, gap ratios, the push and the closed-loop counts."""
    image, text = gamma_cells(rows, 'image', inits), gamma_cells(rows, 'text', inits)
    out = {}
    for key in sorted({(r['kind'], r['pair'], r['phase']) for r in rows}):
        group = [r for r in rows if (r['kind'], r['pair'], r['phase']) == key and r['init'] in inits]
        if not group:
            continue
        med = lambda xs: M.bootstrap_median(xs)
        cell = dict(cls=group[0]['cell'], states=len(group), stage1_b_done=sum(r['b_done'] for r in group),
                    push_readable=sum(r['push']['readable'] for r in group), push_amp2=med([r['push']['amp2'] for r in group]),
                    push_rand2=med([r['push']['rand2'] for r in group]), raw_amp2=med([r['push']['raw_amp2'] for r in group]),
                    raw_rand2=med([r['push']['raw_rand2'] for r in group]), ab_floors=med([r['push']['ab_floors'] for r in group]))
        if key[0] == 'full':
            names = list(group[0]['interface']); windows = three_layer(names)
            values = {n: med([r['interface'][n] for r in group]) for n in names}
            ok = {n: v['median'] for n, v in values.items() if v is not None}
            single = {n[1:3]: ok[n] for n in single_layer(names) if n in ok}
            cell.update(readable=sum(r['interface'][WINDOW] is not None for r in group), window=values[WINDOW],
                        full=med([float(np.mean(r['full_e'])) if r['full_e'] else None for r in group]),
                        best={g: max(((n, ok[n]) for n in windows if n.endswith('_' + g) and n in ok), key=lambda x: x[1], default=None)
                              for g in ('all', 'image', 'text')},
                        peak_single=max(single, key=single.get) if single else None,
                        gamma_image=med(image.get(key[1:], [])), gamma_text=med(text.get(key[1:], [])))
        if loop is not None:
            arms = [loop.get(r['sw'], {}) for r in group]
            if all(set(a) == {'native', 'amp2'} for a in arms):
                cell.update(native=sum(a['native'] for a in arms), amp2=sum(a['amp2'] for a in arms))
        out['%s %s f=%s' % key] = cell
    return out


def backbone_f0(rows, inits=READING_INITS):
    """Per held-out pair at f = 0: the cell median E of patching the image-token residual entering each prefix layer."""
    out = {}
    for pair in HELD_LABELS:
        group = [r for r in rows if r['kind'] == 'full' and r['pair'] == pair and r['phase'] == 0 and r['init'] in inits]
        if group:
            names = [n for n in group[0]['backbone'] if n.endswith('_image')]
            out[pair] = {n[3:5]: (lambda c: None if c is None else c['median'])(M.bootstrap_median([r['backbone'].get(n) for r in group]))
                         for n in names}
    return out


# ---------------------------------------------------------------- summary
def gamma_value(rows):
    """Row -> its image gap ratio (band gap / the band gap of the same pair and init at f = 0), None without a base."""
    base = {(r['pair'], r['init']): M.band_gap(r) for r in rows if r['phase'] == 0}
    return lambda r: M.band_gap(r) / base[(r['pair'], r['init'])] if base.get((r['pair'], r['init']), 0) > 0 else None


def logged_b_linf(r):
    return r['native_b_vs_logged_linf'] if 'native_b_vs_logged_linf' in r else r['native_vs_logged']['b_linf']


def summarize(root, require_tool_ok=False, repair_root=REPAIR_ROOT, pilot_root=PILOT_ROOT):
    root = Path(root); plan_doc, rows = load(root)
    tool, detail = tool_reading(rows)
    inits = tuple(plan_doc['inits']); logged = [logged_b_linf(r) for r in rows]
    summary = dict(mode=plan_doc['mode'], tool=tool, tool_detail=detail, states=len(rows), reading_inits=list(inits),
                   native_b_vs_logged_linf=dict(median=float(np.median(logged)), max=float(max(logged)),
                                                exact=sum(x == 0 for x in logged)))
    if plan_doc['mode'] == 'formal' and tool == 'tool_ok':
        loop = closed_loop_runs(repair_root)
        located = [r for r in rows if r['kind'] == 'full'] + [r for r in pilot_rows(pilot_root) if r['init'] in inits]
        route = t0_route(rows, inits); push_cells = failure_cells(rows, loop, lambda r: r['push']['amp2'], inits)
        summary.update(
            t0_route=route, fade=fade(rows, inits), proxy=proxy(push_cells), weak_reason=weak_reason(push_cells, route),
            proxy_report_only=dict(
                window_e=proxy(failure_cells(located, loop, lambda r: r['interface'][WINDOW], inits)),
                gamma_image=proxy(failure_cells(located, loop, gamma_value(located), inits)),
                push_rand2=proxy(failure_cells(rows, loop, lambda r: r['push']['rand2'], inits))),
            cells=describe(rows, loop, inits), backbone_f0=backbone_f0(rows, inits),
            sources=dict(repair_root=str(repair_root), pilot_root=str(pilot_root)))
    else:
        summary.update(cells=describe(rows, None, inits))
    summary.update(O.provenance())
    write(root/'SUMMARY.json', summary)
    brief = dict(tool=tool, tool_detail=detail)
    if 't0_route' in summary:
        brief.update(t0_route=dict(reading=summary['t0_route']['reading'], pairs={p: v['reading'] for p, v in summary['t0_route']['pairs'].items()}),
                     fade={k: summary['fade'][k] for k in ('reading', 'failure', 'success')},
                     proxy={k: summary['proxy'].get(k) for k in ('reading', 'rho', 'lo', 'hi')},
                     weak_reason={p: v['reading'] for p, v in summary['weak_reason']['pairs'].items()})
    print(json.dumps(brief, indent=1, default=str), flush=True)
    if require_tool_ok and (tool != 'tool_ok' or plan_doc['mode'] != 'smoke'):
        raise SystemExit('Gate failed: %s on a %s root' % (tool, plan_doc['mode']))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['plan', 'run', 'summarize'])
    parser.add_argument('root'); parser.add_argument('out', nargs='?'); parser.add_argument('mode', nargs='?', choices=['smoke', 'formal'])
    parser.add_argument('--shard', type=int, default=0); parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--checkpoint'); parser.add_argument('--require-tool-ok', action='store_true'); args = parser.parse_args()
    if args.action == 'plan':
        plan(args.root, args.out, args.mode)
    elif args.action == 'run':
        run(args.root, args.shard, args.shards, args.checkpoint)
    else:
        summarize(args.root, require_tool_ok=args.require_tool_ok)
