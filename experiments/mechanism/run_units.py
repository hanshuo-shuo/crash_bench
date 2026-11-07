"""Runner, readings and summary of the unit study (docs/generative_verifier/HIL_MECHANISM_UNITS.md).

States are the pilot's open-loop states (selection.py: LIBERO-Goal 1->8, 7->8, 2->9 at f = 0, .25, .5, .75), read on the stored
switch-point input under prompt A and prompt B. Per state, noise seed 0 ranks the units (units.unit_scores) and seeds 1-3
evaluate the real patches (units.unit_eval). Inference only: no simulator, no training, no API.

  run_units.py plan BRANCH_ROOT OUT_JSON smoke|formal
  run_units.py run ROOT --shard K --shards N --checkpoint DIR
  run_units.py summarize ROOT [--require-tool-ok]
"""
import argparse
import gzip
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent/'feasibility'))
import metrics as M  # noqa: E402
import neuron_readings as NR  # noqa: E402
import run_open_loop as O  # noqa: E402
import selection as S  # noqa: E402
import units as U  # noqa: E402

SEEDS = 4                        # seed 0 ranks, seeds 1-3 evaluate
READING_INITS = M.READING_INITS  # 2-19
SMOKE_INITS = S.SMOKE_INITS      # 0-1
CEILING_MIN = .3                 # a state enters a reading only if patching every image-row MLP unit moves E (or E back) by >= .3
MIN_STATES = 20
SPARSE, MODERATE = 100, 10_000   # R1/R2: median units for half the ceiling
HALF = .5
FIDELITY_MEDIAN, FIDELITY_MAX = .1, .5   # pilot P1: |path(hooked A) - path(Policy.infer A)| in noise floors
DENOISE_TOL = 1e-5
AFFINE_TOL = 1e-4
E_CHECK_TOL = 1e-4
FD_K, FD_T = 100, .1      # report only: finite-difference check of the attribution on the top 100 act units
SPECS = U.specs()
NAMES = [s[0] for s in SPECS]
read, write, sha, provenance = O.read, O.write, O.sha, O.provenance


# ---------------------------------------------------------------- plan
def plan(branch_root, out_json, mode):
    branch_root = Path(branch_root); bp = read(branch_root/'PLAN.json')
    with gzip.open(branch_root/'RUNS.jsonl.gz', 'rt') as stream:
        runs = [json.loads(x) for x in stream]
    inits = SMOKE_INITS if mode == 'smoke' else READING_INITS
    states = S.attach_files(S.select(bp, runs, inits=inits), branch_root, bp['prefix_root'])
    for s in states:
        s['tokens_a'] = bp['prompt_tokens'].get(s['prompt_a'])
    out = dict(experiment='mechanism-units', mode=mode, branch_root=str(branch_root), prefix_root=bp['prefix_root'],
               inits=list(inits), seeds=SEEDS, states=states, summary=S.summary(states))
    write(out_json, out)
    return out


# ---------------------------------------------------------------- GPU
def set_seed(sw, salt):
    return int(hashlib.sha256(('mechanism-units:%s:%s' % (salt, sw)).encode()).hexdigest()[:8], 16)


def unit_sources(sc, n, width, sw):
    idx = sc['idx']   # act, nec, kv
    return np.stack([idx[0], U.random_same_position(idx[0], n, width, set_seed(sw, 'act')), idx[2], idx[1],
                     U.random_same_position(idx[1], n, width, set_seed(sw, 'nec'))])


def neuron_sources(sc, depth, width, sw):
    nk = max(U.NEURON_GRID); neuron = sc['neuron']
    act, kv, nec = U.neuron_order(neuron[0], nk), U.neuron_order(neuron[2], nk), U.neuron_order(-neuron[1], nk)
    rand = U.random_same_layer(act, width, depth, set_seed(sw, 'neuron'))
    orders = np.stack([act, rand, kv, nec])
    return orders, np.stack([U.rank_array(o, depth, width) for o in orders])


def expected_count(spec, depth, valid, width):
    _, kind, _, k, _ = spec
    return {U.NONE: 0, U.ALL: depth * valid * width, U.UNIT: k, U.NEURON: k * valid}[kind]


def composition(idx, n, width, k):
    """Where the top k units sit: distinct (layer, row) pairs, distinct rows, distinct neurons, and the layer histogram."""
    u = np.asarray(idx[:k], np.int64); layer = u // (n * width); row = (u // width) % n; unit = u % width
    return dict(k=k, positions=int(len(set(zip(layer.tolist(), row.tolist())))), rows=int(len(set(row.tolist()))),
                neurons=int(len(set(zip(layer.tolist(), unit.tolist())))), layers=np.bincount(layer, minlength=18).tolist(),
                camera=np.bincount(row // 256, minlength=3).tolist())


def effects(env, ok, readable, names=NAMES):
    """Per spec: E at every seed [SEEDS], the held-out mean (seeds 1-3, defined ones; None if the state is unreadable), and the
    in-sample E at the ranking seed."""
    a, b, xs = env[0], env[1], env[2:]
    per = {name: M.effect(xs[i], a, b).tolist() for i, name in enumerate(names)}
    held = {name: (float(np.mean(np.asarray(v[1:])[ok])) if readable else None) for name, v in per.items()}
    return per, held, {name: v[0] for name, v in per.items()}


def run_state(patcher, policy, state, folder, scale, seeds_per_state=SEEDS, spec_list=SPECS):
    from serve import digest
    started = time.monotonic()
    data_a, data_b, _, pixel = O.inputs(state)
    seeds = [(state['seed'] + j) % O.SEED_SPAN for j in range(seeds_per_state)]
    sc = patcher.unit_scores(data_a, data_b, seeds[0], scale)
    depth, width = sc['neuron'].shape[1:]; n = sc['rowmap'].shape[1]
    uidx = unit_sources(sc, n, width, state['sw']); orders, nrank = neuron_sources(sc, depth, width, state['sw'])
    names = [x[0] for x in spec_list]
    env, count = patcher.unit_eval(data_a, data_b, seeds, uidx, nrank, spec_list)
    native_a = O.native(policy, data_a, seeds[1])
    fd = patcher.unit_eps(data_a, data_b, seeds[0], scale, uidx[0][:FD_K], FD_T) / FD_T   # report only
    a, b = env[0], env[1]
    floor = M.noise_floor(a[1:]); ok = M.defined(a[1:], b[1:], floor); readable = int(ok.sum()) >= M.MIN_SEEDS
    per, held, rank_e = effects(env, ok, readable, names)
    valid = int(sc['checks'][4])
    expected = [expected_count(x, depth, valid, width) for x in spec_list]
    vals = sc['vals']; cums = np.cumsum(vals.astype(np.float64), axis=1)
    pred = {r: {str(k): float(cums[i, min(k, cums.shape[1]) - 1]) for k in U.UNIT_GRID} for i, r in enumerate(U.RANKINGS)}
    distance = lambda x, y: float(np.linalg.norm(M.path(x) - M.path(y)))
    row = dict(
        id=state['id'], pair=state['pair'], phase=state['phase'], init=state['init'], b_done=state['b_done'], seeds=seeds,
        input_b_exact=digest(data_b) == state['b_first']['input_sha256'], a_twin_pixel_max_abs=pixel,
        floor=floor, defined=ok.tolist(), readable=readable, ab_floors=distance(a[1], b[1]) / floor if floor > 0 else None,
        e=held, e_seeds=per, e_rank=rank_e,
        noop_max_abs=float(np.abs(env[2 + names.index('none')] - a).max()),
        counts_ok=bool(all(int(c) == e for c, e in zip(count, expected))), count_mismatch=[names[i] for i, (c, e) in enumerate(zip(count, expected)) if int(c) != e],
        valid_rows=valid, fidelity_ratio=distance(a[1], native_a) / floor if floor > 0 else None,
        hook_gap=float(sc['checks'][0]), denoise_gap=float(sc['checks'][1]), e_check=[float(sc['checks'][2]), float(sc['checks'][3])],
        scores_finite=bool(all(np.isfinite(sc[k]).all() for k in ('vals', 'neuron', 'layers', 'rowmap', 'totals'))),
        totals={r: sc['totals'][i].tolist() for i, r in enumerate(U.RANKINGS)}, pred=pred,
        fd_check=dict(k=FD_K, t=FD_T, finite_difference=fd, attribution=float(np.sum(sc['vals'][0][:FD_K], dtype=np.float64))),
        layer_scores={r: sc['layers'][i].tolist() for i, r in enumerate(U.RANKINGS)},
        composition={r: [composition(uidx[j], n, width, k) for k in (10, 100, 1000)] for r, j in (('act', 0), ('nec', 3), ('kv', 2))},
        **provenance())
    row['elapsed_seconds'] = time.monotonic() - started
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder/'UNIT_IDX.npy', uidx); np.save(folder/'NEURON_ORDER.npy', orders.astype(np.int32))
    np.save(folder/'ACTIONS.npy', env.astype(np.float32)); np.save(folder/'ROWMAP.npy', sc['rowmap'].astype(np.float32))
    np.save(folder/'VALS.npy', vals.astype(np.float32)); np.save(folder/'COUNT.npy', np.asarray(count, np.int64))
    write(folder/'ROW.json', row)
    write(folder/'HASHES.json', {p.name: sha(p) for p in sorted(folder.iterdir()) if p.name != 'HASHES.json'})
    return row


def run(root, shard, shards, checkpoint):
    import jax
    from openpi.policies import policy_config
    from openpi.training import config
    root = Path(root); plan_doc = read(root/'PLAN.json'); states = read(root/'STATES.json')['states']
    if sha(root/'STATES.json') != plan_doc['states_sha256']:
        raise RuntimeError('STATES.json changed')
    chosen = set(plan_doc['state_ids'])
    mine = [s for i, s in enumerate(s for s in states if s['id'] in chosen) if i % shards == shard]
    policy = policy_config.create_trained_policy(config.get_config('pi05_libero'), checkpoint)
    if policy._is_pytorch_model:
        raise RuntimeError('JAX policy required')
    patcher = U.UnitPatcher(policy)
    data_a, _, _, _ = O.inputs(mine[0]); scale, affine = patcher.action_scale(data_a)
    write(root/('SCALE_%d.json' % shard), dict(scale=scale.tolist(), affine_deviation=affine))
    progress = root/('PROGRESS_%d.json' % shard); done = []
    for state in mine:
        folder = root/'states'/state['id']
        if (folder/'HASHES.json').exists():
            raise RuntimeError('State already written: %s' % state['id'])
        row = run_state(patcher, policy, state, folder, scale); done.append(state['id'])
        write(progress, dict(done=len(done), total=len(mine), last=state['id'], last_seconds=row['elapsed_seconds']))
    write(root/('SHARD_%d_COMPLETE.json' % shard), dict(passed=True, states=done, scale=scale.tolist(), affine_deviation=affine,
                                                       jax_devices=[str(d) for d in jax.devices()], **provenance()))


# ---------------------------------------------------------------- readings (pure)
def ceiling(row, kind):
    """The all-units effect: E of all_suff (sufficiency), 1 - E of all_nec (necessity); None if unreadable."""
    if not row['readable']:
        return None
    return row['e']['all_suff'] if kind == 'suff' else 1 - row['e']['all_nec']


def fraction(row, name, kind):
    c = ceiling(row, kind)
    if c is None or c <= 0:
        return None
    v = row['e'][name]
    return v / c if kind == 'suff' else (1 - v) / c


def k_half(row, prefix, grid, kind):
    """Smallest grid k whose patch reaches HALF of the ceiling; inf if none does."""
    for k in grid:
        f = fraction(row, prefix % k, kind)
        if f is not None and f >= HALF:
            return k
    return float('inf')


def population(rows, kind, inits=READING_INITS):
    return [r for r in rows if r['init'] in inits and ceiling(r, kind) is not None and ceiling(r, kind) >= CEILING_MIN]


def sparsity(rows, prefix, grid, kind, inits=READING_INITS, scale=1):
    """R1/R2: the median over the population of the units needed for half the ceiling (scale converts neurons to units)."""
    pop = population(rows, kind, inits)
    ks = [k_half(r, prefix, grid, kind) * (scale(r) if callable(scale) else scale) for r in pop]
    if len(ks) < MIN_STATES:
        return dict(reading='insufficient', states=len(ks))
    med = float(np.median(ks))
    reading = 'sparse' if med <= SPARSE else 'moderate' if med <= MODERATE else 'diffuse'
    q = np.quantile(ks, [.25, .75])
    return dict(reading=reading, median=med, q25=float(q[0]), q75=float(q[1]), states=len(ks),
                never=int(sum(np.isinf(ks))), by_cell=by_cell(pop, lambda r: k_half(r, prefix, grid, kind)))


def by_cell(rows, value):
    out = defaultdict(list)
    for r in rows:
        out['%s f=%s' % (r['pair'], r['phase'])].append(value(r))
    return {c: dict(states=len(v), median=float(np.median(v))) for c, v in sorted(out.items())}


def curves(rows, kind, inits=READING_INITS):
    """Median fraction of the ceiling per grid point for every ranking of the given direction, over the population."""
    pop = population(rows, kind, inits); out = {}
    families = (('u_act', U.UNIT_GRID), ('u_actrand', U.UNIT_GRID), ('u_kv', U.UNIT_GRID), ('n_act', U.NEURON_GRID),
                ('n_actrand', U.NEURON_GRID), ('n_kv', U.NEURON_GRID)) if kind == 'suff' else \
               (('u_nec', U.UNIT_GRID), ('u_necrand', U.UNIT_GRID), ('n_nec', U.NEURON_GRID))
    for fam, grid in families:
        out[fam] = {str(k): (lambda v: float(np.median(v)) if v else None)([f for f in (fraction(r, '%s_%d' % (fam, k), kind) for r in pop) if f is not None])
                    for k in grid}
    return dict(states=len(pop), curves=out)


def tool_reading(rows, scales):
    ratios = [r['fidelity_ratio'] for r in rows if r['fidelity_ratio'] is not None]
    checks = dict(
        inputs=all(r['input_b_exact'] for r in rows), finite=all(r['scores_finite'] for r in rows),
        noop=all(r['noop_max_abs'] == 0 for r in rows), counts=all(r['counts_ok'] for r in rows),
        hooks=all(r['hook_gap'] <= NR.HOOK_GAP_TOL for r in rows), denoise=all(r['denoise_gap'] <= DENOISE_TOL for r in rows),
        e_check=all(abs(r['e_check'][0]) <= E_CHECK_TOL and abs(r['e_check'][1] - 1) <= E_CHECK_TOL for r in rows),
        fidelity=bool(ratios) and float(np.median(ratios)) <= FIDELITY_MEDIAN and max(ratios) <= FIDELITY_MAX,
        affine=bool(scales) and all(s['affine_deviation'] <= AFFINE_TOL for s in scales))
    return ('tool_ok' if all(checks.values()) else 'tool_failed'), dict(
        checks, fidelity_median=float(np.median(ratios)) if ratios else None, fidelity_max=max(ratios) if ratios else None,
        hook_gap_max=max(r['hook_gap'] for r in rows), denoise_gap_max=max(r['denoise_gap'] for r in rows),
        scales=[s['scale'] for s in scales])


def atp_agreement(rows, inits=READING_INITS):
    """Report only: Spearman over readable states between the attribution's linear prediction of the E change and the real
    change, for the act ranking (held-out seeds and the ranking seed) at a few k."""
    from heldout_open import spearman
    out = {}
    for k in (100, 1000, 10000):
        pop = [r for r in rows if r['init'] in inits and r['readable']]
        pred = [r['pred']['act'][str(k)] for r in pop]
        held = [r['e']['u_act_%d' % k] for r in pop]; rank = [r['e_rank']['u_act_%d' % k] for r in pop]
        out[str(k)] = dict(states=len(pop), held_out=spearman(pred, held) if len(pop) > 2 else None,
                           ranking_seed=spearman(pred, rank) if len(pop) > 2 else None)
    return out


# ---------------------------------------------------------------- summary
def load(root):
    root = Path(root); plan_doc = read(root/'PLAN.json')
    if sha(root/'STATES.json') != plan_doc['states_sha256']:
        raise RuntimeError('STATES.json changed')
    scales = []
    for k in range(plan_doc['shards']):
        done = read(root/('SHARD_%d_COMPLETE.json' % k))
        if not done['passed']:
            raise RuntimeError('Shard %d did not pass' % k)
        scales.append(dict(scale=done['scale'], affine_deviation=done['affine_deviation']))
    rows = []
    for sid in plan_doc['state_ids']:
        folder = root/'states'/sid
        for name, digest in read(folder/'HASHES.json').items():
            if sha(folder/name) != digest:
                raise RuntimeError('State output changed: %s/%s' % (sid, name))
        rows.append(read(folder/'ROW.json'))
    if {r['id'] for r in rows} != set(plan_doc['state_ids']):
        raise RuntimeError('Rows differ from the plan')
    return plan_doc, rows, scales


def summarize(root, require_tool_ok=False):
    root = Path(root); plan_doc, rows, scales = load(root)
    tool, detail = tool_reading(rows, scales)
    inits = tuple(plan_doc['inits'])
    summary = dict(mode=plan_doc['mode'], tool=tool, tool_detail=detail, states=len(rows), readable=sum(r['readable'] for r in rows))
    to_units = lambda r: r['valid_rows']
    summary['r1'] = sparsity(rows, 'u_act_%d', U.UNIT_GRID, 'suff', inits)
    summary['r2'] = sparsity(rows, 'u_nec_%d', U.UNIT_GRID, 'nec', inits)
    summary['report_only'] = dict(
        units_for_half=dict(u_act=summary['r1'], u_kv=sparsity(rows, 'u_kv_%d', U.UNIT_GRID, 'suff', inits),
                            n_act=sparsity(rows, 'n_act_%d', U.NEURON_GRID, 'suff', inits, to_units),
                            n_kv=sparsity(rows, 'n_kv_%d', U.NEURON_GRID, 'suff', inits, to_units),
                            u_actrand=sparsity(rows, 'u_actrand_%d', U.UNIT_GRID, 'suff', inits),
                            n_nec=sparsity(rows, 'n_nec_%d', U.NEURON_GRID, 'nec', inits, to_units),
                            u_necrand=sparsity(rows, 'u_necrand_%d', U.UNIT_GRID, 'nec', inits)),
        curves_suff=curves(rows, 'suff', inits), curves_nec=curves(rows, 'nec', inits), atp=atp_agreement(rows, inits),
        ceilings=by_cell([r for r in rows if r['init'] in inits and r['readable']], lambda r: r['e']['all_suff']),
        ceilings_nec=by_cell([r for r in rows if r['init'] in inits and r['readable']], lambda r: 1 - r['e']['all_nec']),
        composition_act_100=by_cell([r for r in rows if r['init'] in inits], lambda r: r['composition']['act'][1]['positions']),
        composition_act_100_rows=by_cell([r for r in rows if r['init'] in inits], lambda r: r['composition']['act'][1]['rows']))
    if tool != 'tool_ok':   # no readings without a passing tool check (the smoke's readings, on inits 0-1, are descriptive only)
        summary['r1'] = dict(summary['r1'], reading=None); summary['r2'] = dict(summary['r2'], reading=None)
    summary.update(provenance())
    write(root/'SUMMARY.json', summary)
    brief = dict(tool=tool, tool_detail={k: v for k, v in detail.items() if k != 'scales'}, states=summary['states'],
                 readable=summary['readable'], r1={k: summary['r1'].get(k) for k in ('reading', 'median', 'q25', 'q75', 'states', 'never')},
                 r2={k: summary['r2'].get(k) for k in ('reading', 'median', 'q25', 'q75', 'states', 'never')})
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
