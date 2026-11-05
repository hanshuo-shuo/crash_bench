"""Pure rules of the P4 neuron study (docs/generative_verifier/HIL_MECHANISM_NEURONS.md sections 4-6): rankings, random
sets, the open-loop tool check and the readings N0-N2 (numpy only)."""
import hashlib

import numpy as np

import metrics as M

K_GRID = (10, 30, 100, 300, 1000, 3000, 10000)
TOP_PROFILE = 10                        # neurons per pair that get profiles, single-neuron patches and explanations
DISCOVERY_INITS = tuple(range(10))      # f = 0 states of these inits rank the neurons
READING_INITS = tuple(range(10, 20))    # every open-loop reading uses only these inits
SMOKE_INITS = (0, 1)
MLP_ROUTE, ATTENTION_ROUTE = .5, .3     # N0: median rho_mlp at f = 0
HALF = .5                               # N1: k* reaches HALF of the pair's rho_mlp ceiling
SPARSE_K = 1000                         # N1: sparse when every pair's k* <= SPARSE_K
FADE, KEEP = M.FADE, M.KEEP             # N2, as R2
SUCCESS_CELLS, FAILURE_CELLS = M.SUCCESS_CELLS, M.FAILURE_CELLS
IDENTITY_TOL = 1e-3                     # tool check 3
HOOK_GAP_TOL = 1e-2                     # tool check 4 (keys/values, bf16)


def patch_names():
    return ('noop', 'full', 'attn', 'mlp') + tuple('top%d' % k for k in K_GRID) + tuple('rand%d' % k for k in K_GRID)


def norm_names():
    return tuple('top%d' % k for k in K_GRID) + ('all',)


def seed_of(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def rank(scores, hooks):
    """Mean attribution over a pair's discovery states (scores [n, L, F]) -> (layer, unit, score) on the hook layers, best
    first (stable on ties by layer then unit)."""
    mean = np.asarray(scores, np.float64).mean(0)[:len(hooks)]
    flat = mean.reshape(-1); order = np.argsort(-flat, kind='stable')
    width = mean.shape[1]
    return [(int(i // width), int(i % width), float(flat[i])) for i in order]


def random_set(top, k, pair, width, salt='neurons-random'):
    """Same per-layer counts as the first k of `top`, drawn uniformly from the other units of each layer."""
    chosen = {(l, n) for l, n, *_ in top[:k]}; counts = {}
    for l, n in chosen:
        counts[l] = counts.get(l, 0) + 1
    rng = np.random.default_rng(seed_of('%s:%s:%d' % (salt, pair, k))); out = []
    for l in sorted(counts):
        pool = np.asarray([n for n in range(width) if (l, n) not in chosen])
        out += [(l, int(n)) for n in sorted(rng.choice(pool, counts[l], replace=False))]
    return out


def cell(rows, pair, phase):
    return [r for r in rows if r['pair'] == pair and r['phase'] == phase and r['init'] in READING_INITS]


def tool_reading(rows, screen, ranking_ok):
    """Open-loop tool check (section 5). rows: measured states; screen: screening rows."""
    ratios = [r['mech_native_ratio'] for r in rows if r['mech_native_ratio'] is not None]
    checks = dict(
        inputs=all(r['input_b_exact'] for r in rows + screen),
        noop=all(r['rho']['noop'] == 0 and r['noop_max_abs'] == 0 for r in rows),
        identity=all(abs(r['rho']['full'] - 1) <= IDENTITY_TOL and abs(r['rho']['attn'] - 1) <= IDENTITY_TOL for r in rows),
        hooks=all(r['hook_gap'] <= HOOK_GAP_TOL for r in rows),
        fidelity=bool(ratios) and float(np.median(ratios)) <= M.FIDELITY_MEDIAN and max(ratios) <= M.FIDELITY_MAX,
        attribution=all(r['finite'] for r in screen) and ranking_ok)
    detail = dict(checks, identity_worst=max(max(abs(r['rho']['full'] - 1), abs(r['rho']['attn'] - 1)) for r in rows),
                  hook_gap_max=max(r['hook_gap'] for r in rows), fidelity_median=float(np.median(ratios)) if ratios else None,
                  fidelity_max=max(ratios) if ratios else None, states=len(rows), screened=len(screen))
    return ('tool_ok' if all(checks.values()) else 'tool_failed'), detail


def median_of(rows, get):
    return M.bootstrap_median([get(r) for r in rows])


def route_reading(rows, pairs):
    """N0 on the reading inits at f = 0."""
    cells = {p: median_of(cell(rows, p, 0), lambda r: r['rho']['mlp']) for p in pairs}
    if any(c is None for c in cells.values()):
        return 'insufficient', cells
    med = [c['median'] for c in cells.values()]
    reading = 'mlp_route' if min(med) >= MLP_ROUTE else 'attention_route' if max(med) < ATTENTION_ROUTE else 'shared'
    return reading, cells


def sparsity(rows, pair):
    """N1 for one pair: the rho_mlp ceiling, rho_top(k) and rho_rand(k) medians, and k*."""
    group = cell(rows, pair, 0)
    if not group:
        return None
    ceiling = median_of(group, lambda r: r['rho']['mlp'])
    top = {k: median_of(group, lambda r, k=k: r['rho']['top%d' % k]) for k in K_GRID}
    rand = {k: median_of(group, lambda r, k=k: r['rho']['rand%d' % k]) for k in K_GRID}
    star = next((k for k in K_GRID if ceiling['median'] > 0 and top[k]['median'] >= HALF * ceiling['median']), None)
    return dict(ceiling=ceiling, top=top, rand=rand, k_star=star,
                singles=[median_of(group, lambda r, i=i: r['rho']['single%d' % i]) for i in range(1, TOP_PROFILE + 1)])


def sparsity_reading(rows, pairs):
    """N1: sparse when every pair reaches half its ceiling with <= SPARSE_K neurons; distributed when some pair never does."""
    per = {p: sparsity(rows, p) for p in pairs}
    if any(v is None for v in per.values()):
        return 'insufficient', per
    stars = [v['k_star'] for v in per.values()]
    reading = 'distributed' if any(s is None for s in stars) else 'sparse' if max(stars) <= SPARSE_K else 'moderate'
    return reading, per


def k_used(per):
    """The set size each pair carries into N2 and the closed loop: k*, or the largest k of the grid (10,000)."""
    return {p: (v['k_star'] if v and v['k_star'] is not None else K_GRID[-1]) for p, v in per.items()}


def ratios(rows, pair, phase, get):
    base = {r['init']: get(r) for r in cell(rows, pair, 0)}
    return [get(r) / base[r['init']] for r in cell(rows, pair, phase) if base.get(r['init'], 0) > 0]


def fade_reading(rows, used):
    """N2: does the activation difference of each pair's k* neurons fade where switching fails (R2's rule)?"""
    get = lambda pair: (lambda r: r['D']['top%d' % used[pair]])
    cells = {'%s f=%s' % c: M.bootstrap_median(ratios(rows, c[0], c[1], get(c[0]))) for c in SUCCESS_CELLS + FAILURE_CELLS}
    success = [cells['%s f=%s' % c] for c in SUCCESS_CELLS]; failure = [cells['%s f=%s' % c] for c in FAILURE_CELLS]
    if any(c is None for c in success + failure):
        return 'insufficient', cells
    if all(c['median'] <= FADE and c['hi'] < 1 for c in failure) and all(c['lo'] >= KEEP for c in success):
        return 'neurons_fade', cells
    if all(c['lo'] >= KEEP for c in failure):
        return 'neurons_kept', cells
    return 'mixed', cells


def described(rows, used, pairs, phases):
    """Reported per cell (reading inits): rho of the k* set, rho_mlp, and the paired ratios of |G| and D_all."""
    out = {}
    for p in pairs:
        for f in phases:
            group = cell(rows, p, f)
            if not group:
                continue
            out['%s f=%s' % (p, f)] = dict(
                rho_top_kstar=median_of(group, lambda r: r['rho']['top%d' % used[p]]), rho_mlp=median_of(group, lambda r: r['rho']['mlp']),
                g_ratio=M.bootstrap_median(ratios(rows, p, f, lambda r: r['g_norm'])) if f else None,
                d_all_ratio=M.bootstrap_median(ratios(rows, p, f, lambda r: r['D']['all'])) if f else None,
                switched=sum(r['b_done'] for r in group), states=len(group))
    return out


def layer_curves(rows, pairs, hooks):
    """Reported: median single-layer rho_mlp[l] and rho_attn[l] at f = 0 on the reading inits."""
    return {p: {kind: [median_of(cell(rows, p, 0), lambda r, l=l, kind=kind: r['rho']['%s_L%02d' % (kind, l)])['median'] for l in hooks]
                for kind in ('mlp', 'attn')} for p in pairs if cell(rows, p, 0)}


def attribution_agreement(rows):
    """Reported: correlation of the state's own attribution with the real single-neuron and top-k rho."""
    xs, ys = [], []
    for r in rows:
        for i in range(1, TOP_PROFILE + 1):
            xs.append(r['atp']['single%d' % i]); ys.append(r['rho']['single%d' % i])
    sums = {k: [(r['atp']['top%d' % k], r['rho']['top%d' % k]) for r in rows] for k in K_GRID}
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0 else None
    return dict(singles=corr(xs, ys), top_k={k: dict(corr=corr(*zip(*v)), atp_median=float(np.median([a for a, _ in v])),
                                                     real_median=float(np.median([b for _, b in v]))) for k, v in sums.items()},
                total_vs_mlp=corr([r['atp']['total'] for r in rows], [r['rho']['mlp'] for r in rows]))
