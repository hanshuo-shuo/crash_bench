"""Readouts of the mechanism pilot from action chunks (numpy only; HIL_MECHANISM_IDEA.md sections 6-7).

A chunk is the policy's environment-space output [H, 7] (LIBERO: end-effector translation 0-2, rotation 3-5, gripper 6).
Its path is the cumulative translation over the chunk, a forward-kinematics-free stand-in for the end-effector trajectory
that Goodfire's normalized trajectory shift E is computed on.
"""
import numpy as np

NOISE_MULTIPLE = 3.0     # E is read on a seed only if |path(B) - path(A)| >= 3 noise floors
MIN_SEEDS = 2            # a state counts in a cell only if E is defined on at least 2 of its seeds
FULL_TOLERANCE = .02     # P1: the full interface patch must give E within 1 +/- .02 (re-rotation rounding at unequal counts)
FIDELITY_MEDIAN = .1     # P1: median |path(mech base) - path(native)| <= .1 noise floor ...
FIDELITY_MAX = .5        # ... and every state <= .5 noise floor
FOUND, PARTIAL = .7, .4  # P2: best 3-layer interface window, worst pair median E (Goodfire DROID pi0.5: > .7)


def path(actions):
    """[..., H, 7] -> [..., 3H]: cumulative end-effector translation over the chunk."""
    a = np.asarray(actions, float)
    return np.cumsum(a[..., :3], axis=-2).reshape(a.shape[:-2] + (-1,))


def effect(patched, base, donor):
    """E = <p - a, b - a> / |b - a|^2 on paths: 0 = stays with the base prompt, 1 = the donor prompt's full shift."""
    p, a, b = path(patched), path(base), path(donor)
    d = b - a
    return ((p - a) * d).sum(-1) / np.maximum((d * d).sum(-1), 1e-30)


def noise_floor(base_by_seed):
    """Median pairwise path distance of one prompt under different seeds (base_by_seed [n, H, 7], n >= 2)."""
    p = path(base_by_seed)
    return float(np.median([np.linalg.norm(p[i] - p[j]) for i in range(len(p)) for j in range(i + 1, len(p))]))


def defined(base, donor, floor, multiple=NOISE_MULTIPLE):
    """Per seed ([n, H, 7] each): is the base-donor path gap at least `multiple` noise floors."""
    return np.linalg.norm(path(donor) - path(base), axis=-1) >= multiple * floor


def state_effects(actions, names):
    """One state's interface or backbone result (actions [2 + m, n, H, 7]: base, donor, specs in `names` order after the
    base/donor rows) -> dict(floor, defined [n], effects {name: [n] floats}, state {name: mean over defined seeds or None})."""
    actions = np.asarray(actions)
    base, donor, specs = actions[0], actions[1], actions[2:]
    if len(specs) != len(names):
        raise ValueError('Spec count %d does not match %d names' % (len(specs), len(names)))
    floor = noise_floor(base); ok = defined(base, donor, floor)
    effects = {name: effect(specs[i], base, donor) for i, name in enumerate(names)}
    state = {name: float(e[ok].mean()) if ok.sum() >= MIN_SEEDS else None for name, e in effects.items()}
    return dict(floor=floor, defined=ok.tolist(), effects={k: v.tolist() for k, v in effects.items()}, state=state)


def bootstrap_median(values, draws=2000, seed=0):
    """Median and percentile-bootstrap 95% interval over states (one state per init within a (pair, phase) cell)."""
    v = np.asarray([x for x in values if x is not None], float)
    if not len(v):
        return None
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(v, (draws, len(v))), axis=1)
    return dict(median=float(np.median(v)), lo=float(np.quantile(meds, .025)), hi=float(np.quantile(meds, .975)), n=len(v))


def fidelity_reading(rows):
    """P1 (frozen before the GPU check). rows: one dict per state with input_b_exact, noop_max_abs, full_e (list over defined
    seeds), mech_native_ratio (|path(mech A) - path(native A)| / floor, same job). Returns (reading, details)."""
    full = [e for r in rows for e in r['full_e']]
    ratios = [r['mech_native_ratio'] for r in rows]
    checks = dict(inputs=all(r['input_b_exact'] for r in rows), noop=all(r['noop_max_abs'] == 0 for r in rows),
                  full=bool(full) and all(abs(e - 1) <= FULL_TOLERANCE for e in full),
                  fidelity=bool(ratios) and float(np.median(ratios)) <= FIDELITY_MEDIAN and max(ratios) <= FIDELITY_MAX)
    return ('tool_ok' if all(checks.values()) else 'tool_failed'), dict(checks, median_ratio=float(np.median(ratios)) if ratios
                                                                         else None, full_seeds=len(full))


# P3 (frozen with section 7 after the P1 smoke; written after seeing it, so the smoke inits 0-1 are excluded from the readings).
# Cells by the stage 1 switch rate against the f = 0 ceiling: success >= .9 of it, failure <= .5 of it (7->8 at .25, .75 of
# the ceiling, is neither and only described).
SUCCESS_CELLS = (('1->8', .25), ('1->8', .5), ('1->8', .75))
FAILURE_CELLS = (('7->8', .5), ('7->8', .75), ('2->9', .25), ('2->9', .5), ('2->9', .75))
READING_INITS = tuple(range(2, 20))
GAP_LAYERS = (12, 18)    # the interface band of the smoke (single-layer peak 14, best windows 12-14 .. 14-16) and above
FADE, KEEP = .8, .9      # R2: failure cells median gap ratio <= .8 with upper bound < 1; success cells lower bound >= .9
AUC_STRONG = .7          # R3
MIN_EACH = 5             # R3: a cell enters only with >= 5 successful and >= 5 failed switches among the reading inits


def band_gap(row, group='image'):
    """Mean relative value gap B vs A over GAP_LAYERS for one state."""
    return float(np.mean(row['gaps'][group][GAP_LAYERS[0]:GAP_LAYERS[1]]))


def gap_ratios(rows, pair, phase, group='image'):
    """Per init of the reading inits: band gap at `phase` / band gap at f = 0 of the same init (paired)."""
    rows = [r for r in rows if r['init'] in READING_INITS and r['pair'] == pair]
    base = {r['init']: band_gap(r, group) for r in rows if r['phase'] == 0}
    return [(r, band_gap(r, group) / base[r['init']]) for r in rows if r['phase'] == phase and base.get(r['init'], 0) > 0]


def route_reading(rows):
    """R2: does the instruction's mark on the image tokens at the interface band fade where switching fails?"""
    cells = {'%s f=%s' % c: bootstrap_median([x for _, x in gap_ratios(rows, *c)]) for c in SUCCESS_CELLS + FAILURE_CELLS}
    success = [cells['%s f=%s' % c] for c in SUCCESS_CELLS]; failure = [cells['%s f=%s' % c] for c in FAILURE_CELLS]
    if any(c is None for c in success + failure):
        return 'insufficient', cells
    if all(c['median'] <= FADE and c['hi'] < 1 for c in failure) and all(c['lo'] >= KEEP for c in success):
        return 'image_route_fades', cells
    if all(c['lo'] >= KEEP for c in failure):
        return 'image_route_kept', cells
    return 'mixed', cells


def auroc(positive, negative):
    p, n = np.asarray(positive, float)[:, None], np.asarray(negative, float)[None, :]
    return float(((p > n) + .5 * (p == n)).mean())


def switch_auc(rows, draws=2000, seed=0):
    """R3: within (pair, phase) cells holding both outcomes, does a larger paired gap ratio go with a successful switch?
    Cell-stratified AUROC (weights n_success * n_failure), percentile bootstrap resampling states within each cell."""
    groups = []
    for pair in sorted({r['pair'] for r in rows}):
        for phase in (.25, .5, .75):
            items = gap_ratios(rows, pair, phase)
            pos = [x for r, x in items if r['b_done']]; neg = [x for r, x in items if not r['b_done']]
            if len(pos) >= MIN_EACH and len(neg) >= MIN_EACH:
                groups.append(('%s f=%s' % (pair, phase), pos, neg))
    if not groups:
        return dict(reading='insufficient', cells=[])

    def pooled(gs):
        w = np.asarray([len(p) * len(n) for _, p, n in gs], float)
        return float(np.dot(w, [auroc(p, n) for _, p, n in gs]) / w.sum())
    rng = np.random.default_rng(seed); point = pooled(groups)
    boot = [pooled([(c, rng.choice(p, len(p)), rng.choice(n, len(n))) for c, p, n in groups]) for _ in range(draws)]
    lo, hi = float(np.quantile(boot, .025)), float(np.quantile(boot, .975))
    reading = 'gap_predicts_switch' if point >= AUC_STRONG and lo > .5 else 'gap_weakly_predicts' if lo > .5 else 'gap_does_not_predict'
    return dict(reading=reading, auroc=point, lo=lo, hi=hi,
                cells=[dict(cell=c, success=len(p), failure=len(n), auroc=auroc(p, n)) for c, p, n in groups])


def interface_reading(best_by_pair):
    """P2 (frozen before the GPU smoke). best_by_pair: pair -> the largest cell median E over the 3-layer all-token windows at
    f = 0. The worst pair decides."""
    worst = min(best_by_pair.values())
    return 'interface_found' if worst >= FOUND else 'interface_partial' if worst >= PARTIAL else 'interface_absent'
