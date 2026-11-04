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


def interface_reading(best_by_pair):
    """P2 (frozen before the GPU smoke). best_by_pair: pair -> the largest cell median E over the 3-layer all-token windows at
    f = 0. The worst pair decides."""
    worst = min(best_by_pair.values())
    return 'interface_found' if worst >= FOUND else 'interface_partial' if worst >= PARTIAL else 'interface_absent'
