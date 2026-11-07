"""Action-targeted neuron x image-row attribution on pi0.5 (docs/generative_verifier/HIL_MECHANISM_UNITS.md).

P4 ranked whole neurons (one GeGLU hidden unit summed over every image row) by how much of the layer 12-14 image key/value
gap they restore. This module ranks units = (layer, image row, hidden unit) of the prefix (VLM) MLP by their attributed effect
on the action itself, and patches them for real:
  scores (one program per state, at one noise seed):
    E(x) = <path(x) - path(a), path(b) - path(a)> / |path(b) - path(a)|^2 on the first action chunk, with a, b the chunks of
    prompts A and B (the pilot's E; path = cumulative end-effector translation in environment units, here through the
    per-dimension scale of the affine output transform);
    sufficiency  s = (h_B - h_A) * dE/dh at A's run       (patching a unit of A's run to B's value moves E up by ~s),
    necessity    s = (h_A - h_B) * dE/dh at B's run       (setting a unit of B's run back to A's value moves E down by ~-s),
    key/value    s = (h_B - h_A) * drho/dh at A's run     (P4's target, per unit instead of summed over rows);
    the top K units of each (largest sufficiency and key/value scores, most negative necessity scores) and the per-neuron sums.
  evaluation (one program per state, at the evaluation seeds): actions for a list of specs, each a set of units patched from
    the other prompt's run (sufficiency: A's run takes B's values; necessity: B's run takes A's values).
The gradients need a reverse-differentiable Euler loop: _denoise_scan is patching._denoise with lax.scan in place of
lax.while_loop (the same steps; the tool check compares them).
"""
import jax
import jax.numpy as jnp
import numpy as np

import neurons as N
import patching as P

F32 = jnp.float32
K_MAX = 100_000
UNIT_GRID = (10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000)
NEURON_GRID = (1, 3, 10, 30, 100, 300, 1000, 3000, 10000)
NONE, ALL, UNIT, NEURON = 0, 1, 2, 3
SUFF, NEC = 0, 1
RANKINGS = ('act', 'nec', 'kv')                         # rows of the score programs' outputs
UNIT_SOURCES = ('act', 'act_rand', 'kv', 'nec', 'nec_rand')   # rows of the unit index array given to the evaluation
NEURON_SOURCES = ('act', 'act_rand', 'kv', 'nec')             # rows of the neuron rank array given to the evaluation
BIG = 2 ** 30
_INSTALLED = False


# ---------------------------------------------------------------- pure pieces
def path_w(x, w):
    """Model-space chunks [..., H, D] -> cumulative translation in environment units [..., 3H] (w: per-dimension scale [3])."""
    y = jnp.cumsum(x[..., :3].astype(F32) * w, axis=-2)
    return y.reshape(y.shape[:-2] + (-1,))


def effect_w(x, a, b, w):
    pa, pb, px = path_w(a, w), path_w(b, w), path_w(x, w)
    d = pb - pa
    return ((px - pa) * d).sum(-1) / jnp.maximum((d * d).sum(-1), 1e-30)


def top_units(score, k):
    """Flat indices (over score.shape) and values of the k largest entries of score [L, ...]: per-layer top_k, then a merge."""
    depth = score.shape[0]; flat = score.reshape(depth, -1); kl = min(k, flat.shape[1])
    vals, idx = jax.lax.top_k(flat, kl)
    idx = idx + (jnp.arange(depth) * flat.shape[1])[:, None]
    v, j = jax.lax.top_k(vals.reshape(-1), k)
    return idx.reshape(-1)[j].astype(jnp.int32), v


def specs():
    """The evaluation specs: (name, kind, source row, k, direction)."""
    out = [('none', NONE, 0, 0, SUFF), ('all_suff', ALL, 0, 0, SUFF), ('all_nec', ALL, 0, 0, NEC)]
    for k in UNIT_GRID:
        out += [('u_act_%d' % k, UNIT, 0, k, SUFF), ('u_actrand_%d' % k, UNIT, 1, k, SUFF), ('u_kv_%d' % k, UNIT, 2, k, SUFF),
                ('u_nec_%d' % k, UNIT, 3, k, NEC), ('u_necrand_%d' % k, UNIT, 4, k, NEC)]
    for k in NEURON_GRID:
        out += [('n_act_%d' % k, NEURON, 0, k, SUFF), ('n_actrand_%d' % k, NEURON, 1, k, SUFF), ('n_kv_%d' % k, NEURON, 2, k, SUFF),
                ('n_nec_%d' % k, NEURON, 3, k, NEC)]
    return out


def random_same_position(idx, n, width, seed):
    """For each unit (flat over [L, n, width]) in rank order, a different random unit at the same layer and row; no unit is
    used twice. The first k entries thus share the top k's (layer, row) composition for every k."""
    rng = np.random.default_rng(seed); used = set(); out = np.empty(len(idx), np.int64)
    for i, u in enumerate(np.asarray(idx, np.int64)):
        base = (u // width) * width
        while True:
            c = base + int(rng.integers(width))
            if c not in used:
                used.add(c); out[i] = c; break
    return out.astype(np.int32)


def neuron_order(score, count):
    """The top `count` neurons of score [L, F] (largest first) as flat indices."""
    flat = np.asarray(score, np.float64).reshape(-1); count = min(count, flat.size)
    top = np.argpartition(-flat, count - 1)[:count]
    return top[np.argsort(-flat[top], kind='stable')]


def random_same_layer(order, width, depth, seed):
    """For each neuron of `order` (flat over [L, F]), a random unused neuron of the same layer."""
    rng = np.random.default_rng(seed); perms = {}; pos = {}; out = np.empty(len(order), np.int64)
    for i, u in enumerate(np.asarray(order, np.int64)):
        layer = int(u // width)
        if layer not in perms:
            perms[layer] = rng.permutation(width); pos[layer] = 0
        out[i] = layer * width + perms[layer][pos[layer]]; pos[layer] += 1
    return out


def rank_array(order, depth, width):
    """int32 [L, F]: the rank of each neuron in `order`, BIG elsewhere."""
    r = np.full(depth * width, BIG, np.int64); r[np.asarray(order, np.int64)] = np.arange(len(order))
    return r.reshape(depth, width).astype(np.int32)


# ---------------------------------------------------------------- nnx methods installed on openpi's Pi0
def _denoise_scan(self, observation, k, v, masks, count, noise, num_steps):
    """patching._denoise with lax.scan (reverse-differentiable); the same Euler steps."""
    from openpi.models.pi0 import make_attn_mask
    n = noise.shape[0]; depth = k.shape[0]; dt = -1.0 / num_steps
    k = jnp.broadcast_to(k, (depth, n) + k.shape[2:]); v = jnp.broadcast_to(v, (depth, n) + v.shape[2:])
    masks = jnp.broadcast_to(masks, (depth, n, masks.shape[-1])); count = jnp.broadcast_to(count, (n,))

    def step(carry, _):
        x_t, time = carry
        tokens, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(observation, x_t, jnp.broadcast_to(time, n))
        t = tokens.shape[1]
        suffix_attn = jnp.broadcast_to(make_attn_mask(suffix_mask, suffix_ar)[None], (depth, n, t, t))
        prefix_attn = jnp.broadcast_to(masks[:, :, None, :], (depth, n, t, masks.shape[-1]))
        positions = count[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        out = self.PaliGemma.llm(tokens, positions=positions, masks=jnp.concatenate([prefix_attn, suffix_attn], axis=-1),
                                 cache_k=k, cache_v=v, adarms_cond=adarms_cond, method='mech_suffix')
        return (x_t + dt * self.action_out_proj(out[:, -self.action_horizon:]), time + dt), None
    (x_0, _), _ = jax.lax.scan(step, (noise, jnp.float32(1.0)), None, length=num_steps)
    return x_0


def _acts_scan(self, obs, c, noise, num_steps):
    depth = c['k'].shape[0]
    return _denoise_scan(self, obs, c['k'], c['v'], jnp.broadcast_to(c['mask'][None], (depth,) + c['mask'].shape), c['count'], noise, num_steps)


def _mech_unit_scores(self, obs_a, obs_b, noise, w, window, num_steps=10):
    """Attribution at one noise [1, H, D]: top K unit indices and values [3, K] (act, nec, kv), per-neuron sums [3, L, F],
    per-layer sums [3, L], per-row sums of the act score [L, n], totals [3, 2] (sum, sum of the positive part), and checks
    (largest key/value gap of the hooked prefix vs openpi's own, largest gap of the scan vs the while-loop denoiser, E of
    the hooked chunks a and b under the scan loop, the number of valid image rows)."""
    obs_a, obs_b, pa, pb, ca, cb = N._pair(self, obs_a, obs_b)
    a = jax.lax.stop_gradient(_acts_scan(self, obs_a, ca, noise, num_steps))
    b = jax.lax.stop_gradient(_acts_scan(self, obs_b, cb, noise, num_steps))
    a_while = N._actions(self, obs_a, ca, noise, num_steps)
    native = P._cache(self, obs_a)
    hook_gap = jnp.maximum(jnp.abs(native['k'].astype(F32) - ca['k'].astype(F32)).max(), jnp.abs(native['v'].astype(F32) - ca['v'].astype(F32)).max())
    zero = jnp.zeros(ca['h'].shape, F32)
    g_act = jax.grad(lambda e: effect_w(_acts_scan(self, obs_a, N._run(self, pa, eps=e), noise, num_steps), a, b, w).sum())(zero)
    g_nec = jax.grad(lambda e: effect_w(_acts_scan(self, obs_b, N._run(self, pb, eps=e), noise, num_steps), a, b, w).sum())(zero)
    base = jax.lax.stop_gradient(N._window_kv(ca, pa, window))
    g = jax.lax.stop_gradient([y - x for x, y in zip(N._window_kv(ca, pa, window), N._window_kv(cb, pa, window))])
    g_kv = jax.grad(lambda e: N._rho(N._window_kv(N._run(self, pa, eps=e), pa, window), base, g))(zero)
    diff = jax.lax.stop_gradient(N._diff(ca, cb, pa))[:, 0]                 # [L, n, F], zero off the valid image rows
    scores = [diff * g_act[:, 0], -diff * g_nec[:, 0], diff * g_kv[:, 0]]   # nec: effect on E of setting B's unit to A's
    depth, n, width = diff.shape; k = min(K_MAX, depth * n * width)
    tops = [top_units(scores[0], k), top_units(-scores[1], k), top_units(scores[2], k)]
    idx = jnp.stack([t[0] for t in tops]); vals = jnp.stack([tops[0][1], -tops[1][1], tops[2][1]])
    neuron = jnp.stack([s.sum(1) for s in scores])
    layers = jnp.stack([s.sum((1, 2)) for s in scores])
    totals = jnp.stack([jnp.stack([s.sum(), jnp.maximum(s, 0).sum()]) for s in scores])
    checks = jnp.stack([hook_gap, jnp.abs(a_while - a).max(), effect_w(a, a, b, w).sum(), effect_w(b, a, b, w).sum(),
                        pa['rows'].sum().astype(F32)])
    return idx, vals, neuron, layers, scores[0].sum(-1), totals, checks


def _mech_unit_eval(self, obs_a, obs_b, noise, idx, neuron_rank, kind, src, k, direction, num_steps=10):
    """Actions for each spec (kind, src, k, direction [m]) at the noises [n, H, D]: model-space [2 + m, n, H, D] (A, B,
    specs) and the number of units patched per spec [m]. UNIT takes the first k entries of idx[src] (flat over [L, n_img, F]);
    NEURON the neurons with neuron_rank[src] < k on every valid image row; ALL every valid image-row unit; NONE nothing.
    SUFF patches A's run with B's values, NEC B's run with A's values."""
    obs_a, obs_b, pa, pb, ca, cb = N._pair(self, obs_a, obs_b)
    depth, _, n, width = ca['h'].shape; rows = pa['rows'][0]
    ranks = jnp.arange(idx.shape[1])

    def one(spec):
        kd, s, kk, dr = spec
        unit = jnp.zeros(depth * n * width, bool).at[idx[s]].set(ranks < kk).reshape(depth, n, width)
        neur = jnp.broadcast_to((neuron_rank[s] < kk)[:, None, :], (depth, n, width))
        sel = jnp.where(kd == UNIT, unit, jnp.where(kd == NEURON, neur, kd == ALL)) & rows[None, :, None]
        acts = jax.lax.cond(dr == SUFF, lambda _: N._actions(self, obs_a, N._run(self, pa, sel, cb['h']), noise, num_steps),
                            lambda _: N._actions(self, obs_b, N._run(self, pb, sel, ca['h']), noise, num_steps), None)
        return acts, sel.sum()
    acts, count = jax.lax.map(one, (kind, src, k, direction))
    first = jnp.stack([N._actions(self, obs_a, ca, noise, num_steps), N._actions(self, obs_b, cb, noise, num_steps)])
    return jnp.concatenate([first, acts]), count


def _mech_unit_eps(self, obs_a, obs_b, noise, w, idx, t, num_steps=10):
    """E (scan denoiser) of A's run with the units idx [m] (flat over [L, n_img, F]) offset by t * (h_B - h_A): the
    finite-difference check of the sufficiency attribution (E / t -> the sum of those units' scores as t -> 0)."""
    obs_a, obs_b, pa, pb, ca, cb = N._pair(self, obs_a, obs_b)
    a = _acts_scan(self, obs_a, ca, noise, num_steps); b = _acts_scan(self, obs_b, cb, noise, num_steps)
    diff = N._diff(ca, cb, pa); depth, _, n, width = diff.shape
    sel = jnp.zeros(depth * n * width, F32).at[idx].set(1.).reshape(depth, 1, n, width)
    return effect_w(_acts_scan(self, obs_a, N._run(self, pa, eps=t * diff * sel), noise, num_steps), a, b, w).sum()


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    N.install()
    from openpi.models import pi0
    pi0.Pi0.mech_unit_scores = _mech_unit_scores
    pi0.Pi0.mech_unit_eval = _mech_unit_eval
    pi0.Pi0.mech_unit_eps = _mech_unit_eps
    _INSTALLED = True


# ---------------------------------------------------------------- host wrapper
class UnitPatcher(N.NeuronPatcher):
    """neurons.NeuronPatcher plus the unit programs; all take unbatched policy-input dicts."""

    def __init__(self, policy):
        from openpi.shared import nnx_utils
        install()
        super().__init__(policy)
        self._uscores = nnx_utils.module_jit(policy._model.mech_unit_scores)
        self._ueval = nnx_utils.module_jit(policy._model.mech_unit_eval)
        self._ueps = nnx_utils.module_jit(policy._model.mech_unit_eps)

    def action_scale(self, data):
        """Per-dimension scale [3] of the output transform on the translation dimensions (it is affine per dimension), and the
        largest deviation from affinity over the 7 environment dimensions."""
        _, t = self.observation(data)
        zero = np.zeros((1, self.horizon, self.dim), np.float32); base = self.actions(zero, t)[0]
        scale, dev = [], 0.
        for d in range(7):
            one, two = zero.copy(), zero.copy(); one[..., d] = 1; two[..., d] = 2
            e1, e2 = self.actions(one, t)[0] - base, self.actions(two, t)[0] - base
            dev = max(dev, float(np.abs(e2 - 2 * e1).max()), float(np.abs(np.delete(e1, d, axis=-1)).max()))
            if d < 3:
                scale.append(float(e1[0, d]))
        return np.asarray(scale, np.float32), dev

    def unit_scores(self, base, donor, seed, scale, window=N.WINDOW):
        (oa, _), (ob, _) = self.observation(base), self.observation(donor)
        out = self._uscores(oa, ob, P.policy_noise([seed], self.horizon, self.dim), jnp.asarray(scale), self.window(window))
        names = ('idx', 'vals', 'neuron', 'layers', 'rowmap', 'totals', 'checks')
        return {k: np.asarray(v) for k, v in zip(names, out)}

    def unit_eval(self, base, donor, seeds, idx, neuron_rank, spec_list):
        """Environment actions [2 + m, n, H, 7] and patched-unit counts [m]."""
        (oa, ta), (ob, _) = self.observation(base), self.observation(donor)
        cols = list(zip(*[(kd, s, k, d) for _, kd, s, k, d in spec_list]))
        raw, count = self._ueval(oa, ob, P.policy_noise(seeds, self.horizon, self.dim), jnp.asarray(idx, jnp.int32),
                                 jnp.asarray(neuron_rank, jnp.int32), *[jnp.asarray(c, jnp.int32) for c in cols])
        return self.actions(raw, ta), np.asarray(count)

    def unit_eps(self, base, donor, seed, scale, idx, t):
        (oa, _), (ob, _) = self.observation(base), self.observation(donor)
        return float(self._ueps(oa, ob, P.policy_noise([seed], self.horizon, self.dim), jnp.asarray(scale), jnp.asarray(idx, jnp.int32),
                                jnp.float32(t)))
