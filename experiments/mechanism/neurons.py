"""P4 neuron tools on the native openpi pi0.5 JAX model (docs/generative_verifier/HIL_MECHANISM_NEURONS.md).

The prefix (VLM) pass is run layer by layer with flax method interceptors on the first expert's attention and GeGLU MLP.
On the image rows (the valid image tokens; text tokens are never touched) they can
  - read the MLP hidden units h = gelu(x W_gate) * (x W_up) and the attention output of every layer,
  - set chosen hidden units (mlp_sel [L, F], or [L, n_img, F] per image row) to given values, add a float32 offset to every hidden unit (for gradients),
  - set the attention output of chosen layers (attn_sel [L]) to given values.
The action expert then reads the resulting per-layer keys/values exactly as in patching.py.

Quantities (section 2 of the protocol): the interface difference G is the B - A difference of the image-row keys and values at
the action expert's window layers (12-14); rho(X) = <KV_X - KV_A, G> / |G|^2 for a run X of prompt A with some image-row
values taken from prompt B's run. Clamping every MLP hidden unit and every attention output of the layers below the window's
top to B's values makes the image-row residual B's exactly, hence rho = 1 (the identity of the tool check).
"""
import jax
import jax.numpy as jnp
import numpy as np

import patching as P

WINDOW = (12, 13, 14)
F32 = jnp.float32
_INSTALLED = False


def hook_layers(window):
    """Layers whose MLP / attention outputs can change the keys/values of `window`: 0 .. max(window) - 1."""
    return tuple(range(max(window)))


# ---------------------------------------------------------------- linen method installed on openpi's gemma.Module
def _mech_hooked_prefix(self, embedded, positions, mask, n_img, rows, mlp_sel, mlp_val, attn_sel, attn_val, eps):
    """The prefix pass with image-row hooks (module docstring). rows [b, n_img] are the valid image rows; mlp_sel [L, F] and
    mlp_val [L, b, n_img, F], attn_sel [L] and attn_val [L, b, n_img, D], eps [L, b, n_img, F] may each be None (no hook).
    Returns keys and values [L, b, s, kh, h] and the natural (pre-hook) hidden units [L, b, n_img, F] and attention outputs
    [L, b, n_img, D] of the image rows."""
    from flax import linen as nn
    from openpi.models import gemma
    from openpi.models import lora
    x = embedded.astype(self.embed_dtype)
    mask = jnp.asarray(mask)[:, None, :, :]
    block = P._block(self); nones = [None] * len(self.configs)
    pick = rows[:, :, None]

    def step(x, inputs):
        params, msel, mval, asel, aval, e = inputs
        store = {}

        def hook(next_fun, args, kwargs, context):
            m = context.module
            if context.method_name == '__call__' and isinstance(m, lora.FeedForward) and m.name == 'mlp':
                if m.lora_config is not None:
                    raise ValueError('LoRA feed-forward weights are not supported')
                z = args[0]; dt = z.dtype; w = m.w_gating
                h = nn.gelu(jnp.dot(z, w[0].astype(dt))) * jnp.dot(z, w[1].astype(dt))
                img = h[:, :n_img]; store['h'] = img; new = img
                if msel is not None:  # per layer: [F] (a neuron on every image row) or [n_img, F] (neuron x row)
                    sel = msel[None] if msel.ndim == 2 else msel[None, None, :]
                    new = jnp.where(sel & pick, mval.astype(dt), new)
                if e is not None:
                    new = (new.astype(F32) + jnp.where(pick, e, 0.)).astype(dt)
                h = jnp.concatenate([new, h[:, n_img:]], axis=1)
                return jnp.dot(h, m.w_linear.astype(dt))
            if context.method_name == '__call__' and isinstance(m, gemma.Attention):
                out, kv = next_fun(*args, **kwargs)
                o = out[0]; img = o[:, :n_img]; store['a'] = img; new = img
                if asel is not None:
                    new = jnp.where(asel & pick, aval.astype(o.dtype), new)
                return [jnp.concatenate([new, o[:, n_img:]], axis=1)] + list(out[1:]), kv
            return next_fun(*args, **kwargs)
        with nn.intercept_methods(hook):
            (y, _), (k, v) = block.apply({'params': params}, [x, None], None, positions, mask, nones, True)
        return y, (k, v, store['h'], store['a'])
    _, (k, v, h, a) = jax.lax.scan(step, x, (self.variables['params']['layers'], mlp_sel, mlp_val, attn_sel, attn_val, eps))
    return k, v, h, a


# ---------------------------------------------------------------- nnx helpers on openpi's Pi0
def _prefix(self, obs):
    from openpi.models.pi0 import make_attn_mask
    tokens, valid, ar = self.embed_prefix(obs)
    n = tokens.shape[1] - obs.tokenized_prompt.shape[1]
    return dict(tokens=tokens, valid=valid, attn=make_attn_mask(valid, ar), pos=jnp.cumsum(valid, axis=1) - 1, n=n,
                rows=valid[:, :n], count=valid.sum(-1))


def _run(self, p, mlp_sel=None, mlp_val=None, attn_sel=None, attn_val=None, eps=None):
    k, v, h, a = self.PaliGemma.llm(p['tokens'], positions=p['pos'], mask=p['attn'], n_img=p['n'], rows=p['rows'],
                                    mlp_sel=mlp_sel, mlp_val=mlp_val, attn_sel=attn_sel, attn_val=attn_val, eps=eps,
                                    method='mech_hooked_prefix')
    return dict(k=k, v=v, h=h, a=a, mask=p['valid'], count=p['count'])


def _window_kv(c, p, window):
    """Image-row keys and values at the window layers, float32, zero elsewhere."""
    w = (window[:, None, None, None, None] & p['rows'][None, :, :, None, None]).astype(F32)
    return [c['k'][:, :, :p['n']].astype(F32) * w, c['v'][:, :, :p['n']].astype(F32) * w]


def _rho(x, base, g):
    num = sum(((xi - bi) * gi).sum() for xi, bi, gi in zip(x, base, g)); den = sum((gi ** 2).sum() for gi in g)
    return num / jnp.maximum(den, 1e-30)


def _actions(self, obs, c, noise, num_steps):
    depth = c['k'].shape[0]
    return P._denoise(self, obs, c['k'], c['v'], jnp.broadcast_to(c['mask'][None], (depth,) + c['mask'].shape), c['count'], noise, num_steps)


def _pair(self, obs_a, obs_b):
    from openpi.models import model as model_module
    obs_a = model_module.preprocess_observation(None, obs_a, train=False)
    obs_b = model_module.preprocess_observation(None, obs_b, train=False)
    pa, pb = _prefix(self, obs_a), _prefix(self, obs_b)
    return obs_a, obs_b, pa, pb, _run(self, pa), _run(self, pb)


def _diff(ca, cb, p):
    """h_B - h_A on the valid image rows, float32 [L, b, n, F]."""
    return (cb['h'].astype(F32) - ca['h'].astype(F32)) * p['rows'][None, :, :, None]


def _mech_neuron_patch(self, obs_a, obs_b, noise, mlp_sel, attn_sel, norm_sel, window, num_steps=10):
    """Patch-in specs on prompt A's run from prompt B's run (mlp_sel [m, L, F], attn_sel [m, L]): rho [m] and model-space
    actions [2 + m, n, H, D] (A, B, then each spec). Also |G|, the norms D [q] of h_B - h_A over norm_sel [q, L, F], and the
    largest difference between the hooked prefix (nothing selected) and openpi's own prefix pass (keys, values)."""
    obs_a, obs_b, pa, pb, ca, cb = _pair(self, obs_a, obs_b)
    base, g = _window_kv(ca, pa, window), [y - x for x, y in zip(_window_kv(ca, pa, window), _window_kv(cb, pa, window))]
    native = P._cache(self, obs_a)
    hook_gap = jnp.maximum(jnp.abs(native['k'].astype(F32) - ca['k'].astype(F32)).max(), jnp.abs(native['v'].astype(F32) - ca['v'].astype(F32)).max())
    diff = _diff(ca, cb, pa)
    norms = jnp.sqrt(jnp.einsum('qlf,lbnf->q', norm_sel.astype(F32), diff ** 2))

    def one(spec):
        c = _run(self, pa, spec[0], cb['h'], spec[1], cb['a'])
        return _rho(_window_kv(c, pa, window), base, g), _actions(self, obs_a, c, noise, num_steps)
    rho, acts = jax.lax.map(one, (mlp_sel, attn_sel))
    first = jnp.stack([_actions(self, obs_a, ca, noise, num_steps), _actions(self, obs_b, cb, noise, num_steps)])
    g_norm = jnp.sqrt(sum((gi ** 2).sum() for gi in g))
    return rho, jnp.concatenate([first, acts]), g_norm, norms, hook_gap


def _mech_neuron_rho(self, obs_a, obs_b, mlp_sel, attn_sel, window):
    """rho [m] only (no actions), for the many cheap specs."""
    obs_a, obs_b, pa, pb, ca, cb = _pair(self, obs_a, obs_b)
    base, g = _window_kv(ca, pa, window), [y - x for x, y in zip(_window_kv(ca, pa, window), _window_kv(cb, pa, window))]
    return jax.lax.map(lambda spec: _rho(_window_kv(_run(self, pa, spec[0], cb['h'], spec[1], cb['a']), pa, window), base, g),
                       (mlp_sel, attn_sel))


def _mech_neuron_atp(self, obs_a, obs_b, window):
    """Attribution patching at prompt A's run: score [L, F] = sum over image rows of (h_B - h_A) * d rho / d h, and the
    linear prediction of rho when every hidden unit is patched (the sum of the scores)."""
    obs_a, obs_b, pa, pb, ca, cb = _pair(self, obs_a, obs_b)
    base = jax.lax.stop_gradient(_window_kv(ca, pa, window))
    g = jax.lax.stop_gradient([y - x for x, y in zip(_window_kv(ca, pa, window), _window_kv(cb, pa, window))])
    zero = jnp.zeros(ca['h'].shape, F32)
    grad = jax.grad(lambda e: _rho(_window_kv(_run(self, pa, eps=e), pa, window), base, g))(zero)
    score = (jax.lax.stop_gradient(_diff(ca, cb, pa)) * grad).sum(axis=(1, 2))
    return score, score.sum()


def _mech_neuron_rho_eps(self, obs_a, obs_b, scale, sel, window):
    """rho when every selected hidden unit of prompt A's run is offset by scale * (h_B - h_A): the finite-difference check of
    the attribution (d rho / d scale at 0 equals the sum of the scores over sel)."""
    obs_a, obs_b, pa, pb, ca, cb = _pair(self, obs_a, obs_b)
    base, g = _window_kv(ca, pa, window), [y - x for x, y in zip(_window_kv(ca, pa, window), _window_kv(cb, pa, window))]
    e = scale * _diff(ca, cb, pa) * sel[:, None, None, :]
    return _rho(_window_kv(_run(self, pa, eps=e), pa, window), base, g)


def _mech_neuron_profile(self, obs, index):
    """Activation of chosen hidden units (index [U] flat layer * F + unit) on the valid image rows of one prompt's run:
    stats [U, 3] (mean, max, argmax row) and the per-row map [U, n] (zero on invalid rows)."""
    from openpi.models import model as model_module
    obs = model_module.preprocess_observation(None, obs, train=False)
    p = _prefix(self, obs); c = _run(self, p)
    depth, b, n, f = c['h'].shape
    h = jnp.moveaxis(c['h'][:, 0].astype(F32), -1, 1).reshape(depth * f, n)[index]
    valid = p['rows'][0]
    mean = (h * valid).sum(-1) / jnp.maximum(valid.sum(), 1)
    masked = jnp.where(valid[None], h, -jnp.inf)
    return jnp.stack([mean, masked.max(-1), masked.argmax(-1).astype(F32)], -1), h * valid


RANDOM_DIRECTION = 1


def neuron_delta(diff, ref_sel, apply_sel, scale, random_dir, key, rows):
    """The closed-loop change of the hidden units (protocol 4.2), float32 [L, b, n, F]. Per layer, the target norm is
    |scale| * |diff on ref_sel|. Own direction: the change is diff on apply_sel, signed like scale, rescaled to the target
    (exactly scale * diff when apply_sel == ref_sel). Random direction: a Gaussian on apply_sel with the target norm.
    Returns (delta, stats [L, 2]: target norm, delta norm)."""
    norm = lambda y: jnp.sqrt((y ** 2).sum(axis=(1, 2, 3)))
    target = jnp.abs(scale) * norm(diff * ref_sel[:, None, None, :])
    where = (apply_sel[:, None, None, :] & rows[None, :, :, None]).astype(F32)
    own = diff * where
    noise = jax.random.normal(key, diff.shape, F32) * where
    raw = jnp.where(random_dir == RANDOM_DIRECTION, noise, jnp.sign(scale) * own)
    size = norm(raw)
    delta = raw * jnp.where(size > 0, target / jnp.maximum(size, 1e-30), 0.)[:, None, None, None]
    return delta, jnp.stack([target, norm(delta)], -1)


def _mech_neuron_steer(self, obs, old_obs, noise, ref_sel, apply_sel, scale, random_dir, key, num_steps=10):
    """Closed-loop request for the running prompt `obs` (B): hidden units on apply_sel set to h_B + delta (neuron_delta, with
    h_A from the old prompt on the same observation); returns model-space actions [b, H, D] and stats [L, 3] (target norm,
    delta norm before the cast, norm of the change actually written after the cast)."""
    old_obs, obs, pa, pb, ca, cb = _pair(self, old_obs, obs)
    diff = _diff(ca, cb, pb)
    delta, stats = neuron_delta(diff, ref_sel, apply_sel, scale, random_dir, key, pb['rows'])
    value = (cb['h'].astype(F32) + delta).astype(cb['h'].dtype)
    written = jnp.sqrt((((value.astype(F32) - cb['h'].astype(F32)) * apply_sel[:, None, None, :]) ** 2).sum(axis=(1, 2, 3)))
    c = _run(self, pb, apply_sel, value)
    return _actions(self, obs, c, noise, num_steps), jnp.concatenate([stats, written[:, None]], -1)


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    P.install()
    from openpi.models import gemma
    from openpi.models import pi0
    gemma.Module.mech_hooked_prefix = _mech_hooked_prefix
    pi0.Pi0.mech_neuron_patch = _mech_neuron_patch
    pi0.Pi0.mech_neuron_rho = _mech_neuron_rho
    pi0.Pi0.mech_neuron_atp = _mech_neuron_atp
    pi0.Pi0.mech_neuron_profile = _mech_neuron_profile
    pi0.Pi0.mech_neuron_rho_eps = _mech_neuron_rho_eps
    pi0.Pi0.mech_neuron_steer = _mech_neuron_steer
    _INSTALLED = True


# ---------------------------------------------------------------- host wrapper
class NeuronPatcher(P.Patcher):
    """patching.Patcher plus the neuron programs; all take unbatched policy-input dicts."""

    def __init__(self, policy):
        from openpi.shared import nnx_utils
        install()
        super().__init__(policy)
        model = policy._model; cfg = model.PaliGemma.llm.module.configs[0]
        self.width, self.mlp = cfg.width, cfg.mlp_dim
        self._npatch = nnx_utils.module_jit(model.mech_neuron_patch)
        self._nrho = nnx_utils.module_jit(model.mech_neuron_rho)
        self._natp = nnx_utils.module_jit(model.mech_neuron_atp)
        self._nprofile = nnx_utils.module_jit(model.mech_neuron_profile)
        self._nsteer = nnx_utils.module_jit(model.mech_neuron_steer)
        self._nrho_eps = nnx_utils.module_jit(model.mech_neuron_rho_eps)

    def window(self, window=WINDOW):
        return jnp.asarray(np.isin(np.arange(self.depth), window))

    def neuron_patch(self, base, donor, seeds, mlp_sel, attn_sel, norm_sel, window=WINDOW):
        """rho [m], environment actions [2 + m, n, H, 7], |G|, D [q], hook gap."""
        (oa, ta), (ob, _) = self.observation(base), self.observation(donor)
        rho, raw, g, norms, gap = self._npatch(oa, ob, policy_noise(seeds, self.horizon, self.dim), jnp.asarray(mlp_sel), jnp.asarray(attn_sel),
                                               jnp.asarray(norm_sel), self.window(window))
        return np.asarray(rho), self.actions(raw, ta), float(g), np.asarray(norms), float(gap)

    def neuron_rho(self, base, donor, mlp_sel, attn_sel, window=WINDOW):
        (oa, _), (ob, _) = self.observation(base), self.observation(donor)
        return np.asarray(self._nrho(oa, ob, jnp.asarray(mlp_sel), jnp.asarray(attn_sel), self.window(window)))

    def neuron_atp(self, base, donor, window=WINDOW):
        (oa, _), (ob, _) = self.observation(base), self.observation(donor)
        score, total = self._natp(oa, ob, self.window(window))
        return np.asarray(score), float(total)

    def neuron_rho_eps(self, base, donor, scale, sel, window=WINDOW):
        (oa, _), (ob, _) = self.observation(base), self.observation(donor)
        return float(self._nrho_eps(oa, ob, jnp.float32(scale), jnp.asarray(sel), self.window(window)))

    def neuron_profile(self, data, index):
        o, _ = self.observation(data)
        stats, maps = self._nprofile(o, jnp.asarray(index, jnp.int32))
        return np.asarray(stats), np.asarray(maps)

    def neuron_steer(self, data, old, sample_key, ref_sel, apply_sel, scale, random_dir, direction_key):
        """One closed-loop request: environment actions [H, 7] and stats [L, 3]."""
        (ob, tb), (oa, _) = self.observation(data), self.observation(old)
        noise = jax.random.normal(sample_key, (1, self.horizon, self.dim))
        raw, stats = self._nsteer(ob, oa, noise, jnp.asarray(ref_sel), jnp.asarray(apply_sel), jnp.float32(scale), jnp.int32(random_dir),
                                  direction_key)
        return self.actions(raw[0], tb), np.asarray(stats)


policy_noise = P.policy_noise


# ---------------------------------------------------------------- pure helpers for spec masks
def mask_of(indices, depth, width):
    """Boolean [depth, width] from (layer, unit) pairs."""
    out = np.zeros((depth, width), bool)
    if len(indices):
        idx = np.asarray(indices, int); out[idx[:, 0], idx[:, 1]] = True
    return out


def layer_mask(depth, width, layers):
    out = np.zeros((depth, width), bool); out[list(layers)] = True
    return out
