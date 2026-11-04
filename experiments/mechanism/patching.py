"""Activation patching on the native openpi pi0.5 JAX model (P1 of the mechanism pilot, docs/generative_verifier/HIL_MECHANISM_IDEA.md).

Both interventions act on the conditioning the action expert reads. Base and donor share the observation and differ in the
prompt only.
  interface: at chosen action-expert layers, the expert reads the donor's prefix keys/values for chosen token groups
             (the "action-conditioning interface" of Goodfire's 2610.20750);
  backbone:  the input residual of one prefix (VLM) layer is replaced by the donor's on chosen token groups and the prefix is
             recomputed from that layer on, so the keys/values of that layer and the later ones change.
Prompts of different token counts are handled for the interface. A patched group is read with the donor's mask, and the
donor's keys are re-rotated by the count difference. Every key thus keeps the position it had relative to the action tokens
in the donor run (RoPE logits depend on relative position only). Backbone patching of text tokens needs equal counts;
image tokens are always aligned.

All runs of one request (base, donor, each spec) go through the same traced program. A no-op spec therefore reproduces the
base bit for bit, and a full interface spec reproduces the donor (bit for bit at equal counts). Fidelity to openpi's own
sampler is checked separately: test_patching.py on a toy model, and the P1 GPU check against the stored native chunks.
"""
import jax
import jax.numpy as jnp
import numpy as np

GROUPS = ('image', 'text')
_INSTALLED = False


# ---------------------------------------------------------------- linen methods installed on openpi's gemma.Module
def _block(self):
    from openpi.models import gemma
    return gemma.Block(configs=tuple(self.configs), dropout=self.dropout, dropout_bdims=self.dropout_bdims, parent=None)


def _mech_prefix(self, embedded, positions, mask, patch_layer, donor_pre, rows):
    """The prefix (VLM) pass layer by layer. Before layer `patch_layer` (-1: none), the rows `rows` [b, s] of the residual
    are replaced by the donor's input residual of that layer (`donor_pre` [L, b, s, d]). Returns keys and values [L, b, s, k, h]
    and the input residual of every layer [L, b, s, d]."""
    x = embedded.astype(self.embed_dtype)
    mask = jnp.asarray(mask)[:, None, :, :]
    block = _block(self); depth = self.configs[0].depth; nones = [None] * len(self.configs)

    def step(x, inputs):
        index, params, donor = inputs
        x = jnp.where((index == patch_layer) & rows[..., None], donor.astype(x.dtype), x)
        (y, _), (k, v) = block.apply({'params': params}, [x, None], None, positions, mask, nones, True)
        return y, (x, k, v)
    _, (pre, k, v) = jax.lax.scan(step, x, (jnp.arange(depth), self.variables['params']['layers'], donor_pre))
    return k, v, pre


def _mech_suffix(self, embedded, positions, masks, cache_k, cache_v, adarms_cond):
    """The action-expert pass over the action tokens, reading per-layer prefix keys/values and per-layer attention masks
    (`masks` [L, b, t, s + t]); returns the final-normed action-expert output (openpi's Module.__call__ for [None, suffix])."""
    x = embedded.astype(self.embed_dtype)
    block = _block(self)

    def step(x, inputs):
        params, k, v, m = inputs
        (_, y), _ = block.apply({'params': params}, [None, x], (k, v), positions, m[:, None], [None, adarms_cond], True)
        return y, None
    x, _ = jax.lax.scan(step, x, (self.variables['params']['layers'], cache_k, cache_v, masks))
    return self.final_norms[1](x, adarms_cond)[0]


# ---------------------------------------------------------------- pure helpers
def rope_shift(k, shift):
    """Rotates RoPE keys k [L, b, s, kh, h] by `shift` [b] more positions (exactly k where shift == 0)."""
    from openpi.models import gemma
    layers, b, s = k.shape[:3]
    positions = jnp.broadcast_to(shift[None, :, None], (layers, b, s)).reshape(layers * b, s)
    rotated = gemma._apply_rope(k.reshape((layers * b,) + k.shape[2:]), positions=positions).reshape(k.shape)
    return jnp.where((shift == 0)[None, :, None, None, None], k, rotated)


def mix_interface(base, donor, layers, groups, image_slots):
    """Per-layer keys, values and prefix masks [L, b, s] the action expert reads when `layers` [L] read the donor on the
    token `groups` [2] (image, text). base/donor: dict(k, v [L, b, s, kh, h], mask [b, s], count [b])."""
    slots = jnp.where(image_slots, groups[0], groups[1])
    chosen = layers[:, None] & slots[None, :]
    donor_k = rope_shift(donor['k'], base['count'] - donor['count'])
    pick = chosen[:, None, :, None, None]
    k = jnp.where(pick, donor_k, base['k']); v = jnp.where(pick, donor['v'], base['v'])
    masks = jnp.where(chosen[:, None, :], donor['mask'][None], base['mask'][None])
    return k, v, masks


# ---------------------------------------------------------------- nnx methods installed on openpi's Pi0
def _cache(self, observation):
    """openpi's prefix pass (Pi0.sample_actions): native keys/values, the valid mask and the valid count."""
    from openpi.models.pi0 import make_attn_mask
    tokens, valid, ar = self.embed_prefix(observation)
    _, (k, v) = self.PaliGemma.llm([tokens, None], mask=make_attn_mask(valid, ar), positions=jnp.cumsum(valid, axis=1) - 1)
    return dict(k=k, v=v, mask=valid, count=valid.sum(-1))


def _image_slots(observation, length):
    return jnp.arange(length) < length - observation.tokenized_prompt.shape[1]


def _denoise(self, observation, k, v, masks, count, noise, num_steps):
    """openpi's Euler loop (Pi0.sample_actions) with per-layer prefix keys/values/masks broadcast to the noise batch."""
    from openpi.models.pi0 import make_attn_mask
    n = noise.shape[0]; depth = k.shape[0]; dt = -1.0 / num_steps
    k = jnp.broadcast_to(k, (depth, n) + k.shape[2:]); v = jnp.broadcast_to(v, (depth, n) + v.shape[2:])
    masks = jnp.broadcast_to(masks, (depth, n, masks.shape[-1])); count = jnp.broadcast_to(count, (n,))

    def step(carry):
        x_t, time = carry
        tokens, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(observation, x_t, jnp.broadcast_to(time, n))
        t = tokens.shape[1]
        suffix_attn = jnp.broadcast_to(make_attn_mask(suffix_mask, suffix_ar)[None], (depth, n, t, t))
        prefix_attn = jnp.broadcast_to(masks[:, :, None, :], (depth, n, t, masks.shape[-1]))
        positions = count[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        out = self.PaliGemma.llm(tokens, positions=positions, masks=jnp.concatenate([prefix_attn, suffix_attn], axis=-1),
                                 cache_k=k, cache_v=v, adarms_cond=adarms_cond, method='mech_suffix')
        return x_t + dt * self.action_out_proj(out[:, -self.action_horizon:]), time + dt

    x_0, _ = jax.lax.while_loop(lambda carry: carry[1] >= -dt / 2, step, (noise, 1.0))
    return x_0


def _full(c, depth):
    return c['k'], c['v'], jnp.broadcast_to(c['mask'][None], (depth,) + c['mask'].shape)


def _mech_interface(self, base_obs, donor_obs, noise, layers, groups, num_steps=10):
    """Model-space actions [2 + m, n, H, D]: base, donor, then each interface spec (layers [m, L] bool, groups [m, 2] bool)
    for the noises `noise` [n, H, D]."""
    from openpi.models import model as model_module
    base_obs = model_module.preprocess_observation(None, base_obs, train=False)
    donor_obs = model_module.preprocess_observation(None, donor_obs, train=False)
    b, d = _cache(self, base_obs), _cache(self, donor_obs)
    depth = b['k'].shape[0]; image = _image_slots(base_obs, b['mask'].shape[1])
    run = lambda k, v, masks, count: _denoise(self, base_obs, k, v, masks, count, noise, num_steps)

    def one(spec):
        k, v, masks = mix_interface(b, d, spec[0], spec[1], image)
        return run(k, v, masks, b['count'])
    return jnp.concatenate([jnp.stack([run(*_full(b, depth), b['count']), run(*_full(d, depth), d['count'])]),
                            jax.lax.map(one, (layers, groups))])


def _mech_backbone(self, base_obs, donor_obs, noise, layer, groups, num_steps=10):
    """Model-space actions [2 + m, n, H, D]: base and donor through the layer-by-layer prefix pass, then each backbone spec
    (layer [m] int: the prefix layer whose input residual is patched; groups [m, 2] bool). Text groups need equal counts
    (checked by the host wrapper)."""
    from openpi.models import model as model_module
    from openpi.models.pi0 import make_attn_mask
    base_obs = model_module.preprocess_observation(None, base_obs, train=False)
    donor_obs = model_module.preprocess_observation(None, donor_obs, train=False)

    def inputs(observation):
        tokens, valid, ar = self.embed_prefix(observation)
        return tokens, valid, make_attn_mask(valid, ar), jnp.cumsum(valid, axis=1) - 1
    tokens_b, valid_b, attn_b, pos_b = inputs(base_obs); tokens_d, valid_d, attn_d, pos_d = inputs(donor_obs)
    depth = self.PaliGemma.llm.module.configs[0].depth
    zeros = jnp.zeros((depth,) + tokens_b.shape, tokens_b.dtype); no_rows = jnp.zeros(valid_b.shape, bool)
    prefix = lambda t, p, a, layer, pre, rows: self.PaliGemma.llm(t, positions=p, mask=a, patch_layer=layer, donor_pre=pre,
                                                                  rows=rows, method='mech_prefix')
    k_d, v_d, pre_d = prefix(tokens_d, pos_d, attn_d, -1, zeros, no_rows)
    k_b, v_b, _ = prefix(tokens_b, pos_b, attn_b, -1, zeros, no_rows)
    image = _image_slots(base_obs, valid_b.shape[1])
    run = lambda k, v, valid: _denoise(self, base_obs, k, v, jnp.broadcast_to(valid[None], (depth,) + valid.shape),
                                       valid.sum(-1), noise, num_steps)

    def one(spec):
        rows = jnp.where(image, spec[1][0], spec[1][1])[None, :] & valid_b
        k, v, _ = prefix(tokens_b, pos_b, attn_b, spec[0], pre_d, rows)
        return run(k, v, valid_b)
    return jnp.concatenate([jnp.stack([run(k_b, v_b, valid_b), run(k_d, v_d, valid_d)]), jax.lax.map(one, (layer, groups))])


def _mech_gaps(self, base_obs, donor_obs):
    """How much the prompt changes what the action expert can read, per layer [L, 2] (no intervention): relative value-vector
    difference donor vs base on the image tokens (aligned slots) and on the mean value over each prompt's valid text tokens."""
    from openpi.models import model as model_module
    base_obs = model_module.preprocess_observation(None, base_obs, train=False)
    donor_obs = model_module.preprocess_observation(None, donor_obs, train=False)
    b, d = _cache(self, base_obs), _cache(self, donor_obs)
    image = _image_slots(base_obs, b['mask'].shape[1])
    vb, vd = b['v'].astype(jnp.float32), d['v'].astype(jnp.float32)

    def rel(x, y, weight):
        diff = ((x - y) ** 2 * weight).sum(axis=(1, 2, 3, 4)); norm = (y ** 2 * weight).sum(axis=(1, 2, 3, 4))
        return jnp.sqrt(diff / jnp.maximum(norm, 1e-30))
    w_image = (image[None, :] & b['mask'])[None, :, :, None, None].astype(jnp.float32)

    def text_mean(v, mask):
        w = (~image[None, :] & mask)[None, :, :, None, None].astype(jnp.float32)
        return (v * w).sum(axis=2, keepdims=True) / jnp.maximum(w.sum(axis=2, keepdims=True), 1)
    text = rel(text_mean(vd, d['mask']), text_mean(vb, b['mask']), jnp.ones((1, 1, 1, 1, 1)))
    return jnp.stack([rel(vd, vb, w_image), text], axis=-1)


def install():
    """Adds the mech_* methods to openpi's gemma.Module (linen) and Pi0 (nnx); idempotent, changes no native method."""
    global _INSTALLED
    if _INSTALLED:
        return
    from openpi.models import gemma
    from openpi.models import pi0
    gemma.Module.mech_prefix = _mech_prefix
    gemma.Module.mech_suffix = _mech_suffix
    pi0.Pi0.mech_interface = _mech_interface
    pi0.Pi0.mech_backbone = _mech_backbone
    pi0.Pi0.mech_gaps = _mech_gaps
    _INSTALLED = True


# ---------------------------------------------------------------- host wrapper around an openpi Policy
def policy_noise(seeds, horizon, dim):
    """The noise openpi's Policy.infer draws on its first request after the policy RNG is reset to `seed` (serve.py):
    Policy.infer splits the key once and Pi0.sample_actions draws normal(rng, (1, horizon, dim))."""
    return jnp.concatenate([jax.random.normal(jax.random.split(jax.random.key(int(s)))[1], (1, horizon, dim)) for s in seeds])


def interface_grid(depth, windows=(1, 3)):
    """Layer sets (single layers and sliding windows) crossed with token groups (all, image, text), plus the two identities
    (nothing, everything) first. Returns (names, layers [m, L] bool, groups [m, 2] bool)."""
    names = ['none', 'full']; layers = [np.zeros(depth, bool), np.ones(depth, bool)]; groups = [[False, False], [True, True]]
    for w in windows:
        for start in range(depth - w + 1):
            chosen = np.zeros(depth, bool); chosen[start:start + w] = True
            for g, name in (((True, True), 'all'), ((True, False), 'image'), ((False, True), 'text')):
                names.append('L%02d-%02d_%s' % (start, start + w - 1, name)); layers.append(chosen); groups.append(list(g))
    return names, np.asarray(layers), np.asarray(groups)


def backbone_grid(depth, text):
    """Input residual of each prefix layer, image group (and text group when the prompts have equal counts), plus the no-op."""
    names = ['none']; layer = [-1]; groups = [[False, False]]
    for l in range(depth):
        for g, name in (((True, False), 'image'),) + ((((False, True), 'text'),) if text else ()):
            names.append('pre%02d_%s' % (l, name)); layer.append(l); groups.append(list(g))
    return names, np.asarray(layer, np.int32), np.asarray(groups)


class Patcher:
    """Runs patching requests on a loaded openpi Policy without touching its RNG or native methods."""

    def __init__(self, policy):
        from openpi.shared import nnx_utils
        install()
        self.policy = policy; model = policy._model
        self.horizon, self.dim = model.action_horizon, model.action_dim
        self.depth = model.PaliGemma.llm.module.configs[0].depth
        self._interface = nnx_utils.module_jit(model.mech_interface)
        self._backbone = nnx_utils.module_jit(model.mech_backbone)
        self._gaps = nnx_utils.module_jit(model.mech_gaps)

    def gaps(self, base, donor):
        """Per-layer relative value difference donor vs base [L, 2] (image tokens, mean text token)."""
        (ob, _), (od, _) = self.observation(base), self.observation(donor)
        return np.asarray(self._gaps(ob, od))

    def observation(self, data):
        """openpi's input pipeline for one policy-input dict (Policy.infer): (Observation with batch 1, transformed dict)."""
        from openpi.models import model as model_module
        transformed = self.policy._input_transform(jax.tree.map(lambda x: x, data))
        batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], transformed)
        return model_module.Observation.from_dict(batched), transformed

    def actions(self, model_actions, transformed):
        """openpi's output pipeline for model-space chunks [..., H, D] -> environment actions [..., H, 7]."""
        flat = np.asarray(model_actions).reshape((-1,) + np.asarray(model_actions).shape[-2:])
        out = [np.asarray(self.policy._output_transform({'state': np.asarray(transformed['state']), 'actions': a})['actions'])
               for a in flat]
        return np.stack(out).reshape(np.asarray(model_actions).shape[:-2] + out[0].shape)

    def count(self, transformed):
        return int(np.asarray(transformed['tokenized_prompt_mask']).sum())

    def interface(self, base, donor, seeds, layers, groups):
        """Environment actions [2 + m, n, H, 7] for base, donor and each spec; plus the token counts."""
        (ob, tb), (od, td) = self.observation(base), self.observation(donor)
        raw = self._interface(ob, od, policy_noise(seeds, self.horizon, self.dim), jnp.asarray(layers), jnp.asarray(groups))
        return self.actions(raw, tb), dict(count_base=self.count(tb), count_donor=self.count(td))

    def backbone(self, base, donor, seeds, layer, groups):
        (ob, tb), (od, td) = self.observation(base), self.observation(donor)
        counts = dict(count_base=self.count(tb), count_donor=self.count(td))
        if np.asarray(groups)[:, 1].any() and counts['count_base'] != counts['count_donor']:
            raise ValueError('Backbone patching of text tokens needs equal token counts: %r' % counts)
        raw = self._backbone(ob, od, policy_noise(seeds, self.horizon, self.dim), jnp.asarray(layer), jnp.asarray(groups))
        return self.actions(raw, tb), counts
