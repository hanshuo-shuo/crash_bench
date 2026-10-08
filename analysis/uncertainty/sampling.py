"""One prefix prefill, eight explicitly seeded flow samples, no rollout RNG mutation."""
import time
import numpy as np
from common import disagreement


def sample_shared_prefix(self, observation, noise, num_steps=10):
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_module
    from openpi.models.pi0 import make_attn_mask
    obs = model_module.preprocess_observation(None, observation, train=False)
    tokens, mask, ar = self.embed_prefix(obs)
    _, cache = self.PaliGemma.llm([tokens, None], mask=make_attn_mask(mask, ar),
                                positions=jnp.cumsum(mask, axis=1)-1)
    count = noise.shape[0]
    # Scanned layers add axis 0; the cache's batch axis is axis 1.
    cache = jax.tree.map(lambda x: jnp.repeat(x, count, axis=1), cache)
    obs = jax.tree.map(lambda x: None if x is None else jnp.repeat(x, count, axis=0), obs)
    mask = jnp.repeat(mask, count, axis=0)
    dt = -1. / num_steps

    def step(carry):
        x, clock = carry
        suffix, smask, sar, cond = self.embed_suffix(obs, x, jnp.broadcast_to(clock, count))
        prefix_attention = jnp.broadcast_to(mask[:, None, :], (count, suffix.shape[1], mask.shape[1]))
        full_mask = jnp.concatenate([prefix_attention, make_attn_mask(smask, sar)], axis=-1)
        positions = jnp.sum(mask, axis=-1)[:, None] + jnp.cumsum(smask, axis=-1)-1
        (_, out), _ = self.PaliGemma.llm([None, suffix], mask=full_mask, positions=positions,
                                       kv_cache=cache, adarms_cond=[None, cond])
        return x + dt*self.action_out_proj(out[:, -self.action_horizon:]), clock+dt

    result, _ = jax.lax.while_loop(lambda carry: carry[1] >= -dt/2, step, (noise, 1.))
    return result


class Sampler:
    def __init__(self, policy, cfg, norm_file):
        import json
        import jax
        from openpi.shared import nnx_utils
        self.policy, self.cfg, self.jax = policy, cfg, jax
        policy._model.__class__.uncertainty_sample_shared_prefix = sample_shared_prefix
        self.batch = nnx_utils.module_jit(policy._model.uncertainty_sample_shared_prefix)
        stats = json.loads(norm_file.read_text())['norm_stats']['actions']
        self.scales = np.asarray(stats['std'], dtype=float)
        if self.scales.shape != (7,) or not np.isfinite(self.scales).all() or np.any(self.scales <= 0):
            raise RuntimeError('Invalid checkpoint action scales')

    def observation(self, data):
        import jax.numpy as jnp
        from openpi.models import model as model_module
        # This exactly duplicates native infer preprocessing, once. No noise is an input transform.
        transformed = self.policy._input_transform(self.jax.tree.map(lambda x: x, data))
        transformed = self.jax.tree.map(lambda x: jnp.asarray(x)[None, ...], transformed)
        return model_module.Observation.from_dict(transformed)

    def outputs(self, actions, observation):
        # Each sample follows all native output transforms, including one unnormalization.
        return np.stack([self.policy._output_transform({'state': np.asarray(observation.state[0]),
                                                       'actions': np.asarray(a)})['actions'] for a in actions])

    def infer(self, data, seeds, validate=False):
        import jax.numpy as jnp
        obs = self.observation(data)
        noise = jnp.stack([self.jax.random.normal(self.jax.random.key(int(s)), (10, 32)) for s in seeds])
        before = np.asarray(self.jax.random.key_data(self.policy._rng)).copy()
        start = time.monotonic()
        raw = np.asarray(self.batch(obs, noise, num_steps=10))
        values = self.outputs(raw, obs)
        seconds = time.monotonic()-start
        evidence = {'sample_seeds': seeds, 'samples': len(seeds), 'prefix_prefills': 1,
                    'batch_seconds': seconds, 'output_shape': list(values.shape),
                    'normalization_scales': self.scales.tolist()}
        if values.shape != (8, 10, 7) or not np.isfinite(values).all():
            raise RuntimeError('Invalid batched actions')
        if validate:
            start = time.monotonic()
            serial_raw = np.stack([np.asarray(self.policy._sample_actions(self.jax.random.key(int(s)), obs,
                                    noise=noise[i:i+1], num_steps=10))[0] for i, s in enumerate(seeds)])
            serial = self.outputs(serial_raw, obs)
            evidence.update(serial_seconds=time.monotonic()-start,
                            serial_batch_max_abs=float(np.max(np.abs(values-serial))),
                            serial_batch_internal_max_abs=float(np.max(np.abs(raw-serial_raw))),
                            sampling_atol=self.cfg['sampling_atol'], sampling_rtol=self.cfg['sampling_rtol'],
                            serial_batch_passed=bool(np.allclose(values, serial, atol=self.cfg['sampling_atol'],
                                                                rtol=self.cfg['sampling_rtol'])))
            # Return complete failed evidence; the caller persists it before stopping.
            evidence['serial_actions'] = serial.tolist()
        after = np.asarray(self.jax.random.key_data(self.policy._rng))
        if not np.array_equal(before, after):
            raise RuntimeError('Diagnostic sampling changed rollout RNG')
        evidence['rng_unchanged'] = True
        return values, disagreement(values.tolist(), self.scales.tolist()), evidence
