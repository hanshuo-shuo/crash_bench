"""Unit tests for patching.py on a toy pi0.5 (openpi's dummy Gemma for both experts, SigLIP mu/14, float32, CPU).

Run in the openpi environment with the pinned openpi on PYTHONPATH:
  PYTHONPATH=$HOME/crash_bench/third_party/vlsa-aegis/openpi/src JAX_PLATFORMS=cpu \
    /projects/p33100/siosio/envs/openpi/bin/python -m unittest test_patching
"""
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patching as P
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'feasibility'))

TOKENS = 16


def toy():
    from openpi.models import pi0_config
    from openpi.models import siglip
    original = siglip.Module

    def small(**kwargs):
        kwargs['variant'] = 'mu/14'
        return original(**kwargs)
    config = pi0_config.Pi0Config(pi05=True, paligemma_variant='dummy', action_expert_variant='dummy', action_horizon=10,
                                  discrete_state_input=False, dtype='float32', max_token_len=TOKENS)
    with mock.patch.object(siglip, 'Module', small):
        model = config.create(jax.random.key(0))
    # pi0.5's adaRMS gates start at zero (adaLN-Zero), so a freshly initialized action expert ignores the prefix and every
    # prompt gives the same actions. Perturb every parameter so the toy's actions depend on the prompt.
    from flax import nnx
    params = nnx.state(model, nnx.Param)
    leaves, tree = jax.tree.flatten(params)
    keys = jax.random.split(jax.random.key(1), len(leaves))
    nnx.update(model, jax.tree.unflatten(tree, [x + .05 * jax.random.normal(k, x.shape, x.dtype) for x, k in zip(leaves, keys)]))
    return config, model


def data(config, ids, length):
    """An unbatched model-input dict (the format Policy.infer receives after its input transforms)."""
    obs = config.fake_obs()
    images = {k: np.asarray(jax.random.uniform(jax.random.key(i + 1), v.shape[1:], minval=-1, maxval=1))
              for i, (k, v) in enumerate(obs.images.items())}
    prompt = np.zeros(TOKENS, np.int32); prompt[:length] = ids[:length]
    return dict(image=images, image_mask={k: np.asarray(k != 'right_wrist_0_rgb') for k in images},
                state=np.zeros(config.action_dim, np.float32), tokenized_prompt=prompt,
                tokenized_prompt_mask=np.arange(TOKENS) < length)


class Patching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from openpi.policies import policy as policy_module
        cls.config, cls.model = toy()
        cls.policy = policy_module.Policy(cls.model)
        cls.patcher = P.Patcher(cls.policy)
        rng = np.random.default_rng(0)
        cls.a = data(cls.config, rng.integers(2, 1000, TOKENS), 8)
        cls.b8 = data(cls.config, rng.integers(2, 1000, TOKENS), 8)    # same count as A
        cls.b6 = data(cls.config, rng.integers(2, 1000, TOKENS), 6)    # two tokens fewer
        cls.b11 = data(cls.config, rng.integers(2, 1000, TOKENS), 11)  # three tokens more
        cls.seeds = [835866137, 7]
        cls.names, cls.layers, cls.groups = P.interface_grid(cls.patcher.depth)

    def interface(self, donor):
        return self.patcher.interface(self.a, donor, self.seeds, self.layers, self.groups)[0]

    def test_base_and_donor_match_openpis_policy_infer_with_the_reset_seed(self):
        out = self.interface(self.b6)
        for i, seed in enumerate(self.seeds):
            for row, d in ((0, self.a), (1, self.b6)):
                self.policy._rng = jax.random.key(seed)
                native = self.policy.infer(d)['actions']
                np.testing.assert_allclose(out[row, i], native, atol=2e-5, rtol=0)

    def test_noop_spec_is_the_base_and_full_spec_the_donor_bit_for_bit_at_equal_counts(self):
        out = self.interface(self.b8)
        np.testing.assert_array_equal(out[2], out[0]); np.testing.assert_array_equal(out[3], out[1])
        self.assertGreater(np.abs(out[1] - out[0]).max(), 1e-3)

    def test_full_spec_reproduces_the_donor_at_unequal_counts(self):
        for donor in (self.b6, self.b11):
            out = self.interface(donor)
            np.testing.assert_array_equal(out[2], out[0])
            np.testing.assert_allclose(out[3], out[1], atol=2e-5, rtol=0)
            self.assertTrue(np.isfinite(out).all())

    def test_without_rerotation_the_full_spec_misses_the_donor(self):
        """Negative control: the RoPE re-rotation is what makes unequal-count patches exact."""
        with mock.patch.object(P, 'rope_shift', lambda k, shift: k):
            patcher = P.Patcher(self.policy)
            out = patcher.interface(self.a, self.b6, self.seeds, self.layers, self.groups)[0]
        self.assertGreater(np.abs(out[3] - out[1]).max(), 1e-3)

    def test_partial_specs_lie_between_and_differ_from_identities(self):
        out = self.interface(self.b6)
        text_all = self.names.index('L%02d-%02d_text' % (0, 0)) + 2
        self.assertGreater(np.abs(out[text_all] - out[0]).max(), 0)

    def test_rope_shift_composes_and_is_exact_at_zero(self):
        from openpi.models import gemma
        k = jax.random.normal(jax.random.key(3), (2, 1, 5, 1, 16))
        pos = jnp.arange(5)[None] + 4
        rotated = gemma._apply_rope(k.reshape(2, 5, 1, 16), positions=jnp.broadcast_to(pos, (2, 5))).reshape(k.shape)
        moved = P.rope_shift(rotated, jnp.asarray([3]))
        expect = gemma._apply_rope(k.reshape(2, 5, 1, 16), positions=jnp.broadcast_to(pos + 3, (2, 5))).reshape(k.shape)
        np.testing.assert_allclose(moved, expect, atol=1e-5)
        np.testing.assert_array_equal(P.rope_shift(rotated, jnp.asarray([0])), rotated)

    def test_backbone_identities(self):
        names, layer, groups = P.backbone_grid(self.patcher.depth, text=True)
        out, counts = self.patcher.backbone(self.a, self.b8, self.seeds, layer, groups)
        self.assertEqual(counts, dict(count_base=8, count_donor=8))
        iface = self.interface(self.b8)
        np.testing.assert_allclose(out[0], iface[0], atol=2e-5, rtol=0)  # layer-by-layer prefix = native prefix
        np.testing.assert_allclose(out[1], iface[1], atol=2e-5, rtol=0)
        np.testing.assert_array_equal(out[2], out[0])                                  # none
        np.testing.assert_array_equal(out[2 + names.index('pre00_image')], out[0])     # images do not depend on the prompt
        np.testing.assert_array_equal(out[2 + names.index('pre00_text')], out[1])      # all inputs from the donor
        self.assertGreater(np.abs(out[2 + names.index('pre02_text')] - out[0]).max(), 0)

    def test_backbone_refuses_text_at_unequal_counts(self):
        names, layer, groups = P.backbone_grid(self.patcher.depth, text=True)
        with self.assertRaises(ValueError):
            self.patcher.backbone(self.a, self.b6, self.seeds, layer, groups)
        names, layer, groups = P.backbone_grid(self.patcher.depth, text=False)
        out, _ = self.patcher.backbone(self.a, self.b6, self.seeds, layer, groups)
        np.testing.assert_array_equal(out[2 + names.index('pre00_image')], out[0])

    def test_gaps_are_zero_for_the_same_prompt_and_positive_otherwise(self):
        same = self.patcher.gaps(self.a, self.a)
        self.assertEqual(same.shape, (self.patcher.depth, 2)); np.testing.assert_array_equal(same, 0)
        other = self.patcher.gaps(self.a, self.b6)
        self.assertTrue((other[:, 1] > 0).all()); self.assertTrue((other[1:, 0] > 0).all())

    def test_grids(self):
        names, layers, groups = P.interface_grid(18)
        self.assertEqual(len(names), 2 + (18 + 16) * 3); self.assertEqual(layers.shape, (len(names), 18))
        self.assertEqual(int(layers[names.index('L14-16_all')].sum()), 3)
        names, layer, groups = P.backbone_grid(18, text=False)
        self.assertEqual(len(names), 19); self.assertEqual(int(layer[0]), -1)


class Runner(unittest.TestCase):
    """run_open_loop.run_state and summarize end to end on the toy, behind a LIBERO-shaped input transform."""

    @classmethod
    def setUpClass(cls):
        from openpi.policies import policy as policy_module
        cls.config, model = toy()

        def to_model(d):
            words = d['prompt'].split()
            prompt = np.zeros(TOKENS, np.int32); prompt[:len(words)] = [(sum(map(ord, w)) % 900) + 2 for w in words]
            state = np.zeros(cls.config.action_dim, np.float32); state[:8] = d['observation/state']
            return dict(image={'base_0_rgb': d['observation/image'], 'left_wrist_0_rgb': d['observation/wrist_image'],
                               'right_wrist_0_rgb': np.zeros_like(d['observation/image'])},
                        image_mask={'base_0_rgb': np.True_, 'left_wrist_0_rgb': np.True_, 'right_wrist_0_rgb': np.False_},
                        state=state, tokenized_prompt=prompt, tokenized_prompt_mask=np.arange(TOKENS) < len(words))
        cls.policy = policy_module.Policy(model, transforms=[to_model], output_transforms=[lambda d: dict(d, actions=d['actions'][:, :7])])
        cls.patcher = P.Patcher(cls.policy)

    def state(self, folder, sid, prompt_a, prompt_b, phase, nudge):
        import run_open_loop as R
        from serve import digest
        rng = np.random.default_rng(len(sid))
        obs = {'observation/image': rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
               'observation/wrist_image': rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
               'observation/state': rng.normal(size=8)}
        twin = dict(obs); twin['observation/image'] = obs['observation/image'].copy(); twin['observation/image'][0, 0, 0] ^= nudge
        np.savez(folder/(sid + '_sw.npz'), **obs); np.savez(folder/(sid + '_a.npz'), **twin)
        seed = 1000 + len(sid)
        first = lambda d: dict(input_sha256=digest(d), actions=R.native(self.policy, d, seed).tolist(),
                               native_chunk_sha256=R.chunk_sha(R.native(self.policy, d, seed)))
        return dict(id=sid, pair='1->8' if prompt_a.endswith('stove') else '7->8', phase=phase, init=0, b_done=True,
                    twin_a_done=None, seed=seed, prompt_a=prompt_a, prompt_b=prompt_b,
                    input_npz=str(folder/(sid + '_sw.npz')), a_input_npz=str(folder/(sid + '_a.npz')),
                    a_first=first(dict(twin, prompt=prompt_a)), b_first=first(dict(obs, prompt=prompt_b)))

    def test_run_state_and_summarize(self):
        import tempfile
        import run_open_loop as R
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            states = [self.state(root, 's_equal', 'put the bowl on the stove', 'put the bowl on the plate', 0, 0),
                      self.state(root, 's_unequal_nudged', 'turn on the stove', 'put the bowl on the plate', 0, 1)]
            (root/'STATES.json').write_text(json.dumps(dict(states=states)))
            R.write(root/'PLAN.json', dict(shards=1, seeds=3, state_ids=[s['id'] for s in states], states_sha256=R.sha(root/'STATES.json')))
            rows = [R.run_state(self.patcher, self.policy, s, 3, root/'states'/s['id']) for s in states]
            for row in rows:
                self.assertTrue(row['input_b_exact']); self.assertTrue(row['native_vs_logged']['b_sha_equal'])
                self.assertEqual(row['noop_max_abs'], 0.0)
                self.assertTrue(all(abs(e - 1) < 1e-3 for e in row['full_e']))
                self.assertLess(row['mech_vs_native']['a_linf'], 1e-4); self.assertLess(row['mech_vs_native']['b_linf'], 1e-4)
                self.assertLess(row['backbone_base_linf'], 1e-4)
                self.assertEqual(len(row['gaps']['image']), self.patcher.depth)
            self.assertTrue(rows[0]['input_a_twin_exact']); self.assertEqual(rows[0]['a_twin_pixel_max_abs'], 0)
            self.assertTrue(rows[1]['input_a_twin_exact']); self.assertEqual(rows[1]['a_twin_pixel_max_abs'], 1)
            self.assertIn('pre01_text', rows[0]['backbone']); self.assertNotIn('pre01_text', rows[1]['backbone'])
            self.assertEqual(rows[1]['counts'], dict(count_base=4, count_donor=6))
            R.write(root/'SHARD_0_COMPLETE.json', dict(passed=True))
            summary = R.summarize(root)
            self.assertEqual(summary['states'], 2); self.assertIn('tool', summary)
            self.assertEqual(summary['native_vs_logged']['b_sha_equal'], 2)
            ((root/'states'/'s_equal'/'ROW.json')).write_text('{}')
            with self.assertRaises(RuntimeError):
                R.summarize(root)


if __name__ == '__main__':
    unittest.main()
