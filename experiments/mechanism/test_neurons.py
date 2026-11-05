"""Unit tests for neurons.py on the toy pi0.5 of test_patching (dummy Gemma, depth 4, 128 hidden units, float32, CPU).

  PYTHONPATH=$HOME/crash_bench/third_party/vlsa-aegis/openpi/src JAX_PLATFORMS=cpu \
    /projects/p33100/siosio/envs/openpi/bin/python -m unittest test_neurons
"""
import sys
import unittest
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import neurons as N
import patching as P
from test_patching import TOKENS, data, toy

WINDOW = (2, 3)            # toy window: hooks on layers 0-2


class Neurons(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from openpi.policies import policy as policy_module
        cls.config, cls.model = toy()
        cls.policy = policy_module.Policy(cls.model)
        cls.np = N.NeuronPatcher(cls.policy)
        rng = np.random.default_rng(0)
        cls.a = data(cls.config, rng.integers(2, 1000, TOKENS), 8)
        cls.b = data(cls.config, rng.integers(2, 1000, TOKENS), 6)
        cls.seeds = [835866137, 7]
        cls.L, cls.F = cls.np.depth, cls.np.mlp
        cls.hooks = N.hook_layers(WINDOW)

    def specs(self):
        none = np.zeros((self.L, self.F), bool); every = N.layer_mask(self.L, self.F, self.hooks)
        layers_none = np.zeros(self.L, bool); layers_all = np.isin(np.arange(self.L), self.hooks)
        one = N.mask_of([(1, 5)], self.L, self.F)
        mlp = np.stack([none, every, none, every, one]); attn = np.stack([layers_none, layers_none, layers_all, layers_all, layers_none])
        return mlp, attn

    def patch(self):
        mlp, attn = self.specs()
        return self.np.neuron_patch(self.a, self.b, self.seeds, mlp, attn, mlp, window=WINDOW)

    def test_hooks_without_selection_reproduce_the_native_prefix_and_policy_infer(self):
        rho, acts, g, norms, gap = self.patch()
        self.assertLess(gap, 1e-5)
        for i, seed in enumerate(self.seeds):
            for row, d in ((0, self.a), (1, self.b)):
                self.policy._rng = jax.random.key(seed)
                np.testing.assert_allclose(acts[row, i], self.policy.infer(d)['actions'], atol=2e-5, rtol=0)

    def test_noop_is_the_base_and_the_full_clamp_is_the_identity(self):
        rho, acts, g, norms, gap = self.patch()
        self.assertEqual(rho[0], 0.); np.testing.assert_array_equal(acts[2], acts[0])
        self.assertAlmostEqual(float(rho[3]), 1., places=5)          # every MLP hidden unit and attention output below the window
        # The MLP is token-wise, so attention is the only way text reaches the image rows: B's attention outputs alone already
        # make the image rows B's (the second identity), while B's MLP hidden units alone do not.
        self.assertAlmostEqual(float(rho[2]), 1., places=5)
        self.assertGreater(abs(rho[1] - 1), 1e-3)
        self.assertGreater(g, 0)
        self.assertNotEqual(rho[4], 0.)
        self.assertGreater(np.abs(acts[1] - acts[0]).max(), 1e-3)

    def test_norms_are_the_activation_differences(self):
        mlp, attn = self.specs()
        _, _, _, norms, _ = self.np.neuron_patch(self.a, self.b, self.seeds, mlp, attn, mlp, window=WINDOW)
        _, maps_a = self.np.neuron_profile(self.a, np.arange(self.L * self.F))
        _, maps_b = self.np.neuron_profile(dict(self.a, tokenized_prompt=self.b['tokenized_prompt'],
                                                tokenized_prompt_mask=self.b['tokenized_prompt_mask']), np.arange(self.L * self.F))
        d = (maps_b - maps_a).reshape(self.L, self.F, -1)
        np.testing.assert_allclose(norms[1], np.sqrt((d[list(self.hooks)] ** 2).sum()), rtol=1e-4)
        np.testing.assert_allclose(norms[4], np.sqrt((d[1, 5] ** 2).sum()), rtol=1e-4)
        self.assertEqual(norms[0], 0.)

    def test_rho_only_matches_the_patch_program(self):
        mlp, attn = self.specs()
        rho, *_ = self.np.neuron_patch(self.a, self.b, self.seeds, mlp, attn, mlp, window=WINDOW)
        np.testing.assert_allclose(self.np.neuron_rho(self.a, self.b, mlp, attn, window=WINDOW), rho, atol=1e-6)

    def test_attribution_is_the_derivative_of_rho(self):
        score, total = self.np.neuron_atp(self.a, self.b, window=WINDOW)
        self.assertEqual(score.shape, (self.L, self.F))
        self.assertTrue(np.all(score[max(WINDOW):] == 0))           # layers at or above the window top cannot move it
        t = 1e-3
        for sel in (N.layer_mask(self.L, self.F, self.hooks), N.mask_of([(0, 3), (1, 5), (2, 9)], self.L, self.F)):
            fd = (self.np.neuron_rho_eps(self.a, self.b, t, sel, WINDOW) - self.np.neuron_rho_eps(self.a, self.b, -t, sel, WINDOW)) / (2 * t)
            np.testing.assert_allclose(fd, score[sel].sum(), rtol=2e-2, atol=1e-4)
        np.testing.assert_allclose(total, score.sum(), rtol=1e-5)

    def test_profile_stats_agree_with_the_maps(self):
        index = np.asarray([0 * self.F + 3, 2 * self.F + 100])
        stats, maps = self.np.neuron_profile(self.a, index)
        self.assertEqual(maps.shape[0], 2)
        n_valid = 2 * maps.shape[1] // 3       # two of the three cameras are valid
        np.testing.assert_allclose(stats[:, 0], maps.sum(1) / n_valid, rtol=1e-5)
        np.testing.assert_allclose(stats[:, 1], maps[:, :n_valid].max(1), rtol=1e-6)
        self.assertTrue(np.all(maps[:, n_valid:] == 0))

    def steer(self, ref, apply, scale, random_dir=0):
        key = jax.random.key(11)
        return self.np.neuron_steer(self.b, self.a, key, ref, apply, scale, random_dir, jax.random.key(5))

    def test_steer_at_scale_zero_is_policy_infer(self):
        s = N.mask_of([(0, 3), (1, 5)], self.L, self.F)
        acts, stats = self.steer(s, s, 0.)
        # the key/value steering program at scale 0 is openpi's sampler (test_patching); same sample key, same noise
        plain = self.np.steer(self.b, self.a, jax.random.key(11), np.zeros(self.L, bool), 0., 0, jax.random.key(0))[0]
        np.testing.assert_allclose(acts, plain, atol=2e-5, rtol=0)
        self.assertTrue(np.all(stats == 0))

    def test_steer_arms_hit_their_target_norms(self):
        s = N.mask_of([(0, 3), (1, 5), (1, 7), (2, 9)], self.L, self.F)
        r = N.mask_of([(0, 4), (1, 6), (1, 8), (2, 10)], self.L, self.F)
        base, _ = self.steer(s, s, 0.)
        for apply, scale, rd in ((s, 1., 0), (r, 1., 0), (s, 1., 1), (s, -1., 0), (r, -1., 0)):
            acts, stats = self.steer(s, apply, scale, rd)
            np.testing.assert_allclose(stats[:, 1], stats[:, 0], rtol=1e-5)
            np.testing.assert_allclose(stats[:, 2], stats[:, 0], rtol=1e-5)     # float32 toy: nothing lost in the cast
            self.assertTrue(np.all(stats[list(self.hooks), 0] > 0))
            self.assertGreater(np.abs(acts - base).max(), 0)
        amp, _ = self.steer(s, s, 1.); out, _ = self.steer(s, s, -1.)
        self.assertGreater(np.abs(amp - out).max(), 0)

    def test_neuron_delta_rules(self):
        rng = np.random.default_rng(1)
        diff = rng.standard_normal((3, 1, 4, 6)).astype(np.float32); rows = np.asarray([[True, True, True, False]])
        diff = diff * rows[None, :, :, None]
        s = np.zeros((3, 6), bool); s[0, 1] = s[1, 2] = s[1, 3] = True
        r = np.zeros((3, 6), bool); r[0, 4] = r[1, 0] = r[1, 5] = True
        d, st = N.neuron_delta(diff, s, s, 1., 0, jax.random.key(0), rows)
        np.testing.assert_allclose(d, diff * s[:, None, None, :], rtol=1e-6)
        d, st = N.neuron_delta(diff, s, s, -1., 0, jax.random.key(0), rows)
        np.testing.assert_allclose(d, -diff * s[:, None, None, :], rtol=1e-6)
        d, st = N.neuron_delta(diff, s, r, 1., 0, jax.random.key(0), rows)
        self.assertTrue(np.all(np.asarray(d)[~np.broadcast_to(r[:, None, None, :], d.shape)] == 0))
        np.testing.assert_allclose(st[:, 1], st[:, 0], rtol=1e-6)
        self.assertEqual(float(st[2, 0]), 0.)
        d, st = N.neuron_delta(diff, s, s, 1., 1, jax.random.key(0), rows)
        np.testing.assert_allclose(st[:, 1], st[:, 0], rtol=1e-5)
        self.assertTrue(np.all(np.asarray(d)[:, :, 3] == 0))       # never on invalid rows

    def test_masks(self):
        m = N.mask_of([(1, 2), (3, 0)], 4, 5)
        self.assertEqual(m.sum(), 2); self.assertTrue(m[1, 2] and m[3, 0])
        self.assertEqual(N.hook_layers((12, 13, 14)), tuple(range(14)))
        self.assertEqual(N.layer_mask(4, 5, (0, 2)).sum(), 10)


if __name__ == '__main__':
    unittest.main()
