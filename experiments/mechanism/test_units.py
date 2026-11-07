"""Unit tests for units.py and run_units.py: the unit programs on the toy pi0.5 of test_patching (the scan denoiser, the
attribution against finite differences, the per-row patch counts, the no-op), the pure set helpers, the readings and an
end-to-end summary on a synthetic root.

Run on a compute node (the toy programs need more memory than a login node allows), in the openpi environment:
  PYTHONPATH=$HOME/crash_bench/third_party/vlsa-aegis/openpi/src JAX_PLATFORMS=cpu \
    /projects/p33100/siosio/envs/openpi/bin/python -m unittest test_units
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'feasibility'))
import run_units as RU  # noqa: E402
import units as U  # noqa: E402

WINDOW = (1, 2)


class Toy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from openpi.policies import policy as policy_module
        import test_patching as T
        cls.config, model = T.toy()
        cls.policy = policy_module.Policy(model, transforms=[T.libero_like(cls.config)],
                                          output_transforms=[lambda d: dict(d, actions=d['actions'][:, :7])])
        cls.patcher = U.UnitPatcher(cls.policy)
        rng = np.random.default_rng(3)
        cls.obs = {'observation/image': rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
                   'observation/wrist_image': rng.integers(0, 255, (224, 224, 3), dtype=np.uint8), 'observation/state': rng.normal(size=8)}
        cls.a, cls.b = dict(cls.obs, prompt='turn on the stove'), dict(cls.obs, prompt='put the bowl on the plate')
        cls.scale, cls.affine = cls.patcher.action_scale(cls.a)
        cls.sc = cls.patcher.unit_scores(cls.a, cls.b, 77, cls.scale, WINDOW)

    def test_action_scale_is_affine(self):
        np.testing.assert_allclose(self.scale, 1.); self.assertEqual(self.affine, 0.)

    def test_scores_and_checks(self):
        sc = self.sc; depth, width = sc['neuron'].shape[1:]; n = sc['rowmap'].shape[1]
        self.assertEqual(sc['checks'][0], 0.)               # hooked prefix == openpi's prefix
        self.assertLess(sc['checks'][1], 1e-6)              # scan denoiser == while-loop denoiser
        self.assertLess(abs(sc['checks'][2]), 1e-5); self.assertLess(abs(sc['checks'][3] - 1), 1e-5)
        self.assertEqual(int(sc['checks'][4]), 512)         # two valid cameras of 256 patches
        self.assertEqual(sc['idx'].shape, (3, min(U.K_MAX, depth * n * width)))
        self.assertTrue(np.all(np.diff(sc['vals'][0]) <= 0)); self.assertTrue(np.all(np.diff(sc['vals'][1]) >= 0))
        self.assertGreater(sc['vals'][0][0], 0); self.assertLess(sc['vals'][1][0], 0); self.assertGreater(abs(sc['vals'][2][0]), 0)
        np.testing.assert_allclose(sc['layers'], sc['neuron'].sum(-1), rtol=1e-4, atol=1e-7)
        rows = (sc['idx'][0].astype(np.int64) // width) % n
        self.assertTrue(np.all(rows[:1000] < 512))          # the top units sit on valid image rows
        self.assertTrue(np.isfinite(sc['totals']).all())

    def test_attribution_matches_finite_difference(self):
        idx = self.sc['idx'][0][:50]; pred = float(np.sum(self.sc['vals'][0][:50], dtype=np.float64))
        fd = self.patcher.unit_eps(self.a, self.b, 77, self.scale, idx, 1e-3) / 1e-3
        self.assertAlmostEqual(fd / pred, 1., delta=.05)

    def spec_list(self):
        return [('none', U.NONE, 0, 0, U.SUFF), ('all_suff', U.ALL, 0, 0, U.SUFF), ('all_nec', U.ALL, 0, 0, U.NEC),
                ('u_act_7', U.UNIT, 0, 7, U.SUFF), ('u_actrand_7', U.UNIT, 1, 7, U.SUFF), ('u_kv_7', U.UNIT, 2, 7, U.SUFF),
                ('u_nec_7', U.UNIT, 3, 7, U.NEC), ('n_act_2', U.NEURON, 0, 2, U.SUFF), ('n_nec_3', U.NEURON, 3, 3, U.NEC)]

    def test_eval_counts_noop_and_directions(self):
        sc = self.sc; depth, width = sc['neuron'].shape[1:]; n = sc['rowmap'].shape[1]
        uidx = RU.unit_sources(sc, n, width, 'sw'); orders, nrank = RU.neuron_sources(sc, depth, width, 'sw')
        specs = self.spec_list()
        env, count = self.patcher.unit_eval(self.a, self.b, [77, 78], uidx, nrank, specs)
        self.assertEqual([int(c) for c in count], [RU.expected_count(s, depth, 512, width) for s in specs])
        np.testing.assert_array_equal(env[2], env[0])                           # nothing patched = A
        self.assertFalse(np.array_equal(env[3], env[0])); self.assertFalse(np.array_equal(env[4], env[1]))
        a = env[0]; b = env[1]
        e_act = RU.M.effect(env[5], a, b); self.assertGreater(e_act[0], 0)   # at the ranking seed the top units move A toward B

    def test_run_state(self):
        import run_open_loop as O
        from serve import digest
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            np.savez(folder/'sw.npz', **self.obs); np.savez(folder/'a.npz', **self.obs)
            seed = 1010
            first = lambda d: dict(input_sha256=digest(d), actions=O.native(self.policy, d, seed).tolist(),
                                   native_chunk_sha256=O.chunk_sha(O.native(self.policy, d, seed)))
            state = dict(id='s_f050', sw='s', pair='7->8', phase=.5, init=2, b_done=False, seed=seed, prompt_a=self.a['prompt'],
                         prompt_b=self.b['prompt'], input_npz=str(folder/'sw.npz'), a_input_npz=str(folder/'a.npz'),
                         a_first=first(self.a), b_first=first(self.b))
            old = RU.FD_K; RU.FD_K = 20
            try:
                row = RU.run_state(self.patcher, self.policy, state, folder/'out', self.scale, spec_list=self.spec_list())
            finally:
                RU.FD_K = old
            self.assertTrue(row['input_b_exact']); self.assertTrue(row['counts_ok'], row['count_mismatch'])
            self.assertEqual(row['noop_max_abs'], 0.); self.assertEqual(row['valid_rows'], 512); self.assertTrue(row['scores_finite'])
            self.assertLess(row['fidelity_ratio'], .1); self.assertEqual(len(row['e_seeds']['u_act_7']), RU.SEEDS)
            self.assertEqual(set(row['e']), {s[0] for s in self.spec_list()})
            hashes = json.loads((folder/'out'/'HASHES.json').read_text())
            self.assertEqual(set(hashes), {p.name for p in (folder/'out').iterdir()} - {'HASHES.json'})


# ---------------------------------------------------------------- pure
class Sets(unittest.TestCase):
    def test_random_same_position(self):
        n, width = 6, 50; rng = np.random.default_rng(0)
        idx = rng.choice(3 * n * width, 120, replace=False).astype(np.int32)
        out = U.random_same_position(idx, n, width, 7)
        np.testing.assert_array_equal(out // width, idx // width)                # same layer and row
        self.assertEqual(len(set(out.tolist())), len(out))                       # no unit twice
        np.testing.assert_array_equal(out, U.random_same_position(idx, n, width, 7))
        self.assertGreater(np.mean(out != idx), .9)

    def test_random_same_layer_and_ranks(self):
        width, depth = 40, 5; score = np.random.default_rng(1).normal(size=(depth, width))
        order = U.neuron_order(score, 30)
        self.assertEqual(order[0], int(np.argmax(score)))
        rand = U.random_same_layer(order, width, depth, 3)
        for k in range(1, 31):
            np.testing.assert_array_equal(np.bincount(order[:k] // width, minlength=depth), np.bincount(rand[:k] // width, minlength=depth))
        self.assertEqual(len(set(rand.tolist())), 30)
        ranks = U.rank_array(order, depth, width)
        self.assertEqual(ranks.reshape(-1)[order[4]], 4); self.assertEqual((ranks < U.BIG).sum(), 30)
        self.assertEqual(len(U.neuron_order(score, 10 ** 6)), depth * width)

    def test_specs(self):
        names = [s[0] for s in U.specs()]
        self.assertEqual(len(names), 3 + 5 * len(U.UNIT_GRID) + 4 * len(U.NEURON_GRID)); self.assertEqual(len(set(names)), len(names))


def fake_row(i, ceiling=.8, half_at=100, nec_ceiling=.6, nec_half_at=1000, init=2, readable=True, cell=('7->8', 0)):
    e = dict(none=0., all_suff=ceiling, all_nec=1 - nec_ceiling)
    for k in U.UNIT_GRID:
        e['u_act_%d' % k] = ceiling * (.6 if k >= half_at else .2); e['u_actrand_%d' % k] = .01; e['u_kv_%d' % k] = ceiling * (.6 if k >= 10 * half_at else .1)
        e['u_nec_%d' % k] = 1 - nec_ceiling * (.6 if k >= nec_half_at else .1); e['u_necrand_%d' % k] = .99
    for k in U.NEURON_GRID:
        e['n_act_%d' % k] = ceiling * (.6 if k >= 30 else .1); e['n_actrand_%d' % k] = 0.; e['n_kv_%d' % k] = ceiling * (.6 if k >= 300 else .1)
        e['n_nec_%d' % k] = 1 - nec_ceiling * (.6 if k >= 100 else .1)
    return dict(id='s%d' % i, pair=cell[0], phase=cell[1], init=init, b_done=True, readable=readable, e=e, e_rank=dict(e),
                input_b_exact=True, scores_finite=True, noop_max_abs=0., counts_ok=True, hook_gap=0., denoise_gap=0.,
                e_check=[0., 1.], fidelity_ratio=.01, valid_rows=512,
                pred={'act': {str(k): .001 * k * (i + 1) for k in U.UNIT_GRID}},
                composition={'act': [dict(positions=5, rows=3), dict(positions=40, rows=20), dict(positions=300, rows=100)]})


class Readings(unittest.TestCase):
    def test_k_half_fraction_population(self):
        r = fake_row(0)
        self.assertEqual(RU.k_half(r, 'u_act_%d', U.UNIT_GRID, 'suff'), 100)
        self.assertEqual(RU.k_half(r, 'u_nec_%d', U.UNIT_GRID, 'nec'), 1000)
        self.assertEqual(RU.k_half(r, 'u_actrand_%d', U.UNIT_GRID, 'suff'), float('inf'))
        self.assertAlmostEqual(RU.fraction(r, 'u_nec_1000', 'nec'), .6)
        rows = [r, fake_row(1, ceiling=.2), fake_row(2, readable=False), fake_row(3, init=0)]
        self.assertEqual([x['id'] for x in RU.population(rows, 'suff')], ['s0'])

    def test_sparsity_readings(self):
        for half, reading in ((30, 'sparse'), (3000, 'moderate'), (10 ** 6, 'diffuse')):
            rows = [fake_row(i, half_at=half) for i in range(25)]
            out = RU.sparsity(rows, 'u_act_%d', U.UNIT_GRID, 'suff')
            self.assertEqual(out['reading'], reading, half)
        self.assertEqual(RU.sparsity([fake_row(i) for i in range(5)], 'u_act_%d', U.UNIT_GRID, 'suff')['reading'], 'insufficient')
        out = RU.sparsity([fake_row(i) for i in range(25)], 'n_act_%d', U.NEURON_GRID, 'suff', scale=lambda r: r['valid_rows'])
        self.assertEqual(out['median'], 30 * 512); self.assertEqual(out['reading'], 'diffuse')

    def test_tool_reading(self):
        rows = [fake_row(i) for i in range(3)]; scales = [dict(scale=[1, 1, 1], affine_deviation=0.)]
        self.assertEqual(RU.tool_reading(rows, scales)[0], 'tool_ok')
        for field, bad in (('noop_max_abs', 1e-3), ('counts_ok', False), ('denoise_gap', 1e-3), ('e_check', [0., .9]),
                           ('fidelity_ratio', .9), ('input_b_exact', False), ('scores_finite', False), ('hook_gap', 1.)):
            broken = [dict(rows[0], **{field: bad})] + rows[1:]
            self.assertEqual(RU.tool_reading(broken, scales)[0], 'tool_failed', field)
        self.assertEqual(RU.tool_reading(rows, [dict(scale=[1, 1, 1], affine_deviation=.1)])[0], 'tool_failed')


class Summary(unittest.TestCase):
    def root(self, tmp, mode, rows):
        root = Path(tmp)
        RU.write(root/'STATES.json', dict(states=[dict(id=r['id']) for r in rows]))
        RU.write(root/'PLAN.json', dict(mode=mode, shards=1, inits=list(RU.READING_INITS), state_ids=[r['id'] for r in rows],
                                        states_sha256=RU.sha(root/'STATES.json')))
        RU.write(root/'SHARD_0_COMPLETE.json', dict(passed=True, scale=[1, 1, 1], affine_deviation=0.))
        for r in rows:
            folder = root/'states'/r['id']; folder.mkdir(parents=True); RU.write(folder/'ROW.json', r)
            RU.write(folder/'HASHES.json', {'ROW.json': RU.sha(folder/'ROW.json')})
        return root

    def test_formal_and_gate(self):
        rows = [fake_row(i, half_at=300, cell=('7->8' if i % 2 else '2->9', 0)) for i in range(24)]
        with tempfile.TemporaryDirectory() as tmp:
            root = self.root(tmp, 'formal', rows)
            out = RU.summarize(root)
            self.assertEqual(out['tool'], 'tool_ok'); self.assertEqual(out['r1']['reading'], 'moderate'); self.assertEqual(out['r1']['median'], 300)
            self.assertEqual(out['r2']['reading'], 'moderate'); self.assertEqual(out['r2']['median'], 1000)
            uh = out['report_only']['units_for_half']
            self.assertEqual(uh['u_kv']['median'], 3000); self.assertEqual(uh['n_act']['median'], 30 * 512)
            self.assertEqual(out['report_only']['curves_suff']['curves']['u_act']['300'], .6)
            self.assertEqual(set(out['r1']['by_cell']), {'2->9 f=0', '7->8 f=0'})
            with self.assertRaises(SystemExit):
                RU.summarize(root, require_tool_ok=True)
            (root/'states'/'s3'/'ROW.json').write_text('{}')
            with self.assertRaises(RuntimeError):
                RU.summarize(root)
        bad = [dict(r, noop_max_abs=1.) for r in rows]
        with tempfile.TemporaryDirectory() as tmp:
            root = self.root(tmp, 'smoke', bad)
            with self.assertRaises(SystemExit):
                RU.summarize(root, require_tool_ok=True)
            self.assertIsNone(json.loads((root/'SUMMARY.json').read_text())['r1']['reading'])


if __name__ == '__main__':
    unittest.main()
