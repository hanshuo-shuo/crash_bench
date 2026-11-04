"""Unit tests for selection.py and metrics.py (numpy only): python -m unittest test_selection_metrics"""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M
import selection as S


def unit(task, other, init, phase, condition='SW', seed=11, group='1a'):
    checkpoint = 10 + int(round(phase * 80))
    return dict(id='g_t%d_i%02d_s%03d_%s%s' % (task, init, checkpoint, condition, '_b%d' % other if condition == 'SW' else ''),
                group=group, prefix='g_t%d_i%02d' % (task, init), task=task, other=other if condition == 'SW' else None,
                init=init, phase=phase, checkpoint=checkpoint, condition=condition, seed=seed,
                prompt=('B%d' % other) if condition == 'SW' else 'A%d' % task)


def plan(pairs=((1, 8), (7, 8), (1, 4)), inits=(0, 1, 2)):
    branches = []
    for a, b in pairs:
        for i in inits:
            for f in S.PHASES:
                branches.append(unit(a, b, i, f, seed=100 * a + i))
                if f and not any(u['id'] == unit(a, b, i, f, 'C0')['id'] for u in branches):
                    branches.append(unit(a, b, i, f, 'C0', seed=100 * a + i))
    tokens = {u['prompt']: len(u['prompt']) for u in branches}
    return dict(branches=branches, prompt_tokens=tokens)


class Selection(unittest.TestCase):
    def test_keeps_every_unit_of_the_pairs_and_pairs_each_with_its_twin(self):
        p = plan(); runs = [dict(id=u['id'], b_done=u['init'] % 2 == 0, a_done=True) for u in p['branches']]
        states = S.select(p, runs, pairs=((1, 8), (7, 8)))
        self.assertEqual(len(states), 2 * 3 * len(S.PHASES))
        self.assertEqual({s['pair'] for s in states}, {'1->8', '7->8'})
        for s in states:
            self.assertEqual(s['twin'] is None, s['phase'] == 0)
            if s['twin']:
                self.assertTrue(s['twin'].endswith('_C0')); self.assertIn('s%03d' % s['checkpoint'], s['twin'])
        self.assertEqual(S.summary(states)['1->8 f=0.25'], dict(states=3, b_done=2))

    def test_does_not_select_on_outcome_and_respects_inits(self):
        p = plan(); runs = [dict(id=u['id'], b_done=False, a_done=False) for u in p['branches']]
        self.assertEqual(len(S.select(p, runs, pairs=((1, 8),))), 3 * len(S.PHASES))
        self.assertEqual({s['init'] for s in S.select(p, runs, pairs=((1, 8),), inits=(1,))}, {1})

    def test_rejects_a_twin_with_another_seed(self):
        p = plan(); runs = [dict(id=u['id'], b_done=True, a_done=True) for u in p['branches']]
        next(u for u in p['branches'] if u['condition'] == 'C0')['seed'] = 999
        with self.assertRaises(ValueError):
            S.select(p, runs)


class Metrics(unittest.TestCase):
    def chunks(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(4, 10, 7)) * .01; b = a.copy(); b[..., 0] += .5
        return a, b

    def test_effect_is_zero_at_base_one_at_donor_and_linear(self):
        a, b = self.chunks()
        np.testing.assert_allclose(M.effect(a, a, b), 0); np.testing.assert_allclose(M.effect(b, a, b), 1)
        np.testing.assert_allclose(M.effect(a + .25 * (b - a), a, b), .25, atol=1e-12)

    def test_path_is_cumulative_translation(self):
        a = np.zeros((10, 7)); a[:, 0] = 1; a[:, 5] = 9
        p = M.path(a).reshape(10, 3)
        np.testing.assert_allclose(p[:, 0], np.arange(1, 11)); np.testing.assert_allclose(p[:, 1:], 0)

    def test_defined_only_beyond_three_noise_floors(self):
        a, b = self.chunks(); floor = M.noise_floor(a)
        self.assertGreater(floor, 0); self.assertTrue(M.defined(a, b, floor).all())
        self.assertFalse(M.defined(a, a + 1e-9, floor).any())

    def test_state_effects_orders_specs_after_base_and_donor(self):
        a, b = self.chunks(); actions = np.stack([a, b, a, b, .5 * (a + b)])
        out = M.state_effects(actions, ['none', 'full', 'half'])
        self.assertAlmostEqual(out['state']['none'], 0); self.assertAlmostEqual(out['state']['full'], 1)
        self.assertAlmostEqual(out['state']['half'], .5)
        with self.assertRaises(ValueError):
            M.state_effects(actions, ['none'])

    def test_state_effect_needs_two_defined_seeds(self):
        a, b = self.chunks(); b2 = a.copy(); b2[0] = b[0]  # only seed 0 differs
        out = M.state_effects(np.stack([a, b2, b2]), ['full'])
        self.assertIsNone(out['state']['full'])

    def test_readings(self):
        row = dict(input_b_exact=True, noop_max_abs=0.0, full_e=[1.0, .99], mech_native_ratio=.01)
        self.assertEqual(M.fidelity_reading([row, row])[0], 'tool_ok')
        self.assertEqual(M.fidelity_reading([row, dict(row, noop_max_abs=1e-7)])[0], 'tool_failed')
        self.assertEqual(M.fidelity_reading([row, dict(row, full_e=[.9])])[0], 'tool_failed')
        self.assertEqual(M.fidelity_reading([row, dict(row, mech_native_ratio=.6)])[0], 'tool_failed')
        self.assertEqual(M.interface_reading({'1->8': .9, '7->8': .75}), 'interface_found')
        self.assertEqual(M.interface_reading({'1->8': .9, '7->8': .5}), 'interface_partial')
        self.assertEqual(M.interface_reading({'1->8': .9, '7->8': .1}), 'interface_absent')
        ci = M.bootstrap_median([.1, .2, .3, None, .4])
        self.assertEqual(ci['n'], 4); self.assertLessEqual(ci['lo'], ci['median']); self.assertLessEqual(ci['median'], ci['hi'])


def rows_for(scale, done=lambda pair, phase, init: True):
    """Synthetic summary rows: band gap = 1 at f = 0 and scale(pair, phase, init) later (image and text alike)."""
    rows = []
    for pair in ('1->8', '7->8', '2->9'):
        for phase in S.PHASES:
            for init in range(20):
                g = 1.0 if phase == 0 else scale(pair, phase, init)
                rows.append(dict(pair=pair, phase=phase, init=init, b_done=done(pair, phase, init),
                                 gaps=dict(image=[9.0] * 12 + [g] * 6, text=[1.0] * 18)))
    return rows


class P3Readings(unittest.TestCase):
    def test_route_fades_only_where_switching_fails(self):
        fade = lambda pair, phase, init: (1.0 + .01 * init) if pair == '1->8' else .5 + .01 * init
        reading, cells = M.route_reading(rows_for(fade))
        self.assertEqual(reading, 'image_route_fades'); self.assertEqual(cells['7->8 f=0.5']['n'], 18)  # inits 0-1 left out
        self.assertEqual(M.route_reading(rows_for(lambda *a: 1.0))[0], 'image_route_kept')
        self.assertEqual(M.route_reading(rows_for(lambda p, f, i: .5 if p == '2->9' else 1.0))[0], 'mixed')
        self.assertEqual(M.route_reading([r for r in rows_for(fade) if r['init'] < 2])[0], 'insufficient')

    def test_band_gap_uses_layers_12_to_17(self):
        row = dict(gaps=dict(image=[9.0] * 12 + [2.0] * 6)); self.assertEqual(M.band_gap(row), 2.0)

    def test_switch_auc(self):
        split = lambda pair, phase, init: init % 2 == 0
        good = M.switch_auc(rows_for(lambda p, f, i: 1.0 + (i % 2 == 0), split))
        self.assertEqual(good['reading'], 'gap_predicts_switch'); self.assertAlmostEqual(good['auroc'], 1.0)
        self.assertEqual({c['success'] for c in good['cells']}, {9})
        flat = M.switch_auc(rows_for(lambda p, f, i: 1.0 + .001 * i, split))
        self.assertEqual(flat['reading'], 'gap_does_not_predict')
        self.assertEqual(M.switch_auc(rows_for(lambda *a: 1.0))['reading'], 'insufficient')
        self.assertAlmostEqual(M.auroc([1, 2], [0, 2]), .625)


if __name__ == '__main__':
    unittest.main()
