"""Tests for the P4 closed loop: the neuron arms of serve_steer.py on the toy pi0.5, and neuron_loop.py's plan and rules.

  PYTHONPATH=$HOME/crash_bench/third_party/vlsa-aegis/openpi/src JAX_PLATFORMS=cpu \
    /projects/p33100/siosio/envs/openpi/bin/python -m unittest test_neuron_loop
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import closed_loop as C  # noqa: E402
import neuron_loop as L  # noqa: E402


class ServerNeuronArms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from openpi.policies import policy as policy_module
        import neurons as N
        import serve_steer
        from test_patching import libero_like, toy
        cls.config, model = toy()
        cls.policy = policy_module.Policy(model, transforms=[libero_like(cls.config)], output_transforms=[lambda d: dict(d, actions=d['actions'][:, :7])])
        cls.tmp = tempfile.TemporaryDirectory()
        cls.sets = dict(hooks=[0, 1, 2], width=128, sets={'7->8': dict(k=4, top=[[0, 3], [1, 5], [1, 7], [2, 9]], random=[[0, 4], [1, 6], [1, 8], [2, 10]])})
        cls.server = serve_steer.SteeredPolicy(cls.policy, N.NeuronPatcher(cls.policy), cls.tmp.name, cls.sets)
        rng = np.random.default_rng(3)
        cls.obs = {'observation/image': rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
                   'observation/wrist_image': rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
                   'observation/state': rng.normal(size=8), 'prompt': 'put the bowl on the plate'}

    @classmethod
    def tearDownClass(cls):
        cls.server.close(); cls.tmp.cleanup()

    def arm(self, name, lam, neurons='top', direction='own'):
        return dict(name=name, kind='nrn', lam=lam, layers=[0, 1, 2], prompt_a='turn on the stove', direction_seed=9,
                    neuron_set='7->8', neurons=neurons, direction=direction)

    def run_arm(self, arm, n=2, seed=77):
        ack = self.server.infer({'__paired_reset_rng__': seed, 'run_id': arm['name'], '__mech_arm__': arm})
        self.assertEqual(ack['arm'], arm)
        out = [self.server.infer(dict(self.obs)) for _ in range(n)]
        with open(Path(self.tmp.name)/(arm['name'] + '.jsonl')) as stream:
            return out, [json.loads(x) for x in stream]

    def test_neuron_arms_hit_targets_and_keep_the_noise(self):
        native, nlog = self.run_arm(dict(name='native', kind='native', lam=1., layers=[12 % 4], prompt_a='turn on the stove', direction_seed=9))
        for arm in (self.arm('nrn2', 2.), self.arm('nrnrand2', 2., 'random'), self.arm('nrndir2', 2., 'top', 'random'),
                    self.arm('mlpall2', 2., 'all'), self.arm('nrnout', 0.), self.arm('nrnoutrand', 0., 'random'), self.arm('mlpallout', 0., 'all')):
            out, log = self.run_arm(arm)
            self.assertEqual([e['rng_before'] for e in log], [e['rng_before'] for e in nlog])
            self.assertLess(log[0]['fidelity_linf'], 1e-4)
            self.assertGreater(log[0]['first_shift_linf'], 0)
            self.assertEqual(L.scale_error(log, 'nrn', arm['lam']) <= 1e-5, True, arm['name'])
            self.assertGreater(sum(log[0]['target']), 0); self.assertEqual(log[0]['target'][3], 0.)   # layer 3 holds no hook
        top, _ = self.run_arm(self.arm('t_top', 2.)); rand, _ = self.run_arm(self.arm('t_rand', 2., 'random'))
        self.assertGreater(np.abs(top[0]['actions'] - rand[0]['actions']).max(), 0)

    def test_matched_random_set_uses_the_top_set_norms(self):
        _, a = self.run_arm(self.arm('m_top', 2.)); _, b = self.run_arm(self.arm('m_rand', 2., 'random'))
        np.testing.assert_allclose(a[0]['target'], b[0]['target'], rtol=1e-6)
        _, o = self.run_arm(self.arm('m_out', 0.))
        np.testing.assert_allclose(o[0]['target'], a[0]['target'], rtol=1e-6)    # lam 0 and lam 2 move by the same norm

    def test_bad_neuron_arms_are_refused(self):
        import serve_steer
        good = self.arm('x', 2.)
        for bad in (dict(good, neuron_set='1->8'), dict(good, neurons='some'), dict(good, direction='up'), dict(good, lam=1.),
                    dict(good, lam=.5), {k: v for k, v in good.items() if k != 'direction'}):
            with self.assertRaises(ValueError):
                serve_steer.check_arm(bad, 4, self.sets)
        with self.assertRaises(ValueError):
            serve_steer.check_arm(good, 4, None)
        self.assertEqual(serve_steer.check_arm(good, 4, self.sets), good)
        with self.assertRaises(ValueError):
            serve_steer.check_arm(dict(name='n', kind='nrn', lam=2., layers=[0], prompt_a='a', direction_seed=1), 4, self.sets)


def run(pair, phase, init, arm, b, a=False):
    return dict(sw='%s_%s_%d' % (pair, phase, init), pair=pair, phase=phase, init=init, arm=arm, b_done=b, a_done=a, b_after=10 if b else None,
                first_shift_linf=.1, kind='nrn')


class Rules(unittest.TestCase):
    def world(self, rates):
        """rates: arm -> (failure rate, success/start rate); deterministic per init."""
        out = []
        for cells, which in ((L.FAILURE, 0), (L.SUCCESS + L.START, 1)):
            for pair, phase in cells:
                for i in L.FORMAL_INITS:
                    for arm, r in rates.items():
                        out.append(run(pair, phase, i, arm, (i - 2) / 18. < r[which]))
        return out

    def test_expected_world(self):
        w = self.world(dict(native=(.2, 1.), amp2=(.7, 1.), nrn2=(.6, 1.), nrnrand2=(.2, 1.), nrndir2=(.25, 1.), mlpall2=(.65, 1.),
                            nrnout=(0, .3), nrnoutrand=(0, 1.), mlpallout=(0, .1)))
        r = L.readings(w, L.FORMAL_INITS)
        self.assertEqual((r['N3']['reading'], r['N4']['reading'], r['N5']['reading']), ('restores_switch', 'selective', 'necessary'))
        self.assertAlmostEqual(r['reported']['nrn2_share_of_amp2'], (11 - 4) / (13 - 4))   # 11, 4 and 13 switches of 18 per cell

    def test_nonspecific_and_not_necessary(self):
        w = self.world(dict(native=(.2, 1.), amp2=(.7, 1.), nrn2=(.6, .5), nrnrand2=(.6, 1.), nrndir2=(.2, 1.), mlpall2=(.6, 1.),
                            nrnout=(0, 1.), nrnoutrand=(0, 1.), mlpallout=(0, 1.)))
        r = L.readings(w, L.FORMAL_INITS)
        self.assertEqual((r['N3']['reading'], r['N4']['reading'], r['N5']['reading']), ('nonspecific', 'damages', 'not_necessary'))
        w = self.world(dict(native=(.2, 1.), amp2=(.7, 1.), nrn2=(.2, 1.), nrnrand2=(.2, 1.), nrndir2=(.2, 1.), mlpall2=(.2, 1.),
                            nrnout=(0, .5), nrnoutrand=(0, .5), mlpallout=(0, 1.)))
        r = L.readings(w, L.FORMAL_INITS)
        self.assertEqual((r['N3']['reading'], r['N5']['reading']), ('no_restore', 'nonspecific'))

    def test_scale_error(self):
        self.assertEqual(L.scale_error([dict(target=[1., 0.], delta=[1., 0.])], 'nrn', 2.), 0.)
        self.assertEqual(L.scale_error([dict(target=[1., 0.], delta=[1., 1e-9])], 'nrn', 2.), np.inf)
        self.assertAlmostEqual(L.scale_error([dict(target=[2.], delta=[2.002])], 'nrn', 0.), 1e-3)
        self.assertEqual(L.scale_error([dict(delta_k=2., diff_k=2., delta_v=1., diff_v=1.)], 'amp', 2.), 0.)

    def test_tool_reading(self):
        base = dict(physics_equal=True, fidelity_linf=0., first_shift_linf=.1, max_delta=1., scale_error=0., prompt_ok=True, kind='nrn')
        runs = [dict(base, kind='native', first_shift_linf=0., max_delta=0.)] + [dict(base) for _ in range(3)]
        self.assertEqual(L.tool_reading(runs)[0], 'tool_ok')
        self.assertEqual(L.tool_reading(runs + [dict(base, scale_error=.01)])[0], 'tool_failed')
        self.assertEqual(L.tool_reading(runs + [dict(base, first_shift_linf=0.)])[0], 'tool_failed')
        self.assertEqual(L.tool_reading(runs + [dict(base, kind='native', max_delta=0., first_shift_linf=.1)])[0], 'tool_failed')

    def test_units_cover_the_cells_with_their_arms(self):
        stage1 = C.read(Path('/projects/p33100/siosio/crashbench_safelibero/hil_suite/20261030T092049Z_branch_pi05_libero_4f435c7d06db')/'PLAN.json')
        import gzip
        with gzip.open('/projects/p33100/siosio/crashbench_safelibero/hil_suite/20261030T092049Z_branch_pi05_libero_4f435c7d06db/RUNS.jsonl.gz', 'rt') as s:
            runs = [json.loads(x) for x in s]
        formal = L.units(stage1, runs, L.FORMAL_INITS, list(range(14)))
        self.assertEqual(len(formal), 90 * 6 + 108 * 5)
        smoke = L.units(stage1, runs, L.SMOKE_INITS, list(range(14)))
        self.assertEqual(len(smoke), 10 * 6 + 12 * 5)
        nrn = [u for u in formal if u['arm']['kind'] == 'nrn']
        self.assertTrue(all(u['arm']['neuron_set'] == u['pair'] and u['arm']['layers'] == list(range(14)) for u in nrn))
        self.assertTrue(all(u['arm']['layers'] == [12, 13, 14] for u in formal if u['arm']['kind'] != 'nrn'))
        self.assertEqual({u['arm']['lam'] for u in formal if u['arm']['name'].endswith('out') or 'out' in u['arm']['name']}, {0.})

    def test_neuron_sets_need_a_formal_tool_ok_root(self):
        import neuron_readings as NR
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            top = [[l, n, 1. / (1 + l * 10 + n)] for l in range(3) for n in range(10)]
            C.write(root/'NEURONS.json', dict(pairs={'7->8': dict(top=top, random={})}, width=10, hooks=[0, 1, 2]))
            C.write(root/'NEURONS.sha256.json', dict(sha256=C.sha(root/'NEURONS.json')))
            C.write(root/'SUMMARY.json', dict(tool='tool_ok', mode='smoke', k_used={'7->8': 5}, readings={}))
            with self.assertRaises(RuntimeError):
                L.neuron_sets(root)
            C.write(root/'SUMMARY.json', dict(tool='tool_ok', mode='formal', k_used={'7->8': 5}, readings={}))
            s = L.neuron_sets(root)['sets']['7->8']
            self.assertEqual(s['top'], [x[:2] for x in top[:5]])
            self.assertEqual(sorted(l for l, _ in s['random']), sorted(l for l, _ in s['top']))
            self.assertFalse({tuple(x) for x in s['random']} & {tuple(x) for x in s['top']})
            self.assertNotEqual(s['random'], NR.random_set(top, 5, '7->8', 10))      # its own seed, not the open loop's


if __name__ == '__main__':
    unittest.main()
