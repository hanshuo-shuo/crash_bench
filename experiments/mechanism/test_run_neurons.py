"""Tests for the P4 open-loop runner (run_neurons.py) and its pure rules (neuron_readings.py).

  PYTHONPATH=$HOME/crash_bench/third_party/vlsa-aegis/openpi/src JAX_PLATFORMS=cpu \
    /projects/p33100/siosio/envs/openpi/bin/python -m unittest test_run_neurons
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import neuron_readings as NR  # noqa: E402


def row(pair, phase, init, mlp=.6, top=None, rand=None, D=None, b_done=True):
    top = top or {k: min(1., k / 1000.) * mlp for k in NR.K_GRID}
    rand = rand or {k: .01 for k in NR.K_GRID}
    D = D or {k: 1. for k in NR.K_GRID}
    rho = dict(noop=0., full=1., attn=1., mlp=mlp, **{'top%d' % k: v for k, v in top.items()}, **{'rand%d' % k: v for k, v in rand.items()},
               **{'single%d' % i: .01 for i in range(1, 11)})
    return dict(pair=pair, phase=phase, init=init, b_done=b_done, rho=rho, D=dict({'top%d' % k: v for k, v in D.items()}, all=2.),
                g_norm=1., input_b_exact=True, noop_max_abs=0., hook_gap=0., mech_native_ratio=.01)


class Rules(unittest.TestCase):
    def test_rank_is_by_mean_score_with_stable_ties_on_hook_layers_only(self):
        s = np.zeros((2, 4, 3)); s[:, 0, 2] = 3; s[:, 1, 0] = 3; s[:, 2, 1] = 1; s[:, 3, 0] = 99   # layer 3 is not a hook layer
        r = NR.rank(s, hooks=[0, 1, 2])
        self.assertEqual([tuple(x[:2]) for x in r[:3]], [(0, 2), (1, 0), (2, 1)])
        self.assertEqual(len(r), 9)

    def test_random_set_matches_layer_counts_and_avoids_the_top(self):
        top = [(0, 1, 1.), (0, 2, 1.), (3, 5, 1.)] + [(1, n, 0.) for n in range(10)]
        out = NR.random_set(top, 3, 'p', width=20)
        self.assertEqual(sorted(l for l, _ in out), [0, 0, 3])
        self.assertFalse({(l, n) for l, n in out} & {(0, 1), (0, 2), (3, 5)})
        self.assertEqual(out, NR.random_set(top, 3, 'p', width=20)); self.assertNotEqual(out, NR.random_set(top, 3, 'q', width=20))

    def rows(self, **kw):
        return [row(p, f, i, **kw) for p in ('1->8', '7->8', '2->9') for f in (0, .25, .5, .75) for i in range(20)]

    def test_route_reading(self):
        pairs = ['1->8', '7->8', '2->9']
        self.assertEqual(NR.route_reading(self.rows(mlp=.6), pairs)[0], 'mlp_route')
        self.assertEqual(NR.route_reading(self.rows(mlp=.2), pairs)[0], 'attention_route')
        self.assertEqual(NR.route_reading(self.rows(mlp=.4), pairs)[0], 'shared')

    def test_sparsity_finds_k_star_and_reads(self):
        pairs = ['1->8', '7->8', '2->9']
        reading, per = NR.sparsity_reading(self.rows(), pairs)        # top(k) = k/1000 * .6: half at k = 1000 (k=300 gives .3*.6)
        self.assertEqual(per['1->8']['k_star'], 1000); self.assertEqual(reading, 'sparse')
        flat = self.rows(top={k: .1 for k in NR.K_GRID})
        reading, per = NR.sparsity_reading(flat, pairs)
        self.assertEqual(reading, 'distributed'); self.assertEqual(NR.k_used(per)['7->8'], 10000)
        mixed = [r if r['pair'] != '2->9' else row(r['pair'], r['phase'], r['init'], top={k: (.6 if k >= 3000 else 0.) for k in NR.K_GRID})
                 for r in self.rows()]
        self.assertEqual(NR.sparsity_reading(mixed, pairs)[0], 'moderate')

    def test_only_reading_inits_count(self):
        rows = [r if r['init'] >= 10 else row(r['pair'], r['phase'], r['init'], mlp=0.) for r in self.rows(mlp=.6)]
        self.assertEqual(NR.route_reading(rows, ['1->8', '7->8', '2->9'])[0], 'mlp_route')

    def test_fade_reading(self):
        used = {'1->8': 100, '7->8': 100, '2->9': 100}

        def world(fail, ok):
            out = []
            for p in ('1->8', '7->8', '2->9'):
                for f in (0, .25, .5, .75):
                    for i in range(10, 20):
                        bad = (p, f) in NR.FAILURE_CELLS
                        v = 1. if f == 0 else (fail if bad else ok)
                        out.append(row(p, f, i, D={k: v * (1 + .01 * i) for k in NR.K_GRID}))
            return out
        self.assertEqual(NR.fade_reading(world(.5, 1.), used)[0], 'neurons_fade')
        self.assertEqual(NR.fade_reading(world(1., 1.), used)[0], 'neurons_kept')
        self.assertEqual(NR.fade_reading(world(.5, .5), used)[0], 'mixed')

    def test_tool_reading(self):
        rows = self.rows(); screen = [dict(input_b_exact=True, finite=True)]
        self.assertEqual(NR.tool_reading(rows, screen, True)[0], 'tool_ok')
        bad = [dict(r, rho=dict(r['rho'], full=.99)) for r in rows]
        self.assertEqual(NR.tool_reading(bad, screen, True)[0], 'tool_failed')
        self.assertEqual(NR.tool_reading(rows, [dict(input_b_exact=True, finite=False)], True)[0], 'tool_failed')
        self.assertEqual(NR.tool_reading([dict(r, hook_gap=.5) for r in rows], screen, True)[0], 'tool_failed')

    def test_prompts(self):
        import run_neurons as RN
        p = RN.prompts(12)
        self.assertEqual(len(p), 14); self.assertEqual(p[8][1], 'put the bowl on the plate'); self.assertEqual(p[10], ('empty', ''))
        from verifier_dataset import language_ablation as A
        self.assertEqual(p[11][1], A.random_prompt(12))


class Pipeline(unittest.TestCase):
    """screen -> select -> measure -> summarize on the toy pi0.5 behind a LIBERO-shaped input transform."""

    @classmethod
    def setUpClass(cls):
        from test_patching import Runner
        Runner.setUpClass()
        cls.runner = Runner
        import neurons as N
        cls.policy = Runner.policy; cls.patcher = N.NeuronPatcher(Runner.policy)

    def test_end_to_end(self):
        import run_neurons as RN
        from test_patching import Runner
        maker = Runner('test_run_state_and_summarize')
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(NR, 'K_GRID', (2, 5, 10)), \
                mock.patch.object(NR, 'READING_INITS', (10,)), mock.patch.object(NR, 'DISCOVERY_INITS', (0,)), \
                mock.patch.object(RN, 'policy_and_patcher', lambda checkpoint: (self.policy, self.patcher)):
            root = Path(tmp)
            spec = [('s0', 0, 0), ('s10', 0, 10), ('s10h', .5, 10)]
            states = []
            for sid, phase, init in spec:
                s = maker.state(root, sid, 'put the bowl on the stove', 'put the bowl on the plate', phase, 0)
                states.append(dict(s, id=sid, init=init, pair='1->8'))
            (root/'STATES.json').write_text(json.dumps(dict(states=states)))
            RN.write(root/'PLAN.json', dict(mode='smoke', shards=1, seeds=3, state_ids=[s['id'] for s in states], discovery_ids=['s0'],
                                            states_sha256=RN.sha(root/'STATES.json'), hooks=[0, 1, 2], window=[2, 3]))
            RN.screen(root, None)
            out = RN.select(root); RN.write(root/'NEURONS.sha256.json', dict(sha256=RN.sha(root/'NEURONS.json')))
            self.assertEqual(len(out['pairs']['1->8']['top']), 10)              # the largest k of the (mocked) grid
            self.assertEqual(len(out['union']), NR.TOP_PROFILE)
            RN.measure(root, 0, 1, None)
            summary = RN.summarize(root)
            d = summary['tool_detail']
            for check in ('inputs', 'noop', 'identity', 'hooks', 'attribution'):
                self.assertTrue(d[check], check)
            self.assertLess(d['identity_worst'], 1e-4)
            r = RN.read(root/'states'/'s10'/'ROW.json')
            self.assertEqual(set(NR.patch_names()) - set(r['rho']), set())
            self.assertIn('attn_L02', r['rho']); self.assertIn('single10', r['rho'])
            self.assertEqual(np.load(root/'states'/'s10'/'PROFILE.npy').shape, (14, NR.TOP_PROFILE, 3))
            self.assertEqual(np.load(root/'states'/'s10'/'MAPS.npy').shape[:2], (2, NR.TOP_PROFILE))
            prof = RN.read(root/'PROFILES_DISCOVERY.json')
            self.assertEqual(len(prof), NR.TOP_PROFILE)
            self.assertEqual(set(next(iter(prof.values()))['mean']), {'1->8 f=0'})      # discovery init 0 only
            (root/'states'/'s10'/'ROW.json').write_text('{}')
            with self.assertRaises(RuntimeError):
                RN.summarize(root)
            RN.write(root/'NEURONS.json', dict(out, union=[]))
            with self.assertRaises(RuntimeError):
                RN.measure(root, 0, 1, None)


if __name__ == '__main__':
    unittest.main()
