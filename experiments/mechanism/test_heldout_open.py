"""Unit tests for heldout_open.py: the plan (pure), the open-loop push on the toy pi0.5 of test_patching (against the
closed-loop server's request 1), the readings and an end-to-end summary on synthetic roots.

Run in the openpi environment with the pinned openpi on PYTHONPATH:
  PYTHONPATH=$HOME/crash_bench/third_party/vlsa-aegis/openpi/src JAX_PLATFORMS=cpu \
    /projects/p33100/siosio/envs/openpi/bin/python -m unittest test_heldout_open
"""
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'feasibility'))
import heldout_open as H  # noqa: E402
import metrics as M  # noqa: E402

GOAL, TEN, OBJECT = H.GOAL, H.TEN, H.OBJECT
L = lambda suite, pair: H.R.label(suite, pair)


# ---------------------------------------------------------------- plan
def unit(uid, suite, task, other, init, phase, condition='SW', group='1a', checkpoint=None, prefix=None, seed=7):
    prefix = prefix or '%s_t%d_i%02d' % (suite, task, init)
    return dict(id=uid, group=group, suite=suite, task=task, other=other, init=init, phase=phase, condition=condition,
                checkpoint=checkpoint if checkpoint is not None else int(100 * phase), prefix=prefix, seed=seed,
                prompt='prompt %s' % other)


class Plan(unittest.TestCase):
    def branch_plan(self, extra=()):
        units = [unit('ten_sw', TEN, 1, 0, 3, .5, group='1b'), unit('ten_c0', TEN, 1, None, 3, .5, 'C0', group='1b'),
                 unit('ten_sw0', TEN, 1, 0, 3, 0, group='1b'),
                 unit('pilot_sw', GOAL, 7, 8, 3, .75), unit('pilot_c0', GOAL, 7, None, 3, .75, 'C0'),
                 unit('pilot_ok', GOAL, 1, 8, 3, .5), unit('pilot_ok_c0', GOAL, 1, None, 3, .5, 'C0'),
                 unit('smoke_sw', TEN, 1, 0, 0, .5, group='1b'), unit('smoke_c0', TEN, 1, None, 0, .5, 'C0', group='1b'),
                 unit('other', GOAL, 0, 5, 3, .5)] + list(extra)
        tokens = {'prompt %d' % k: 5 + k for k in range(10)}
        return dict(branches=units, prompt_tokens=tokens)

    def runs(self, plan):
        return [dict(id=u['id'], b_done=u['id'] == 'ten_sw0', a_done=u['condition'] == 'C0') for u in plan['branches']]

    def test_kinds_classes_and_twins(self):
        plan = self.branch_plan()
        states = {s['sw']: s for s in H.select(plan, self.runs(plan), H.READING_INITS)}
        self.assertEqual(set(states), {'ten_sw', 'ten_sw0', 'pilot_sw'})
        self.assertEqual((states['ten_sw']['kind'], states['ten_sw']['cell'], states['ten_sw']['twin']), ('full', 'derails', 'ten_c0'))
        self.assertEqual((states['ten_sw0']['cell'], states['ten_sw0']['twin']), ('start', None))
        self.assertEqual((states['pilot_sw']['kind'], states['pilot_sw']['cell'], states['pilot_sw']['twin']), ('steer', 'in_failure', 'pilot_c0'))
        self.assertEqual(states['ten_sw']['pair'], 'libero_10 1->0'); self.assertEqual(states['ten_sw']['id'], 'ten_sw_f050')
        self.assertEqual(states['ten_sw']['direction_seed'], H.C.direction_seed('ten_sw'))
        self.assertTrue(states['ten_sw0']['b_done']); self.assertTrue(states['ten_sw']['twin_a_done'])
        smoke = H.select(plan, self.runs(plan), H.SMOKE_INITS)
        self.assertEqual([s['sw'] for s in smoke], ['smoke_sw'])

    def test_refuses_unclassed_cells_duplicate_and_mismatched_twins(self):
        plan = self.branch_plan([unit('obj_sw', OBJECT, 1, 0, 3, .25, group='1c')])
        with self.assertRaises(ValueError):
            H.select(plan, self.runs(plan), H.READING_INITS)
        plan = self.branch_plan([unit('dup', TEN, 1, None, 3, .5, 'C0', group='1b')])
        with self.assertRaises(ValueError):
            H.select(plan, self.runs(plan), H.READING_INITS)
        plan = self.branch_plan(); plan['branches'][1]['seed'] = 8
        with self.assertRaises(ValueError):
            H.select(plan, self.runs(plan), H.READING_INITS)


# ---------------------------------------------------------------- push (pure)
def chunks_for(gap, n=4, horizon=10, push=.5, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, .01, (n, horizon, 7)); b = a.copy(); b[..., 0] += gap
    return dict(a=a, b=b, amp2=b + push * (b - a), rand2=b.copy())


def stats(n=4, error=0.):
    return dict(amp2=np.tile([2. * (1 + error), 2., 9., 3., 3., 9.], (n, 1)), rand2=np.tile([2., 2., 9., 3., 3., 9.], (n, 1)))


class Push(unittest.TestCase):
    def test_push_is_e_minus_one_on_the_readable_seeds(self):
        c = chunks_for(1.); out = H.push_fields(c, stats(), dict(a=c['a'][0], b=c['b'][0]))['push']
        self.assertTrue(out['readable']); self.assertAlmostEqual(out['amp2'], .5, places=6); self.assertAlmostEqual(out['rand2'], 0, places=9)
        self.assertEqual(out['a_vs_native_linf'], 0); self.assertTrue(out['amp2_changed'])
        self.assertLess(out['scale_error']['amp2'], 1e-12); self.assertGreater(out['raw_amp2'], 3)

    def test_unreadable_and_scale_error(self):
        c = chunks_for(0.); out = H.push_fields(c, stats(error=.01), dict(a=c['a'][0], b=c['b'][0] + .1))['push']
        self.assertFalse(out['readable']); self.assertIsNone(out['amp2']); self.assertFalse(out['amp2_changed'])
        self.assertAlmostEqual(out['scale_error']['amp2'], .01, places=9); self.assertAlmostEqual(out['b_vs_native_linf'], .1, places=9)


# ---------------------------------------------------------------- push on the toy pi0.5 (JAX, CPU)
class ToyPush(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from openpi.policies import policy as policy_module
        import patching as P
        import serve_steer
        import test_patching as T
        cls.config, model = T.toy()
        cls.policy = policy_module.Policy(model, transforms=[T.libero_like(cls.config)],
                                          output_transforms=[lambda d: dict(d, actions=d['actions'][:, :7])])
        cls.patcher = P.Patcher(cls.policy)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.server = serve_steer.SteeredPolicy(cls.policy, cls.patcher, Path(cls.tmp.name)/'server')
        rng = np.random.default_rng(3)
        cls.obs = {'observation/image': rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
                   'observation/wrist_image': rng.integers(0, 255, (224, 224, 3), dtype=np.uint8), 'observation/state': rng.normal(size=8)}
        cls.prompt_a, cls.prompt_b = 'turn on the stove', 'put the bowl on the plate'

    @classmethod
    def tearDownClass(cls):
        cls.server.close(); cls.tmp.cleanup()

    def request1(self, kind, seed, direction_seed=9):
        arm = dict(name=kind, kind=kind, lam=2. if kind != 'native' else 1., layers=[0, 1], prompt_a=self.prompt_a, direction_seed=direction_seed)
        self.server.infer({'__paired_reset_rng__': seed, 'run_id': '%s_%d' % (kind, seed), '__mech_arm__': arm})
        return np.asarray(self.server.infer(dict(self.obs, prompt=self.prompt_b))['actions'])

    def test_push_requests_are_the_closed_loop_request_one(self):
        seeds = [77, 78]
        chunks, st = H.push_requests(self.patcher, dict(self.obs, prompt=self.prompt_a), dict(self.obs, prompt=self.prompt_b), seeds, 9, [0, 1])
        for j, s in enumerate(seeds):
            np.testing.assert_array_equal(chunks['b'][j], self.request1('native', s))
            np.testing.assert_array_equal(chunks['amp2'][j], self.request1('amp', s))
            np.testing.assert_array_equal(chunks['rand2'][j], self.request1('rand', s))
        self.assertFalse(np.array_equal(chunks['amp2'], chunks['b'])); self.assertFalse(np.array_equal(chunks['a'], chunks['b']))
        np.testing.assert_allclose(st['amp2'][:, 0], st['amp2'][:, 1], rtol=1e-5); np.testing.assert_allclose(st['rand2'][:, 3], st['rand2'][:, 4], rtol=1e-5)

    def state(self, folder, sid, kind):
        import run_open_loop as O
        from serve import digest
        np.savez(folder/(sid + '_sw.npz'), **self.obs); np.savez(folder/(sid + '_a.npz'), **self.obs)
        seed = 1010  # shared, so the full and steer states' push chunks can be compared
        first = lambda d: dict(input_sha256=digest(d), actions=O.native(self.policy, d, seed).tolist(),
                               native_chunk_sha256=O.chunk_sha(O.native(self.policy, d, seed)))
        return dict(id=sid, kind=kind, cell='derails', pair=L(TEN, '1->0'), sw=sid + '_sw', phase=.5, init=2, b_done=False,
                    twin_a_done=True, seed=seed, prompt_a=self.prompt_a, prompt_b=self.prompt_b, direction_seed=11,
                    input_npz=str(folder/(sid + '_sw.npz')), a_input_npz=str(folder/(sid + '_a.npz')),
                    a_first=first(dict(self.obs, prompt=self.prompt_a)), b_first=first(dict(self.obs, prompt=self.prompt_b)))

    def test_full_and_steer_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = H.run_full_state(self.patcher, self.policy, self.state(root, 'full_state', 'full'), 3, root/'states'/'full_state', [0, 1])
            steer = H.run_steer_state(self.patcher, self.policy, self.state(root, 'steer_state', 'steer'), 3, root/'states'/'steer_state', [0, 1])
            for row in (full, steer):
                p = row['push']
                self.assertTrue(row['input_b_exact']); self.assertLess(max(p['a_vs_native_linf'], p['b_vs_native_linf']), 2e-5)
                self.assertLess(max(p['scale_error'].values()), 1e-5); self.assertTrue(p['amp2_changed']); self.assertEqual(len(p['defined']), 3)
            self.assertIn('interface', full); self.assertEqual(full['noop_max_abs'], 0.); self.assertNotIn('interface', steer)
            self.assertEqual(steer['native_b_vs_logged_linf'], 0.)
            for sid in ('full_state', 'steer_state'):
                arrays = np.load(root/'states'/sid/'ACTIONS.npz')
                self.assertEqual(arrays['push_amp2'].shape, (3, 10, 7)); self.assertEqual(arrays['push_stats_rand2'].shape, (3, 6))
                hashes = json.loads((root/'states'/sid/'HASHES.json').read_text())
                self.assertEqual(set(hashes), {p.name for p in (root/'states'/sid).iterdir()} - {'HASHES.json'})
            np.testing.assert_array_equal(np.load(root/'states'/'full_state'/'ACTIONS.npz')['push_b'][0],
                                          np.load(root/'states'/'steer_state'/'ACTIONS.npz')['push_b'][0])


# ---------------------------------------------------------------- readings and summary on synthetic rows
INITS = (2, 3, 4, 5, 6, 7)
WINDOW_NAMES = ['none', 'full'] + ['L%02d-%02d_%s' % (s, s + w - 1, g) for w in (1, 3) for s in range(18 - w + 1) for g in ('all', 'image', 'text')]
HELD_CELLS = {L(GOAL, '1->4'): (0, .25, .5, .75), L(GOAL, '4->8'): (0, .25, .5, .75), L(GOAL, '8->1'): (0, .25, .5, .75),
              L(GOAL, '8->7'): (0, .25, .5, .75), L(GOAL, '9->2'): (0, .25, .5, .75), L(TEN, '0->1'): (0, .25, .5, .75),
              L(TEN, '1->0'): (0, .25, .5, .75), L(OBJECT, '1->0'): (0, .5)}
CAL_CELLS = [(L(GOAL, '7->8'), .5), (L(GOAL, '7->8'), .75), (L(GOAL, '2->9'), .25), (L(GOAL, '2->9'), .5), (L(GOAL, '2->9'), .75)]
GAIN = {L(GOAL, '7->8'): 4, L(GOAL, '2->9'): 5, L(TEN, '0->1'): 3, L(GOAL, '8->7'): 2, L(TEN, '1->0'): 0, L(OBJECT, '1->0'): 0}
PUSH = {L(GOAL, '7->8'): .6, L(GOAL, '2->9'): .8, L(TEN, '0->1'): .4, L(GOAL, '8->7'): .3, L(TEN, '1->0'): .1, L(OBJECT, '1->0'): None}


def push_row(value, readable=True):
    return dict(floor=.1, defined=[readable] * 4, readable=readable, amp2=value, rand2=0. if readable else None, raw_amp2=5., raw_rand2=.5,
                ab_floors=4., a_vs_native_linf=0., b_vs_native_linf=0., scale_error=dict(amp2=1e-7, rand2=1e-7), amp2_changed=True)


def full_row(pair, phase, init, window=.5, gap=1., push=None):
    cell = H.R.cell_class((pair, phase)); value = PUSH.get(pair) if cell in H.FAILURE_CLASSES else .2
    interface = {n: .05 for n in WINDOW_NAMES}; interface.update(none=0., full=1.); interface[H.WINDOW] = window
    interface['L14-14_all'] = .3
    if window is None:
        interface = {n: None for n in interface}
    sid = '%s_%s_i%02d' % (pair.replace(' ', '_').replace('->', 'to'), phase, init)
    return dict(id=sid + '_f', kind='full', cell=cell, pair=pair, sw=sid, phase=phase, init=init, b_done=False, input_b_exact=True,
                noop_max_abs=0., full_e=[1.] * 4, mech_native_ratio=.001, native_vs_logged=dict(b_linf=0.), interface=interface,
                backbone={'none': 0., 'pre00_image': .9, 'pre10_image': .5},
                gaps=dict(image=[gap] * 18, text=[.8] * 18), push=push_row(value, value is not None) if push is None else push)


def rows_fixture():
    rows = []
    for pair, phases in HELD_CELLS.items():
        for phase in phases:
            cell = H.R.cell_class((pair, phase))
            gap = 1. if phase == 0 else .5 if cell in ('commits', 'derails') else 1.05
            for init in INITS:
                rows.append(full_row(pair, phase, init, .5 if phase == 0 else .1, gap))
    for pair, phase in CAL_CELLS:
        for init in INITS:
            sid = 'cal_%s_%s_i%02d' % (pair[-4:], phase, init)
            rows.append(dict(id=sid + '_f', kind='steer', cell='in_failure', pair=pair, sw=sid, phase=phase, init=init, b_done=False,
                             input_b_exact=True, native_b_vs_logged_linf=0., push=push_row(PUSH[pair])))
    return rows


def loop_fixture(rows):
    """Closed-loop native/amp2 per failure state: GAIN[pair] of the six states switch only under amp2."""
    out = {}
    for r in rows:
        if r['cell'] in H.FAILURE_CLASSES:
            k = INITS.index(r['init'])
            out[r['sw']] = dict(native=False, amp2=k < GAIN[r['pair']])
        else:
            out[r['sw']] = dict(native=True, amp2=True)
    return out


class Readings(unittest.TestCase):
    def test_t0_route(self):
        rows = rows_fixture()
        route = H.t0_route(rows, INITS)
        self.assertEqual(route['reading'], 'shared'); self.assertEqual(route['pairs'][L(TEN, '1->0')]['reading'], 'image_window')
        self.assertEqual(route['pairs'][L(TEN, '1->0')]['peak_single'], '14')
        for r in rows:
            if r['pair'] == L(TEN, '1->0') and r['phase'] == 0:
                r['interface'][H.WINDOW] = .2; r['interface']['L15-17_text'] = .6
            if r['pair'] == L(OBJECT, '1->0') and r['phase'] == 0:
                r['interface'] = {n: (.2 if n.startswith('L') else v) for n, v in r['interface'].items()}
            if r['pair'] == L(GOAL, '1->4') and r['phase'] == 0 and r['init'] > 3:
                r['interface'] = {n: None for n in r['interface']}
        route = H.t0_route(rows, INITS)
        self.assertEqual(route['reading'], 'differs')
        self.assertEqual([route['pairs'][p]['reading'] for p in (L(TEN, '1->0'), L(OBJECT, '1->0'), L(GOAL, '1->4'))],
                         ['elsewhere', 'diffuse', 'unreadable'])
        self.assertEqual(route['pairs'][L(TEN, '1->0')]['best'], 'L15-17_text')

    def test_fade(self):
        rows = rows_fixture()
        out = H.fade(rows, INITS)
        self.assertEqual(out['reading'], 'image_route_fades'); self.assertAlmostEqual(out['failure']['median'], .5)
        self.assertEqual(out['failure']['cells'], 8); self.assertEqual(out['success']['cells'], 10)
        for r in rows:
            if r['phase'] and r['cell'] in ('commits', 'derails'):
                r['gaps']['image'] = [1.] * 18
        self.assertEqual(H.fade(rows, INITS)['reading'], 'image_route_kept')
        for r in rows:
            if r['phase'] and r['cell'] == 'success':
                r['gaps']['image'] = [.85] * 18
            if r['phase'] and r['cell'] in ('commits', 'derails'):
                r['gaps']['image'] = [.5] * 18
        self.assertEqual(H.fade(rows, INITS)['reading'], 'mixed')

    def test_spearman_with_ties(self):
        np.testing.assert_array_equal(H.ranks([3, 1, 3, 2]), [3.5, 1, 3.5, 2])
        self.assertAlmostEqual(H.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.)
        self.assertAlmostEqual(H.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.)
        self.assertTrue(np.isnan(H.spearman([1, 1, 1], [1, 2, 3])))

    def test_proxy(self):
        cells = {('p%d' % k, .5): [(float(k), k / 10.)] * 4 for k in range(10)}
        out = H.proxy(cells, draws=200)
        self.assertEqual(out['reading'], 'proxy_tracks'); self.assertAlmostEqual(out['rho'], 1.)
        cells = {('p%d' % k, .5): [(float(k), -k / 10.)] * 4 for k in range(10)}
        self.assertEqual(H.proxy(cells, draws=200)['reading'], 'proxy_fails')
        cells = {('p%d' % k, .5): [(None, 0.)] * 4 for k in range(10)}
        self.assertEqual(H.proxy(cells, draws=50)['reading'], 'insufficient')

    def test_weak_reason_rule_order_and_frozen_lists(self):
        rows = rows_fixture(); loop = loop_fixture(rows)
        cells = H.failure_cells(rows, loop, lambda r: r['push']['amp2'], INITS)
        route = H.t0_route(rows, INITS)
        out = H.weak_reason(cells, route)
        self.assertEqual(out['effective'], sorted(H.EFFECTIVE)); self.assertEqual(out['weak'], sorted(H.WEAK))
        self.assertEqual(out['pairs'][L(TEN, '1->0')]['reading'], 'weak_push')          # .1 < half of the reference push
        self.assertEqual(out['pairs'][L(OBJECT, '1->0')]['reading'], 'nothing_to_push')  # nothing readable
        route['pairs'][L(OBJECT, '1->0')]['reading'] = 'elsewhere'
        self.assertEqual(H.weak_reason(cells, route)['pairs'][L(OBJECT, '1->0')]['reading'], 'window')
        cells2 = {k: [(.7 if v is not None and k[0] == L(TEN, '1->0') else v, g) for v, g in items] for k, items in cells.items()}
        self.assertEqual(H.weak_reason(cells2, route)['pairs'][L(TEN, '1->0')]['reading'], 'downstream')
        loop2 = dict(loop); loop2.update({r['sw']: dict(native=False, amp2=True) for r in rows if r['pair'] == L(OBJECT, '1->0') and r['phase']})
        with self.assertRaises(RuntimeError):
            H.weak_reason(H.failure_cells(rows, loop2, lambda r: r['push']['amp2'], INITS), route)
        del loop2[rows[-1]['sw']]
        with self.assertRaises(RuntimeError):
            H.failure_cells(rows, loop2, lambda r: r['push']['amp2'], INITS)

    def test_tool_reading(self):
        rows = rows_fixture()
        self.assertEqual(H.tool_reading(rows)[0], 'tool_ok')
        for field, bad in (('scale_error', dict(amp2=.01, rand2=0.)), ('b_vs_native_linf', .2), ('amp2_changed', False)):
            broken = [dict(r, push=dict(r['push'], **{field: bad})) if k == 0 else r for k, r in enumerate(rows)]
            self.assertEqual(H.tool_reading(broken)[0], 'tool_failed', field)
        self.assertEqual(H.tool_reading([dict(r, full_e=[1.1]) if r['kind'] == 'full' else r for r in rows])[0], 'tool_failed')
        self.assertEqual(H.tool_reading([dict(r, input_b_exact=False) if r['kind'] == 'steer' else r for r in rows])[0], 'tool_failed')


class Summary(unittest.TestCase):
    def write_state(self, root, row):
        folder = root/'states'/row['id']; folder.mkdir(parents=True)
        keep = {k: v for k, v in row.items() if k not in ('kind', 'cell', 'sw')}
        H.write(folder/'ROW.json', keep)
        H.write(folder/'HASHES.json', {'ROW.json': H.sha(folder/'ROW.json')})

    def roots(self, tmp, mode='formal'):
        tmp = Path(tmp); rows = rows_fixture(); loop = loop_fixture(rows)
        root = tmp/'run'; root.mkdir()
        H.write(root/'STATES.json', dict(states=[{k: r[k] for k in ('id', 'kind', 'cell', 'pair', 'sw', 'phase', 'init')} for r in rows]))
        H.write(root/'PLAN.json', dict(mode=mode, shards=1, seeds=4, inits=list(INITS), state_ids=[r['id'] for r in rows],
                                       states_sha256=H.sha(root/'STATES.json')))
        for r in rows:
            self.write_state(root, r)
        H.write(root/'SHARD_0_COMPLETE.json', dict(passed=True))
        repair = tmp/'repair'; repair.mkdir()
        with gzip.open(repair/'RUNS.jsonl.gz', 'wt') as stream:
            for sw, arms in loop.items():
                for arm, done in arms.items():
                    stream.write(json.dumps(dict(sw=sw, arm=arm, b_done=done)) + '\n')
                stream.write(json.dumps(dict(sw=sw, arm='rand2', b_done=False)) + '\n')
            stream.write(json.dumps(dict(sw='cal_pilot_f0', arm='native', b_done=True)) + '\n')
        H.write(repair/'HASHES.json', {'RUNS.jsonl.gz': H.sha(repair/'RUNS.jsonl.gz')})
        pilot = tmp/'pilot'; pilot.mkdir(); prow = []
        for r in rows:
            if r['kind'] == 'steer':
                p = full_row(r['pair'], r['phase'], r['init'], window=.15, gap=.6)
                p.update(id=r['id'], sw=r['sw']); prow.append(p)
        for pair in (L(GOAL, '7->8'), L(GOAL, '2->9')):
            for init in INITS:
                p = full_row(pair, 0, init); p.update(id='pilot0_%s_%d' % (pair[-4:], init), sw='cal_pilot_f0'); prow.append(p)
        H.write(pilot/'STATES.json', dict(states=[dict(id=p['id'], pair=p['pair'].split(' ')[1], sw=p['sw'], phase=p['phase']) for p in prow]))
        H.write(pilot/'PLAN.json', dict(state_ids=[p['id'] for p in prow]))
        for p in prow:
            self.write_state(pilot, p)
        return root, repair, pilot

    def test_formal_summary_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, repair, pilot = self.roots(tmp)
            out = H.summarize(root, repair_root=repair, pilot_root=pilot)
            self.assertEqual(out['tool'], 'tool_ok'); self.assertEqual(out['t0_route']['reading'], 'shared')
            self.assertEqual(out['fade']['reading'], 'image_route_fades'); self.assertEqual(out['proxy']['cells_used'], 12)
            self.assertIn(out['proxy']['reading'], ('proxy_tracks', 'proxy_weak'))
            self.assertEqual({p: v['reading'] for p, v in out['weak_reason']['pairs'].items()},
                             {L(TEN, '1->0'): 'weak_push', L(OBJECT, '1->0'): 'nothing_to_push'})
            self.assertEqual(out['proxy_report_only']['window_e']['cells_used'], 13)
            self.assertAlmostEqual(out['proxy_report_only']['gamma_image']['cells']['libero_goal 7->8 f=0.5']['value'], .6)
            cell = out['cells']['full libero_10 0->1 f=0.5']
            self.assertEqual((cell['cls'], cell['states'], cell['native'], cell['amp2']), ('derails', 6, 0, 3))
            self.assertAlmostEqual(cell['gamma_image']['median'], .5); self.assertEqual(cell['peak_single'], '14')
            self.assertIn(L(GOAL, '8->7'), out['backbone_f0'])
            self.assertTrue((root/'SUMMARY.json').exists())
            with self.assertRaises(SystemExit):
                H.summarize(root, require_tool_ok=True, repair_root=repair, pilot_root=pilot)

    def test_gate_and_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, repair, pilot = self.roots(tmp, mode='smoke')
            out = H.summarize(root, require_tool_ok=True, repair_root=repair, pilot_root=pilot)
            self.assertNotIn('t0_route', out); self.assertIn('cells', out)
            first = sorted((root/'states').iterdir())[0]
            (first/'ROW.json').write_text('{}')
            with self.assertRaises(RuntimeError):
                H.summarize(root, repair_root=repair, pilot_root=pilot)
        with tempfile.TemporaryDirectory() as tmp:
            root, repair, pilot = self.roots(tmp)
            with gzip.open(repair/'RUNS.jsonl.gz', 'at') as stream:
                stream.write('{}\n')
            with self.assertRaises(RuntimeError):
                H.summarize(root, repair_root=repair, pilot_root=pilot)


if __name__ == '__main__':
    unittest.main()
