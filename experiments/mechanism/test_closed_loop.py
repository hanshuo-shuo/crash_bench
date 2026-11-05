"""Unit tests for closed_loop.py and the HIL runner's arm hand-off (numpy only): python -m unittest test_closed_loop"""
import gzip
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import closed_loop as C
from verifier_dataset import hil_suite as S
from verifier_dataset import hil_suite_run as H


def branch_plan(inits=range(4)):
    prompts = {1: 'put the bowl on the stove', 7: 'turn on the stove', 2: 'put the wine bottle on top of the cabinet', 3: 'open the drawer'}
    prefixes = [dict(id='libero_goal_t%d_i%02d' % (t, i), task=t, init=i, prompt=prompts[t]) for t in (1, 7, 2, 3) for i in inits]
    branches = []
    for (a, b) in ((1, 8), (7, 8), (2, 9), (3, 8)):
        for i in inits:
            for f in (0, .25, .5, .75):
                c = 10 + int(round(80 * f))
                branches.append(dict(id='libero_goal_t%d_i%02d_s%03d_SW_b%d' % (a, i, c, b), group='1a', condition='SW', task=a, other=b,
                                     init=i, phase=f, prefix='libero_goal_t%d_i%02d' % (a, i), prompt='B%d' % b, seed=100 + i, checkpoint=c))
                if f:
                    branches.append(dict(branches[-1], id=branches[-1]['id'].replace('SW_b%d' % b, 'C0'), condition='C0', prompt=prompts[a]))
    return dict(branches=branches, prefix_order=prefixes, prefix_root='/prefix', prefix_hashes={})


class Plan(unittest.TestCase):
    def test_units_cover_the_cells_with_their_arms_and_nothing_else(self):
        p = branch_plan(); runs = [dict(id=u['id'], b_done=u['init'] % 2 == 0) for u in p['branches']]
        out = C.units(p, runs, inits=(0, 1, 2, 3))
        cells = {(u['pair'], u['phase'], u['arm']['name']) for u in out}
        self.assertEqual(len(out), 4 * (5 * 5 + 3 * 3 + 3 * 3))
        self.assertNotIn('3->8', {u['pair'] for u in out}); self.assertNotIn(('7->8', .25, 'native'), cells)
        self.assertIn(('2->9', .25, 'rand4'), cells); self.assertNotIn(('1->8', .5, 'rand2'), cells)
        self.assertIn(('1->8', 0, 'amp4'), cells)
        self.assertEqual(len({u['id'] for u in out}), len(out))
        for u in out:
            self.assertEqual(u['condition'], 'SW'); self.assertEqual(u['id'], u['sw'] + '__' + u['arm']['name'])
            self.assertEqual(u['arm']['layers'], C.LAYERS); self.assertNotEqual(u['arm']['prompt_a'], u['prompt'])
            self.assertEqual(u['arm']['kind'] == 'native', u['arm']['lam'] == 1.)
            self.assertEqual(u['arm']['direction_seed'], C.direction_seed(u['sw']))
        self.assertEqual({u['init'] for u in C.units(p, runs, inits=(2,))}, {2})

    def test_arms_are_the_servers_valid_arms(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'feasibility'))
        import serve_steer
        p = branch_plan(); runs = [dict(id=u['id'], b_done=True) for u in p['branches']]
        for u in C.units(p, runs, inits=(0,)):
            serve_steer.check_arm(json.loads(json.dumps(u['arm'])), 18)

    def test_b_equal_to_a_is_refused(self):
        p = branch_plan(); p['prefix_order'][0]['prompt'] = 'B8'
        with self.assertRaises(ValueError):
            C.units(p, [dict(id=u['id'], b_done=True) for u in p['branches']], inits=(0,))


def runs_for(rate):
    """Synthetic summary rows: 18 states per planned cell; b_done drawn from rate(cell, arm, init) deterministically."""
    out = []
    for cells, kind in ((C.FAILURE, 'failure'), (C.SUCCESS, 'success'), (C.START, 'start')):
        for pair, phase in cells:
            for init in range(2, 20):
                for arm in C.CELL_ARMS[kind]:
                    out.append(dict(sw='%s_%s_%d' % (pair, phase, init), pair=pair, phase=phase, init=init, arm=arm,
                                    b_done=bool((init * 7919 + len(arm)) % 100 < 100 * rate(kind, arm)), a_done=False, b_after=50,
                                    gap_first=.5, gap_run=.4, first_shift_linf=0. if arm == 'native' else .1))
    return out


class Readings(unittest.TestCase):
    inits = C.FORMAL_INITS

    def test_paired_difference_and_interval(self):
        runs = runs_for(lambda kind, arm: .8 if arm == 'amp2' else .2)
        d = C.paired(runs, 'amp2', 'native', C.FAILURE, self.inits)
        self.assertEqual(d['n'], 90); self.assertAlmostEqual(d['mean'], d['x_rate'] - d['y_rate'])
        self.assertLessEqual(d['lo'], d['mean']); self.assertLessEqual(d['mean'], d['hi']); self.assertGreater(d['lo'], .4)
        self.assertIsNone(C.paired(runs, 'amp2', 'native', C.FAILURE, (0,)))

    def test_restoration_readings(self):
        reading = lambda rate, dose=2: C.restoration(runs_for(rate), dose, self.inits)['reading']
        self.assertEqual(reading(lambda k, a: .7 if a == 'amp2' else .2), 'restores_switch')
        self.assertEqual(reading(lambda k, a: .7 if a in ('amp2', 'rand2') else .2), 'nonspecific')
        self.assertEqual(reading(lambda k, a: .2), 'no_restore')
        self.assertEqual(reading(lambda k, a: 0. if a == 'amp4' else .5, 4), 'harms')
        weak = runs_for(lambda k, a: 0.)  # every arm fails; amp2 then switches at inits 9 and 18 of each failure cell (10 of 90)
        for r in weak:
            if r['arm'] == 'amp2' and (r['pair'], r['phase']) in C.FAILURE and r['init'] % 9 == 0:
                r['b_done'] = True
        out = C.restoration(weak, 2, self.inits)
        self.assertEqual(out['reading'], 'weakly_restores'); self.assertAlmostEqual(out['gain']['mean'], 10 / 90)

    def test_selectivity_readings(self):
        reading = lambda rate: C.selectivity(runs_for(rate), 4, self.inits)['reading']
        self.assertEqual(reading(lambda k, a: .95), 'selective')
        self.assertEqual(reading(lambda k, a: .4 if a == 'amp4' else .95), 'damages')

    def test_tables(self):
        t = C.tables(runs_for(lambda k, a: .5), self.inits)
        self.assertEqual(set(t['7->8 f=0.5']), set(C.CELL_ARMS['failure'])); self.assertEqual(t['7->8 f=0.5']['amp2']['n'], 18)
        self.assertEqual(set(t['1->8 f=0']), set(C.CELL_ARMS['start']))
        self.assertEqual(t['7->8 f=0.5']['rand4']['first_shift_median'], .1)
        self.assertEqual(C.readings(runs_for(lambda k, a: .5), self.inits)['amp2_vs_rand4']['n'], 90)

    def test_tool_reading(self):
        good = dict(physics_equal=True, fidelity_linf=.002, max_delta=0., first_shift_linf=0., scale_error=0., arm='native')
        steered = dict(good, arm='amp2', max_delta=3., first_shift_linf=.1, scale_error=1e-6)
        self.assertEqual(C.tool_reading([good, steered])[0], 'tool_ok')
        for bad in (dict(good, physics_equal=False), dict(good, fidelity_linf=.06), dict(good, first_shift_linf=1e-3),
                    dict(steered, scale_error=.01), dict(steered, first_shift_linf=0.)):
            self.assertEqual(C.tool_reading([good, steered, bad])[0], 'tool_failed')


class Summary(unittest.TestCase):
    """run_row and summarize on a fake run root with the runner's file layout."""

    def unit(self, arm, b_done):
        u = dict(id='libero_goal_t7_i02_s050_SW_b8__' + arm, sw='libero_goal_t7_i02_s050_SW_b8', cell='failure', pair='7->8', phase=.5,
                 init=2, other=8, stage1_b_done=False, checkpoint=50,
                 arm=dict(name=arm, kind='native' if arm == 'native' else arm[:-1], lam=1. if arm == 'native' else float(arm[-1]),
                          layers=C.LAYERS, prompt_a='turn on the stove', direction_seed=1))
        return u, b_done

    def write_run(self, root, u, b_done, physics='p50'):
        folder = root/'branch'/u['id']; folder.mkdir(parents=True)
        trace = [dict(step=50 + k, eef=[0., 0., .01 * k], a=False, others={'8': b_done and k == 30}, done=False, physics='x') for k in range(41)]
        trace[0]['physics'] = physics
        scale = u['arm']['lam'] - 1
        requests = [dict(request_index=i + 1, input_sha256='in%d' % i, rng_before='r%d' % i, rng_after='r%d' % (i + 1),
                         native_chunk_sha256='c%d' % i, output_replayed=False, native_output_linf_difference=0.) for i in range(8)]
        log = [dict(request_index=i + 1, input_sha256='in%d' % i, rng_before='r%d' % i, rng_after='r%d' % (i + 1), chunk_sha256='c%d' % i,
                    delta_k=2. * scale, diff_k=2., norm_k=4., delta_v=scale, diff_v=1., norm_v=2., seconds=.1) for i in range(8)]
        log[0].update(fidelity_linf=.001, first_shift_linf=.05 if scale else 0.)
        run_id = 'job_' + u['id']
        (root/'server').mkdir(exist_ok=True)
        (root/'server'/(run_id + '.jsonl')).write_text(''.join(json.dumps(e) + '\n' for e in log))
        (folder/'POLICY.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in requests))
        row = dict(u, outcome=S.outcome(trace, u, [0., 0., 0.]), start_eef=[0., 0., 0.], rng_reset=dict(run_id=run_id, arm=u['arm']),
                   physics_at_checkpoint='p50', first_policy_input_sha256='in0', actions_outside_spec=0, elapsed_seconds=20.)
        H.save(folder, trace, row)

    def test_summarize_reads_the_runs_and_checks_the_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage1 = root/'stage1'; (stage1/'branch'/'libero_goal_t7_i02_s050_SW_b8').mkdir(parents=True)
            (stage1/'branch'/'libero_goal_t7_i02_s050_SW_b8'/'ROW.json').write_text(json.dumps(dict(physics_at_checkpoint='p50', first_policy_input_sha256='in0')))
            (stage1/'HASHES.json').write_text('{}')
            units = [self.unit('native', False), self.unit('amp2', True)]
            for u, b in units:
                self.write_run(root, u, b)
            C.write(root/'PLAN.json', dict(mode='smoke', shards=1, branches=[u for u, _ in units], inits=[2], layers=C.LAYERS,
                                           stage1_root=str(stage1), stage1_hashes_sha256=C.sha(stage1/'HASHES.json')))
            C.write(root/'SHARD_branch_0_COMPLETE.json', dict(passed=True))
            summary = C.summarize(root)
            self.assertEqual(summary['tool'], 'tool_ok'); self.assertEqual(summary['runs'], 2)
            with gzip.open(root/'RUNS.jsonl.gz', 'rt') as stream:
                rows = {r['arm']: r for r in map(json.loads, stream)}
            self.assertFalse(rows['native']['b_done']); self.assertTrue(rows['amp2']['b_done']); self.assertEqual(rows['amp2']['b_after'], 30)
            self.assertAlmostEqual(rows['amp2']['scale_error'], 0); self.assertAlmostEqual(rows['native']['gap_first'], .5)
            g = summary['readings']['restoration']['lam2']['gain']
            self.assertEqual((g['n'], g['mean']), (1, 1.))
            log = root/'server'/'job_libero_goal_t7_i02_s050_SW_b8__amp2.jsonl'
            lines = log.read_text().splitlines(); lines[3] = lines[3].replace('"c3"', '"cx"'); log.write_text('\n'.join(lines) + '\n')
            with self.assertRaises(RuntimeError):
                C.summarize(root)


class Outcome(unittest.TestCase):
    def test_same_outcome_tolerates_last_bit_distances_only(self):
        x = dict(b_done=True, b_after=30, a_after=None, stop_path=0.3830529717522603, start_gap=.19106342706898555)
        self.assertTrue(C.same_outcome(x, dict(x, stop_path=0.38305297175226016, start_gap=.19106342706898552)))
        for k, v in (('b_done', False), ('b_after', 31), ('a_after', 3), ('stop_path', .3831), ('b_done', 1)):
            self.assertFalse(C.same_outcome(x, dict(x, **{k: v})))
        self.assertFalse(C.same_outcome(x, dict(x, extra=1)))


class ClientArm(unittest.TestCase):
    """hil_suite_run.Client sends a unit's arm on the reset and refuses an acknowledgement without it."""

    def fake(self, echo):
        sent = []

        class Fake:
            def __init__(self, host, port):
                pass

            def infer(self, message):
                sent.append(message)
                ack = dict(seed=message['__paired_reset_rng__'], run_id=message['run_id'], requests=0)
                if echo and '__mech_arm__' in message:
                    ack['arm'] = message['__mech_arm__']
                return ack
        module = types.ModuleType('openpi_client.websocket_client_policy'); module.WebsocketClientPolicy = Fake
        return sent, mock.patch.dict(sys.modules, {'openpi_client': types.ModuleType('openpi_client'),
                                                   'openpi_client.websocket_client_policy': module})

    def test_arm_hand_off(self):
        arm = dict(name='amp2', lam=2.)
        sent, patch = self.fake(echo=True)
        with patch:
            H.Client(1, 5, 'run', arm=arm); H.Client(1, 5, 'plain')
        self.assertEqual(sent[0]['__mech_arm__'], arm); self.assertNotIn('__mech_arm__', sent[1])
        sent, patch = self.fake(echo=False)
        with patch, self.assertRaises(RuntimeError):
            H.Client(1, 5, 'run', arm=arm)


if __name__ == '__main__':
    unittest.main()
