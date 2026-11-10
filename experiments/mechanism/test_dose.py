"""Unit tests for dose.py (numpy only): python -m unittest test_dose"""
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import closed_loop as C
import dose as D
import repair as R
from test_repair import stage1
from verifier_dataset import hil_suite as S
from verifier_dataset import hil_suite_run as H


def stage1_root(tmp, runs_edit=None):
    p, runs = stage1()
    if runs_edit:
        runs_edit(runs)
    root = Path(tmp); C.write(root/'PLAN.json', p); (root/'HASHES.json').write_text('{}')
    with gzip.open(root/'RUNS.jsonl.gz', 'wt') as stream:
        stream.writelines(json.dumps(r) + '\n' for r in runs)
    return root


class Plan(unittest.TestCase):
    def test_cells_arms_and_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            body = D.plan(stage1_root(tmp), 'formal', 4)
        self.assertEqual(body['inits'], list(range(2, 20))); self.assertEqual(body['shards'], 4)
        arms = {}
        for u in body['branches']:
            arms.setdefault(u['cell'], set()).add(u['arm']['name'])
            self.assertEqual(u['id'], u['sw'] + '__' + u['arm']['name']); self.assertEqual(u['prompt'], u['prompt_b'])
            self.assertEqual(u['arm']['layers'], D.ALL_LAYERS if u['arm']['name'] == 'wk0all' else D.LAYERS)
            self.assertEqual(u['arm']['direction_seed'], R.push_seed(u['sw']) if u['arm']['kind'] == 'actrand' else C.direction_seed(u['sw']))
        self.assertEqual(arms, {k: set(v) for k, v in D.CELL_ARMS.items()})
        states = {(u['pair'], u['phase'], u['init']) for u in body['branches']}
        self.assertEqual(len(states), 18 * sum(len(v) for v in D.CELLS.values()))
        self.assertEqual(len(body['branches']), 18 * (4 * 8 + 10 * 4 + 3 * 4))
        self.assertNotIn(R.label(R.GOAL, '7->8'), {u['pair'] for u in body['branches'] if u['phase'] in (.5, .75)})
        ids = [p['id'] for p in body['prefix_order']]
        self.assertEqual(len(ids), len(set(ids))); self.assertEqual({u['prefix'] for u in body['branches']}, set(ids))

    def test_arms_are_the_servers_valid_arms(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'feasibility'))
        import serve_steer
        with tempfile.TemporaryDirectory() as tmp:
            for u in D.plan(stage1_root(tmp), 'smoke', 2)['branches']:
                serve_steer.check_arm(json.loads(json.dumps(u['arm'])), 18)
        self.assertEqual({D.ARMS[a][1] for a in D.CURVE}, {1., .75, .5, .25, 0.})

    def test_plan_refuses_a_stage_1_run_with_other_classes(self):
        def lower_pilot(runs):
            for r in runs:
                if r['id'].startswith('libero_goal_t1_') and '_s050_SW_b8' in r['id'] and int(r['id'].split('_i')[1][:2]) < 5:
                    r['b_done'] = False
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                D.plan(stage1_root(tmp, lower_pilot), 'smoke', 2)

        def middle_succeeds(runs):
            for r in runs:
                if r['id'].startswith('libero_goal_t7_') and '_s030_SW_b8' in r['id']:
                    r['b_done'] = True
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                D.plan(stage1_root(tmp, middle_succeeds), 'smoke', 2)


def rows(rate, inits=range(2, 20), commits=True):
    """Synthetic summary rows for every planned cell: b_done from rate(cls, arm) deterministically per init; a failed run
    finishes A when commits."""
    out = []
    for cls, cells in D.CELLS.items():
        for pair, phase in cells:
            for init in inits:
                for arm in D.CELL_ARMS[cls]:
                    done = bool((init * 7919 + 13 * (sum(map(ord, pair)) % 7)) % 100 < 100 * rate(cls, arm))
                    out.append(dict(sw='%s_%s_%d' % (pair, phase, init), pair=pair, phase=phase, init=init, arm=arm, kind=D.ARMS[arm][0],
                                    b_done=done, a_done=(not done) and commits, b_after=40, first_shift_linf=0. if arm == 'native' else .1,
                                    gap_first=.3))
    return out


class Readings(unittest.TestCase):
    inits = D.FORMAL_INITS

    def test_fade_blocks_and_removal_necessary(self):
        r = D.readings(rows(lambda c, a: dict(native=.97, wk75=.9, wk50=.5, wk25=.3, wk0=.1, wk0all=.05, rnd0=.95, act0=.92)[a]), self.inits)
        self.assertEqual(r['fade']['reading'], 'fade_blocks'); self.assertEqual(r['removal']['reading'], 'necessary')
        self.assertEqual(r['failure_mode']['reading'], 'commits_like_natural')
        curve = r['reported']['curve_pilot']
        self.assertEqual(curve['states'], 72); self.assertGreater(curve['wk75']['rate'], curve['wk0']['rate'])
        self.assertIsNotNone(r['fade']['pilot_vs_rnd0']); self.assertIsNone(r['reported']['fade_heldout']['pilot_vs_rnd0'])

    def test_fade_tolerated_removal_contributes_and_derails(self):
        r = D.readings(rows(lambda c, a: dict(native=.97, wk50=.95, wk0=.6, act0=.97).get(a, .97), commits=False), self.inits)
        self.assertEqual(r['fade']['reading'], 'fade_tolerated'); self.assertEqual(r['removal']['reading'], 'contributes')
        self.assertEqual(r['failure_mode']['reading'], 'derails')

    def test_nonspecific_and_not_necessary(self):
        r = D.readings(rows(lambda c, a: dict(native=.97, wk50=.4, wk0=.2, act0=.2).get(a, .97)), self.inits)
        self.assertEqual(r['fade']['reading'], 'nonspecific'); self.assertEqual(r['removal']['reading'], 'nonspecific')
        r = D.readings(rows(lambda c, a: .97), self.inits)
        self.assertEqual(r['fade']['reading'], 'fade_tolerated'); self.assertEqual(r['removal']['reading'], 'not_necessary')
        self.assertEqual(r['failure_mode']['reading'], 'insufficient')

    def test_tool_reading(self):
        good = dict(physics_equal=True, fidelity_linf=.002, max_delta=0., first_shift_linf=0., scale_error=0., push_match_error=0.,
                    prompt_ok=True, arm='native', kind='native')
        weak = dict(good, arm='wk50', kind='amp', max_delta=1., first_shift_linf=.05, scale_error=1e-6)
        push = dict(weak, arm='act0', kind='actrand', push_match_error=1e-7)
        runs = [good, weak, push]
        self.assertEqual(D.tool_reading(runs)[0], 'tool_ok')
        for bad in (dict(good, physics_equal=False), dict(good, fidelity_linf=.06), dict(good, first_shift_linf=1e-3), dict(weak, scale_error=.01),
                    dict(weak, first_shift_linf=0.), dict(push, push_match_error=1e-3), dict(good, prompt_ok=False)):
            self.assertEqual(D.tool_reading(runs + [bad])[0], 'tool_failed', bad)

    def test_scale_errors_use_the_dose_size(self):
        log = [dict(delta_k=.5 * 2., diff_k=2., delta_v=.5, diff_v=1.)]
        self.assertEqual(D.scale_errors(log, 'amp', .5), [0., 0.]); self.assertEqual(D.scale_errors(log, 'amp', 1.5), [0., 0.])
        self.assertAlmostEqual(max(D.scale_errors(log, 'amp', 0.)), .5); self.assertEqual(D.scale_errors(log, 'native', 1.), [0.])


class Summary(unittest.TestCase):
    """run_row and summarize on a fake run root with the runner's file layout, one state under every arm."""
    SW = 'libero_goal_t1_i02_s050_SW_b8'

    def unit(self, name):
        kind, lam, layers = D.ARMS[name]
        a, b = 'put the bowl on the stove', 'put the bowl on the plate'
        arm = dict(name=name, kind=kind, lam=lam, layers=list(layers), prompt_a=a, direction_seed=R.push_seed(self.SW) if kind == 'actrand' else 1)
        return dict(id=self.SW + '__' + name, sw=self.SW, cell='pilot_success', pair=R.label(R.GOAL, '1->8'), phase=.5, init=2, other=8,
                    stage1_b_done=True, checkpoint=50, arm=arm, prompt=b, prompt_b=b)

    def write_run(self, root, u, b_done):
        folder = root/'branch'/u['id']; folder.mkdir(parents=True)
        trace = [dict(step=50 + k, eef=[0., 0., .01 * k], a=not b_done and k == 40, others={'8': b_done and k == 30}, done=False, physics='x')
                 for k in range(41)]
        kind, lam = u['arm']['kind'], u['arm']['lam']; size = abs(lam - 1) if kind != 'native' else 0.
        requests = [dict(request_index=i + 1, input_sha256='in%d' % i, rng_before='r%d' % i, rng_after='r%d' % (i + 1),
                         native_chunk_sha256='c%d' % i, output_replayed=False, native_output_linf_difference=0.) for i in range(8)]
        log = [dict(request_index=i + 1, input_sha256='in%d' % i, rng_before='r%d' % i, rng_after='r%d' % (i + 1), chunk_sha256='c%d' % i,
                    delta_k=2. * size, diff_k=2., norm_k=4., delta_v=size, diff_v=1., norm_v=2., seconds=.1) for i in range(8)]
        for e in log:
            if kind == 'actrand':
                e.update(push_match_error=1e-8, push_l2=.3, amp_shift_l2=.3)
        log[0].update(fidelity_linf=.001, first_shift_linf=0. if kind == 'native' else .05)
        run_id = 'job_' + u['id']
        (root/'server').mkdir(exist_ok=True)
        (root/'server'/(run_id + '.jsonl')).write_text(''.join(json.dumps(e) + '\n' for e in log))
        (folder/'POLICY.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in requests))
        row = dict(u, outcome=S.outcome(trace, u, [0., 0., 0.]), start_eef=[0., 0., 0.], rng_reset=dict(run_id=run_id, arm=u['arm']),
                   physics_at_checkpoint='p50', first_policy_input_sha256='in0', actions_outside_spec=0, elapsed_seconds=20.)
        H.save(folder, trace, row)

    def fake_root(self, root, mode='smoke'):
        stage1_ = root/'stage1'; (stage1_/'branch'/self.SW).mkdir(parents=True)
        (stage1_/'branch'/self.SW/'ROW.json').write_text(json.dumps(dict(physics_at_checkpoint='p50', first_policy_input_sha256='in0')))
        (stage1_/'HASHES.json').write_text('{}')
        units = [(self.unit(n), n in ('native', 'wk75', 'rnd0', 'act0')) for n in D.ARMS]
        for u, b in units:
            self.write_run(root, u, b)
        C.write(root/'PLAN.json', dict(mode=mode, shards=1, branches=[u for u, _ in units], inits=[2], layers=D.LAYERS,
                                       stage1_root=str(stage1_), stage1_hashes_sha256=C.sha(stage1_/'HASHES.json')))
        C.write(root/'SHARD_branch_0_COMPLETE.json', dict(passed=True))

    def test_summarize_reads_every_arm_and_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fake_root(root)
            summary = D.summarize(root, require_tool_ok=True)
            self.assertEqual(summary['tool'], 'tool_ok'); self.assertEqual(summary['runs'], len(D.ARMS))
            with gzip.open(root/'RUNS.jsonl.gz', 'rt') as stream:
                got = {r['arm']: r for r in map(json.loads, stream)}
            self.assertFalse(got['wk0']['b_done']); self.assertTrue(got['wk0']['a_done']); self.assertTrue(got['act0']['b_done'])
            self.assertAlmostEqual(got['wk0']['scale_error'], 0); self.assertAlmostEqual(got['wk75']['scale_error'], 0)
            self.assertAlmostEqual(got['act0']['push_l2_median'], .3)
            d = summary['readings']['removal']['drop']
            self.assertEqual((d['n'], d['mean']), (1, 1.))
            self.assertEqual(summary['readings']['reported']['curve_pilot']['wk75']['b_done'], 1)

    def test_wrong_size_or_tampered_log_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fake_root(root)
            log = root/'server'/('job_%s__wk50.jsonl' % self.SW)
            lines = [json.loads(x) for x in log.read_text().splitlines()]; lines[2]['delta_v'] = 1.
            log.write_text(''.join(json.dumps(e) + '\n' for e in lines))
            self.assertEqual(D.summarize(root)['tool'], 'tool_failed')
            with self.assertRaises(SystemExit):
                D.summarize(root, require_tool_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fake_root(root)
            log = root/'server'/('job_%s__act0.jsonl' % self.SW)
            lines = log.read_text().splitlines(); lines[3] = lines[3].replace('"c3"', '"cx"'); log.write_text('\n'.join(lines) + '\n')
            with self.assertRaises(RuntimeError):
                D.summarize(root)

    def test_gate_refuses_a_formal_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fake_root(root, mode='formal')
            with self.assertRaises(SystemExit):
                D.summarize(root, require_tool_ok=True)


if __name__ == '__main__':
    unittest.main()
