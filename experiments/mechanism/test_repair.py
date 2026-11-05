"""Unit tests for repair.py and serve_steer.action_push (numpy only): python -m unittest test_repair"""
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import closed_loop as C
import repair as R
from verifier_dataset import hil_suite as S
from verifier_dataset import hil_suite_run as H

GOAL_PROMPTS = {1: 'put the bowl on the stove', 7: 'turn on the stove', 2: 'put the wine bottle on top of the cabinet', 8: 'put the bowl on the plate',
                9: 'put the wine bottle on the rack', 4: 'put the bowl on top of the cabinet'}
# (suite, A, B) -> B rate and A rate among failures per phase, mimicking stage 1's classes.
PAIRS = {('libero_goal', 1, 8): (1, 1, 1, 1), ('libero_goal', 7, 8): (1, .75, .15, .4), ('libero_goal', 2, 9): (.9, .2, .3, 0),
         ('libero_goal', 1, 4): (.9, .95, .95, .95), ('libero_goal', 4, 8): (1, 1, 1, .95), ('libero_goal', 8, 1): (1, 1, 1, 1),
         ('libero_goal', 8, 7): (1, .8, .65, .4), ('libero_goal', 9, 2): (.95, .85, 1, .8),
         ('libero_10', 0, 1): (1, .05, .2, .05), ('libero_10', 1, 0): (.8, .15, .3, 0),
         ('libero_object', 0, 1): (.05, None, .05, None), ('libero_object', 1, 0): (.95, None, 0, None)}
COMMITTED = {('libero_10', 0, 1, .75), ('libero_10', 1, 0, .75), ('libero_object', 1, 0, .5), ('libero_goal', 2, 9, .75)}


def stage1(inits=range(20)):
    """A stage 1 plan and runs with the class structure of the real one (B rate per cell; failures finish A in COMMITTED)."""
    prefixes, branches, runs = [], [], []
    for (suite, a, b), rates in PAIRS.items():
        for i in inits:
            pid = '%s_t%d_i%02d' % (suite, a, i)
            if pid not in {p['id'] for p in prefixes}:
                prefixes.append(dict(id=pid, suite=suite, task=a, init=i, prompt=GOAL_PROMPTS.get(a, '%s task %d' % (suite, a))))
            for f, rate in zip((0, .25, .5, .75), rates):
                if rate is None:
                    continue
                uid = '%s_t%d_i%02d_s%03d_SW_b%d' % (suite, a, i, 10 + int(80 * f), b)
                branches.append(dict(id=uid, group='1a', suite=suite, condition='SW', task=a, other=b, init=i, phase=f, prefix=pid,
                                     prompt=GOAL_PROMPTS.get(b, '%s task %d' % (suite, b)), seed=100 + i, checkpoint=10 + int(80 * f)))
                done = i < round(rate * len(inits))
                runs.append(dict(id=uid, b_done=done, a_done=(not done) and (suite, a, b, f) in COMMITTED))
    return dict(branches=branches, prefix_order=prefixes, prefix_root='/prefix', prefix_hashes={}), runs


class Plan(unittest.TestCase):
    def test_classify_reproduces_the_frozen_held_out_cells(self):
        p, runs = stage1()
        derived = R.classify(R.stage1_cells(p, runs))
        self.assertEqual({k: set(v) for k, v in derived.items()}, {k: set(v) for k, v in R.HELD.items()})
        self.assertNotIn(R.label('libero_object', '0->1'), {c for v in derived.values() for c, _ in v})
        for pair, _ in R.IN_FAILURE:
            self.assertNotIn(pair, {c for v in derived.values() for c, _ in v})

    def test_classify_moves_a_cell_when_stage_1_differs(self):
        p, runs = stage1()
        for r in runs:
            if r['id'].startswith('libero_10_t0') and '_SW_b1' in r['id'] and '_s070_' in r['id']:
                r['a_done'] = False
        derived = R.classify(R.stage1_cells(p, runs))
        self.assertIn((R.label('libero_10', '0->1'), .75), derived['derails'])

    def test_units_have_their_class_arms_and_prompts(self):
        p, runs = stage1(); out = R.units(p, runs, inits=(0, 1))
        by = {}
        for u in out:
            by.setdefault(u['cell'], set()).add(u['arm']['name'])
            self.assertEqual(u['id'], u['sw'] + '__' + u['arm']['name']); self.assertEqual(u['condition'], 'SW')
            self.assertNotEqual(u['arm']['prompt_a'], u['prompt_b'])
            if u['arm']['name'] == 'rephrase':
                self.assertEqual(u['prompt'], 'do not %s. %s instead' % (u['arm']['prompt_a'], u['prompt_b']))
            else:
                self.assertEqual(u['prompt'], u['prompt_b'])
            self.assertEqual(u['arm']['direction_seed'], R.push_seed(u['sw']) if u['arm']['kind'] == 'actrand' else C.direction_seed(u['sw']))
            self.assertEqual(u['arm']['layers'], R.ALL_LAYERS if u['arm']['name'] == 'ampall' else R.LAYERS)
        self.assertEqual({k: v for k, v in by.items()}, {k: set(v) for k, v in R.CELL_ARMS.items()})
        states = {(u['pair'], u['phase'], u['init']) for u in out}
        self.assertEqual(len(states), 2 * (len(R.IN_FAILURE) + sum(len(v) for v in R.HELD.values())))
        self.assertNotIn(R.label('libero_goal', '1->8'), {u['pair'] for u in out})
        self.assertNotIn((R.label('libero_goal', '7->8'), .25), {(u['pair'], u['phase']) for u in out})
        self.assertEqual(len({u['id'] for u in out}), len(out))
        self.assertEqual({u['init'] for u in R.units(p, runs, inits=(5,))}, {5})

    def test_arms_are_the_servers_valid_arms(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'feasibility'))
        import serve_steer
        p, runs = stage1()
        for u in R.units(p, runs, inits=(0,)):
            serve_steer.check_arm(json.loads(json.dumps(u['arm'])), 18)

    def test_plan_refuses_a_stage_1_run_with_other_classes(self):
        p, runs = stage1()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); C.write(root/'PLAN.json', p); (root/'HASHES.json').write_text('{}')
            with gzip.open(root/'RUNS.jsonl.gz', 'wt') as stream:
                stream.writelines(json.dumps(r) + '\n' for r in runs)
            body = R.plan(root, 'smoke', 2)
            self.assertEqual(body['inits'], [0, 1]); self.assertEqual(body['shards'], 2)
            self.assertEqual([p_['id'] for p_ in body['prefix_order']], sorted([p_['id'] for p_ in body['prefix_order']],
                             key=lambda x: next((q['suite'], q['task'], q['init']) for q in p['prefix_order'] if q['id'] == x)))
            for r in runs:
                if '_s070_SW_b0' in r['id'] and r['id'].startswith('libero_10'):
                    r['b_done'] = True
            with gzip.open(root/'RUNS.jsonl.gz', 'wt') as stream:
                stream.writelines(json.dumps(r) + '\n' for r in runs)
            with self.assertRaises(RuntimeError):
                R.plan(root, 'smoke', 2)


class Push(unittest.TestCase):
    def test_action_push_matches_norms_per_row_and_group(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'feasibility'))
        import serve_steer as V
        rng = np.random.default_rng(0)
        plain = rng.normal(size=(10, 7)).astype(np.float32); delta = rng.normal(size=(10, 7)) * .1; delta[3] = 0
        directions = V.push_directions(12)
        self.assertEqual([len(u) for u in directions], [3, 3, 1]); self.assertEqual(abs(directions[2][0]), 1.)
        for u in directions:
            self.assertAlmostEqual(np.linalg.norm(u), 1.)
        actions, error = V.action_push(plain, delta, directions)
        self.assertEqual(actions.dtype, np.float32); self.assertLess(error, 1e-5)
        moved = actions.astype(np.float64) - plain
        for (lo, hi), u in zip(V.ACTION_GROUPS, directions):
            np.testing.assert_allclose(np.linalg.norm(moved[:, lo:hi], axis=1), np.linalg.norm(delta[:, lo:hi], axis=1), atol=1e-6)
            np.testing.assert_allclose(moved[:, lo:hi] @ u, np.linalg.norm(delta[:, lo:hi], axis=1), atol=1e-6)
        np.testing.assert_array_equal(moved[3], 0)
        self.assertEqual([x.tolist() for x in V.push_directions(12)], [x.tolist() for x in directions])
        self.assertNotEqual(V.push_directions(13)[0].tolist(), directions[0].tolist())


def rows(rate, inits=range(2, 20)):
    """Synthetic summary rows for every planned cell; b_done from rate(cls, arm) deterministically per init."""
    out = []
    cells = [(c, 'in_failure') for c in R.IN_FAILURE] + [(c, k) for k, v in R.HELD.items() for c in v]
    for (pair, phase), cls in cells:
        for init in inits:
            for arm in R.CELL_ARMS[cls]:
                out.append(dict(sw='%s_%s_%d' % (pair, phase, init), pair=pair, phase=phase, init=init, arm=arm, kind=R.ARMS[arm][0],
                                b_done=bool((init * 7919 + 3 * len(arm)) % 100 < 100 * rate(cls, arm)), a_done=False, b_after=40,
                                first_shift_linf=0. if arm in ('native', 'rephrase') else .1))
    return out


class Readings(unittest.TestCase):
    inits = R.FORMAL_INITS

    def test_expected_world(self):
        """amp2 restores commits-type failures only and leaves successes alone; the push and the alternatives do not."""
        def rate(cls, arm):
            if cls in ('success', 'start', 'middle'):
                return .95
            if cls == 'derails':
                return .1
            return .75 if arm == 'amp2' else .15
        r = R.readings(rows(rate), self.inits)
        self.assertEqual(r['action_control']['reading'], 'direction_specific')
        self.assertEqual(r['heldout_restoration']['commits']['reading'], 'restores_switch')
        self.assertEqual(r['heldout_restoration']['derails']['reading'], 'no_restore')
        self.assertEqual(r['heldout_selectivity']['reading'], 'selective')
        self.assertEqual({k: v['reading'] for k, v in r['alternatives'].items()}, dict.fromkeys(R.ALTERNATIVES, 'amp_better'))
        self.assertEqual(r['tables']['%s f=%s' % R.IN_FAILURE[0]]['cls'], 'in_failure')

    def test_push_suffices_and_comparable_alternatives(self):
        def rate(cls, arm):
            if cls in ('success', 'start', 'middle'):
                return .95
            return .15 if arm in ('native', 'rand2', 'rephrase') else .75
        r = R.readings(rows(rate), self.inits)
        self.assertEqual(r['action_control']['reading'], 'push_suffices')
        self.assertEqual(r['heldout_restoration']['derails']['reading'], 'restores_switch')
        self.assertEqual(r['alternatives']['actdiff']['reading'], 'x_comparable')
        self.assertEqual(r['alternatives']['rephrase']['reading'], 'amp_better')

    def test_damage_and_nonspecific(self):
        def rate(cls, arm):
            if cls in ('success', 'start'):
                return .2 if arm == 'amp2' else .95
            return .15 if arm == 'native' else .6
        r = R.readings(rows(rate), self.inits)
        self.assertEqual(r['heldout_selectivity']['reading'], 'damages')
        self.assertEqual(r['heldout_restoration']['commits']['reading'], 'nonspecific')
        self.assertEqual(r['action_control']['reading'], 'push_suffices')

    def test_tool_reading(self):
        good = dict(physics_equal=True, fidelity_linf=.002, max_delta=0., first_shift_linf=0., scale_error=0., push_match_error=0.,
                    prompt_ok=True, arm='native', kind='native')
        rephrased = dict(good, arm='rephrase')
        amp = dict(good, arm='amp2', kind='amp', max_delta=3., first_shift_linf=.1, scale_error=1e-6)
        push = dict(amp, arm='actrand', kind='actrand', push_match_error=1e-7)
        diff = dict(good, arm='actdiff', kind='actdiff', first_shift_linf=.2)
        runs = [good, rephrased, amp, push, diff]
        self.assertEqual(R.tool_reading(runs)[0], 'tool_ok')
        for bad in (dict(good, physics_equal=False), dict(good, fidelity_linf=.06), dict(rephrased, first_shift_linf=1e-3),
                    dict(amp, scale_error=.01), dict(push, push_match_error=1e-3), dict(diff, max_delta=1.), dict(diff, first_shift_linf=0.),
                    dict(push, first_shift_linf=0.), dict(rephrased, prompt_ok=False)):
            self.assertEqual(R.tool_reading(runs + [bad])[0], 'tool_failed', bad)


class Summary(unittest.TestCase):
    """run_row and summarize on a fake run root with the runner's file layout, one state under every arm."""
    SW = 'libero_10_t0_i02_s070_SW_b1'

    def unit(self, name):
        kind, lam, layers = R.ARMS[name]
        a, b = 'put both the alphabet soup and the tomato sauce in the basket', 'put both the cream cheese box and the butter in the basket'
        arm = dict(name=name, kind=kind, lam=lam, layers=list(layers), prompt_a=a, direction_seed=R.push_seed(self.SW) if kind == 'actrand' else 1)
        return dict(id=self.SW + '__' + name, sw=self.SW, cell='commits', pair=R.label('libero_10', '0->1'), phase=.75, init=2, other=1,
                    stage1_b_done=False, checkpoint=70, arm=arm, prompt=R.rephrase(a, b) if name == 'rephrase' else b, prompt_b=b)

    def write_run(self, root, u, b_done):
        folder = root/'branch'/u['id']; folder.mkdir(parents=True)
        trace = [dict(step=70 + k, eef=[0., 0., .01 * k], a=False, others={'1': b_done and k == 30}, done=False, physics='x') for k in range(41)]
        kind = u['arm']['kind']; scale = u['arm']['lam'] - 1 if kind in ('amp', 'rand', 'actrand') else 0.
        requests = [dict(request_index=i + 1, input_sha256='in%d' % i, rng_before='r%d' % i, rng_after='r%d' % (i + 1),
                         native_chunk_sha256='c%d' % i, output_replayed=False, native_output_linf_difference=0.) for i in range(8)]
        log = [dict(request_index=i + 1, input_sha256='in%d' % i, rng_before='r%d' % i, rng_after='r%d' % (i + 1), chunk_sha256='c%d' % i,
                    delta_k=2. * scale, diff_k=2., norm_k=4., delta_v=scale, diff_v=1., norm_v=2., seconds=.1) for i in range(8)]
        for e in log:
            if kind == 'actrand':
                e.update(push_match_error=1e-8, push_l2=.3, amp_shift_l2=.3)
            if kind == 'actdiff':
                e.update(ab_action_l2=.5, ab_action_linf=.2)
        log[0].update(fidelity_linf=.001, first_shift_linf=0. if kind == 'native' else .05)
        run_id = 'job_' + u['id']
        (root/'server').mkdir(exist_ok=True)
        (root/'server'/(run_id + '.jsonl')).write_text(''.join(json.dumps(e) + '\n' for e in log))
        (folder/'POLICY.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in requests))
        row = dict(u, outcome=S.outcome(trace, u, [0., 0., 0.]), start_eef=[0., 0., 0.], rng_reset=dict(run_id=run_id, arm=u['arm']),
                   physics_at_checkpoint='p70', first_policy_input_sha256='in0', actions_outside_spec=0, elapsed_seconds=20.)
        H.save(folder, trace, row)

    def fake_root(self, root, mode='smoke'):
        stage1 = root/'stage1'; (stage1/'branch'/self.SW).mkdir(parents=True)
        (stage1/'branch'/self.SW/'ROW.json').write_text(json.dumps(dict(physics_at_checkpoint='p70', first_policy_input_sha256='in0')))
        (stage1/'HASHES.json').write_text('{}')
        units = [(self.unit(n), n in ('amp2', 'ampall')) for n in R.ARMS]
        for u, b in units:
            self.write_run(root, u, b)
        C.write(root/'PLAN.json', dict(mode=mode, shards=1, branches=[u for u, _ in units], inits=[2], layers=R.LAYERS,
                                       stage1_root=str(stage1), stage1_hashes_sha256=C.sha(stage1/'HASHES.json')))
        C.write(root/'SHARD_branch_0_COMPLETE.json', dict(passed=True))

    def test_summarize_reads_every_arm_and_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fake_root(root)
            summary = R.summarize(root, require_tool_ok=True)
            self.assertEqual(summary['tool'], 'tool_ok'); self.assertEqual(summary['runs'], len(R.ARMS))
            with gzip.open(root/'RUNS.jsonl.gz', 'rt') as stream:
                got = {r['arm']: r for r in map(json.loads, stream)}
            self.assertTrue(got['amp2']['b_done']); self.assertFalse(got['actrand']['b_done']); self.assertTrue(got['rephrase']['prompt_ok'])
            self.assertAlmostEqual(got['actrand']['scale_error'], 0); self.assertEqual(got['actdiff']['max_delta'], 0)
            self.assertAlmostEqual(got['actrand']['push_l2_median'], .3); self.assertAlmostEqual(got['actdiff']['ab_action_l2_median'], .5)
            g = summary['readings']['heldout_restoration']['commits']['gain']
            self.assertEqual((g['n'], g['mean']), (1, 1.))
            self.assertEqual(summary['readings']['alternatives']['ampall']['x_vs_native']['mean'], 1.)

    def test_tampered_prompt_or_log_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fake_root(root)
            folder = root/'branch'/(self.SW + '__rephrase')
            data, trace = H.load(folder); data['prompt'] = data['prompt_b']; H.save(folder, trace, data)  # a run that sent B, not the rephrase
            self.assertEqual(R.summarize(root)['tool_detail']['prompt_errors'], 1)
            with self.assertRaises(SystemExit):
                R.summarize(root, require_tool_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fake_root(root)
            log = root/'server'/('job_%s__actrand.jsonl' % self.SW)
            lines = log.read_text().splitlines(); lines[3] = lines[3].replace('"c3"', '"cx"'); log.write_text('\n'.join(lines) + '\n')
            with self.assertRaises(RuntimeError):
                R.summarize(root)

    def test_gate_refuses_a_formal_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fake_root(root, mode='formal')
            with self.assertRaises(SystemExit):
                R.summarize(root, require_tool_ok=True)


if __name__ == '__main__':
    unittest.main()
