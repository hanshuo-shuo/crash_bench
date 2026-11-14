"""Unit tests for retreat.py (numpy only): python -m unittest test_retreat"""
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import census as K
import closed_loop as C
import retreat as R
import test_census as T
from verifier_dataset import hil_suite as S
from verifier_dataset import hil_suite_run as H

K.DRAWS = R.DRAWS = 300   # the readings' bootstrap at test size (2000 in the summary)


def census_units():
    every = T.prefixes()
    return every, K.branch_units(every, {p['id']: T.tape(p) for p in every}, 'formal')[0]


class Plan(unittest.TestCase):
    def test_units_per_mode(self):
        _, units = census_units(); census_plan = dict(branches=units)
        formal = R.units(census_plan, 'formal'); arms = lambda us: {a: sum(u['retreat_arm'] == a for u in us) for a in R.ARMS}
        self.assertEqual(arms(formal), dict(home=1800, release=1800, native=180))          # 90 pairs x 10 inits x 2 phases; recheck on one init
        self.assertEqual({u['init'] for u in formal if u['retreat_arm'] == 'native'}, {R.RECHECK_INIT})
        self.assertEqual(len({u['id'] for u in formal}), len(formal))
        self.assertEqual(arms(R.units(census_plan, 'smoke')), dict(home=20, release=20, native=20))
        self.assertEqual(arms(R.units(census_plan, 'dry')), dict(home=20, release=20, native=0))
        for u in R.units(census_plan, 'smoke'):
            self.assertEqual((u['init'], u['other']), (0, (u['task'] + 3) % 10))

    def test_units_keep_the_census_state_and_the_native_arm(self):
        _, units = census_units(); by = {u['id']: u for u in units}
        for u in R.units(dict(branches=units), 'formal')[::97]:
            census = by[u['sw'] + '__native']
            self.assertEqual(u['id'], u['sw'] + '__' + u['retreat_arm']); self.assertEqual(u['condition'], R.ARMS[u['retreat_arm']])
            self.assertIn(u['phase'], (.5, .75)); self.assertEqual(u['arm'], census['arm']); self.assertEqual(u['arm']['name'], 'native')
            for key in ('sw', 'prefix', 'checkpoint', 'prompt', 'seed', 'budget', 'stop_on', 'record', 'pair', 'covariates', 'task', 'other', 'init'):
                self.assertEqual(u[key], census[key])
        self.assertEqual({u['condition'] for u in R.units(dict(branches=units), 'formal')}, {'HOME', 'RELEASE', 'SW'})
        self.assertTrue({'HOME', 'RELEASE'} <= set(S.RETREAT))

    def census_root(self, tmp, mode='formal', tool='tool_ok', experiment='mechanism-census'):
        every, units = census_units(); root = Path(tmp)/'census'; root.mkdir()
        C.write(root/'PLAN.json', dict(experiment=experiment, branches=units, prefix_root='/prefix', prefix_hashes={'PREFIXES.json': 'x'},
                                       prefix_order=every, excluded=[dict(reason='prefix task failed')]))
        C.write(root/'SUMMARY.json', dict(mode=mode, tool=tool)); (root/'RUNS.jsonl.gz').write_bytes(b'')
        C.write(root/'HASHES.json', {n: C.sha(root/n) for n in ('PLAN.json', 'RUNS.jsonl.gz', 'SUMMARY.json')})
        return root

    def test_plan_and_refusals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.census_root(tmp); out = Path(tmp)/'out'; out.mkdir()
            body = R.write_plan(root, out, 'smoke')
            self.assertEqual(C.read(out/'PLAN.json'), json.loads(json.dumps(body)))
            self.assertEqual((body['mode'], body['experiment'], body['shards'], len(body['branches'])), ('smoke', 'mechanism-retreat', 2, 60))
            self.assertEqual(body['params'], S.RETREAT_PARAMS); self.assertEqual(body['prefix_root'], '/prefix')
            self.assertEqual(body['census_hashes_sha256'], C.sha(root/'HASHES.json')); self.assertEqual(body['inits'], [0])
            self.assertEqual([p['id'] for p in body['prefix_order']], ['libero_goal_t%d_i00' % k for k in range(10)])
            with self.assertRaises(RuntimeError):
                R.write_plan(root, out, 'smoke')                                            # a plan is never overwritten
            dry = R.plan(root, 'dry', dict(slow=dict(max_action=.5)))
            self.assertEqual(dry['settings'], dict(slow=dict(max_action=.5))); self.assertEqual(R.plan(root, 'dry')['settings'], dict(default={}))
            self.assertEqual(len(R.plan(root, 'formal')['branches']), 3780); self.assertEqual(R.plan(root, 'formal')['shards'], 6)
            (root/'SUMMARY.json').write_text('{}')
            with self.assertRaises(RuntimeError):
                R.plan(root, 'formal')                                                      # the census run changed
        for bad in (dict(mode='smoke'), dict(tool='tool_failed'), dict(experiment='mechanism-dose')):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(RuntimeError):
                    R.plan(self.census_root(tmp, **bad), 'formal')


def world(native, home, release, handover=lambda s: False, inits=range(10)):
    """Per-state rows of a synthetic world: census native / amp2 from T.world, then this batch's arms drawn from the given
    rates (functions of the state row)."""
    census = K.states(T.world(native, inits=inits)); _, classes = K.classes(K.cell_table(census))
    for s in census:
        s['start_gap'] = .3
    runs = []
    for s in census:
        if not s['phase']:
            continue
        for arm, rate in (('home', home), ('release', release)):
            done = T.noise(s['sw'], arm) < rate(s)
            runs.append(dict(sw=s['sw'], arm=arm, b_done=done, a_done=not done and s['a_done'], b_at_handover=handover(s) and arm == 'home',
                             reached=True, scripted_steps=40, within_budget=done, near_b=done, moved_b=.0, gap_first=.3))
        if s['init'] == R.RECHECK_INIT:
            runs.append(dict(sw=s['sw'], arm='native', b_done=s['native'], a_done=s['a_done']))
    return R.state_rows(runs, census, classes)


SAME = lambda a, b: K.MOVED[a] == K.MOVED[b]
NATIVE = lambda a, b, f, cov: 1. if f == 0 else .95 if SAME(a, b) else .1      # from the start all switch; mid-task only same-object pairs do


class Readings(unittest.TestCase):
    def test_retreat_works_and_the_pose_matters(self):
        r = R.readings(world(NATIVE, lambda s: .99 if s['covariates']['same_object'] else .9, lambda s: .95 if s['covariates']['same_object'] else .15))
        u1 = r['upper_bound']
        self.assertEqual((u1['reading'], u1['home']['n']), ('high', 1800)); self.assertGreater(u1['home']['lo'], .8)
        self.assertGreater(u1['vs_native']['lo'], .5); self.assertEqual(r['release_enough']['reading'], 'pose_matters')
        self.assertEqual(r['cost']['reading'], 'harmless'); self.assertEqual(r['cost']['change']['pairs'], 14)     # the same-object pairs
        rep = r['reported']
        self.assertEqual(rep['classes']['failure'], 152); self.assertEqual(rep['classes']['middle'] + rep['classes']['success'], 28)
        self.assertEqual(rep['recheck'], dict(states=180, same_b=180, same_outcome=180, recheck_b=rep['recheck']['census_b'], census_b=rep['recheck']['census_b']))
        self.assertEqual(rep['handover_b'], dict(home=0, release=0)); self.assertEqual(rep['reached']['home']['rate'], 1.)
        self.assertEqual(rep['pairs']['count'], 90); self.assertEqual(rep['gamma_post']['median'], 1.)
        self.assertEqual(sum(rep['flow_native_to_home'].values()), 1800)
        self.assertEqual(set(rep['home_by']['same_object']), {'False', 'True'}); self.assertEqual(rep['home_by']['same_object']['True']['n'], 280)
        self.assertEqual(rep['went_to_b']['home']['done'], 0)                               # failed runs never came near B in this world

    def test_letting_go_is_enough(self):
        r = R.readings(world(NATIVE, lambda s: .6, lambda s: .6))
        self.assertEqual((r['upper_bound']['reading'], r['release_enough']['reading']), ('moderate', 'release_enough'))

    def test_neither_helps(self):
        r = R.readings(world(NATIVE, lambda s: .3, lambda s: .25))
        self.assertEqual((r['upper_bound']['reading'], r['release_enough']['reading']), ('low', 'unclear'))

    def test_retreat_costs_working_switches(self):
        r = R.readings(world(NATIVE, lambda s: .5 if s['covariates']['same_object'] else .9, lambda s: .2))
        self.assertEqual(r['cost']['reading'], 'costs'); self.assertLess(r['cost']['change']['hi'], 0)

    def test_b_true_at_the_handover_leaves_the_state_out(self):
        rows = world(NATIVE, lambda s: 1., lambda s: 0., handover=lambda s: s['init'] == 0)
        r = R.readings(rows)
        self.assertEqual(r['upper_bound']['home']['n'], 1620); self.assertEqual(r['upper_bound']['vs_native']['n'], 1620)
        self.assertEqual(r['reported']['handover_b'], dict(home=180, release=0)); self.assertEqual(r['reported']['release']['n'], 1800)
        self.assertTrue(all(s['home'] is None for s in rows if s['init'] == 0))

    def test_rate_and_cell_median(self):
        rows = [dict(pair='0->1', phase=.5, x=1, g=2.), dict(pair='0->1', phase=.5, x=0, g=4.), dict(pair='0->2', phase=.5, x=None, g=None),
                dict(pair='0->2', phase=.75, x=1, g=10.)]
        r = R.rate(rows, 'x')
        self.assertEqual((r['n'], r['done'], r['cells'], r['pairs']), (3, 2, 2, 2)); self.assertAlmostEqual(r['mean'], 2 / 3)
        self.assertTrue(r['lo'] <= r['mean'] <= r['hi'])
        self.assertEqual(R.rate(rows, 'x', {('0->2', .75)})['n'], 1); self.assertIsNone(R.rate(rows, 'x', {('9->9', .5)}))
        m = R.cell_median(rows, 'g', {('0->1', .5), ('0->2', .75), ('0->2', .5)})
        self.assertEqual((m['n'], m['cells'], m['median']), (3, 2, 6.5))                    # cell medians 3 and 10


class Tool(unittest.TestCase):
    good = dict(arm='home', physics_equal=True, script_in_range=True, script_contiguous=True, scripted_steps=40, reached=True, fidelity_linf=.002,
                max_delta=0., first_shift_linf=0., prompt_ok=True, b_at_switch=False)
    check = dict(passed=True, runs=3, cross_checks=3)

    def test_tool_reading(self):
        P = dict(S.RETREAT_PARAMS); release = dict(self.good, arm='release', scripted_steps=20); native = dict(self.good, arm='native', scripted_steps=0, reached=None)
        runs = [self.good] * 9 + [release, native]
        reading, detail = R.tool_reading(runs, self.check, P)
        self.assertEqual(reading, 'tool_ok'); self.assertEqual((detail['home_runs'], detail['home_reach_rate']), (9, 1.))
        self.assertEqual(R.tool_reading(runs + [dict(self.good, reached=False)], self.check, P)[0], 'tool_ok')          # 9 of 10 arrive
        self.assertEqual(R.tool_reading(runs + [dict(self.good, reached=False)] * 2, self.check, P)[0], 'tool_failed')  # 9 of 11
        for bad in (dict(self.good, physics_equal=False), dict(self.good, script_in_range=False), dict(native, script_contiguous=False),
                    dict(self.good, scripted_steps=P['home_steps'] + 1), dict(release, scripted_steps=P['release_steps'] + 1),
                    dict(self.good, scripted_steps=0), dict(self.good, fidelity_linf=.2), dict(self.good, max_delta=1.),
                    dict(self.good, first_shift_linf=.1), dict(self.good, prompt_ok=False), dict(self.good, b_at_switch=True)):
            self.assertEqual(R.tool_reading(runs + [bad], self.check, P)[0], 'tool_failed', bad)
        self.assertEqual(R.tool_reading(runs, {}, P)[0], 'tool_failed'); self.assertEqual(R.tool_reading(runs, dict(self.check, passed=False), P)[0], 'tool_failed')
        self.assertEqual(R.tool_reading(runs, self.check, dict(P, pos_gain=.1))[0], 'tool_failed')    # run with other constants than the code's
        self.assertEqual(R.tool_reading([release, native], self.check, P)[0], 'tool_failed')          # no home run at all


class Summary(unittest.TestCase):
    """run_row and summarize on fake roots: census prefix t1 / init 0 with B = 4, then this batch's six smoke units."""

    def build(self, root, mode='smoke', reached=(True, True), action=1.):
        pre = root/'prefix_root'; every = T.prefixes(); ps = [p for p in every if p['id'] == 'libero_goal_t1_i00']
        for p in ps:
            folder = pre/'prefix'/p['id']; folder.mkdir(parents=True); H.save(folder, T.tape(p), p)
        C.write(pre/'PREFIXES.json', ps); hashes = {'PREFIXES.json': C.sha(pre/'PREFIXES.json')}
        units, _ = K.branch_units(every, {p['id']: T.tape(p) for p in every}, 'formal')
        units = [u for u in units if u['prefix'] == 'libero_goal_t1_i00' and u['other'] == 4]
        census = root/'census'; census.mkdir(); runs = []
        for u in units:
            runs.append(dict(id=u['id'], sw=u['sw'], pair=u['pair'], task=1, other=4, init=0, phase=u['phase'], arm=u['arm']['name'], kind=u['arm']['kind'],
                             b_done=u['phase'] == 0, a_done=u['phase'] != 0, covariates=u['covariates'], gap_first=.4 if u['phase'] == 0 else .2))
        start = next(u for u in units if u['phase'] == 0); folder = census/'branch'/start['id']; folder.mkdir(parents=True)
        with gzip.open(folder/'STEPS.jsonl.gz', 'wt') as stream:    # B from the start closes the gripper at x = .3: B's work point
            stream.writelines(json.dumps(dict(step=10 + k, eef=[.01 * k, 0., 1.], gripper=[.04, -.04] if k < 30 else [.01, -.01])) + '\n' for k in range(40))
        with gzip.open(census/'RUNS.jsonl.gz', 'wt') as stream:
            stream.writelines(json.dumps(r) + '\n' for r in runs)
        C.write(census/'PLAN.json', dict(experiment='mechanism-census', branches=units, prefix_root=str(pre), prefix_hashes=hashes, prefix_order=ps, excluded=[]))
        C.write(census/'SUMMARY.json', dict(mode='formal', tool='tool_ok'))
        C.write(census/'HASHES.json', {n: C.sha(census/n) for n in ('PLAN.json', 'RUNS.jsonl.gz', 'SUMMARY.json')})
        body = R.plan(census, 'smoke'); body['mode'] = mode; body['shards'] = 1; run = root/'run'; arrive = iter(reached)
        for u in body['branches']:
            folder = run/'branch'/u['id']; folder.mkdir(parents=True); n = dict(home=40, release=20, native=0)[u['retreat_arm']]
            done = u['retreat_arm'] == 'home'; t = u['checkpoint']; rec = lambda k, **x: dict(step=t + k, eef=[.3 if 0 < k <= n else 0., 0., 1.], a=False, others={'4': False}, done=False, physics='x', **x)
            trace = [rec(0)] + [rec(k, action=[action, 0., 0., 0., 0., 0., -1.], scripted=True) for k in range(1, n + 1)]
            trace += [rec(n + k) for k in range(1, 30)] + [dict(rec(n + 30), others={'4': done})]
            requests = [dict(request_index=i + 1, input_sha256='in%d' % i, rng_before='r%d' % i, rng_after='r%d' % (i + 1),
                             native_chunk_sha256='c%d' % i, output_replayed=False, native_output_linf_difference=0.) for i in range(4)]
            log = [dict(request_index=i + 1, input_sha256='in%d' % i, rng_before='r%d' % i, rng_after='r%d' % (i + 1), chunk_sha256='c%d' % i,
                        delta_k=0., diff_k=2., norm_k=4., delta_v=0., diff_v=.36 if done else .2, norm_v=1., seconds=.1) for i in range(4)]
            log[0].update(fidelity_linf=.001, first_shift_linf=0.)
            run_id = 'job_' + u['id']; (run/'server').mkdir(exist_ok=True, parents=True)
            (run/'server'/(run_id + '.jsonl')).write_text(''.join(json.dumps(e) + '\n' for e in log))
            (folder/'POLICY.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in requests))
            row = dict(u, outcome=(S.retreat_outcome if n else S.outcome)(trace, u, [0., 0., 0.]), start_eef=[0., 0., 0.], rng_reset=dict(run_id=run_id, arm=u['arm']),
                       physics_at_checkpoint='libero_goal_t1_i00_h%d' % t, first_policy_input_sha256='in0', actions_outside_spec=0, elapsed_seconds=10.)
            if n:
                ok = next(arrive) if done else True
                row['retreat'] = dict(applied=True, status='reached' if ok else 'cap', reached=ok, steps=n, position_error=.01, orientation_error=.05,
                                      moved=dict(akita_black_bowl_1=.05, plate_1=0.))
            H.save(folder, trace, row)
        C.write(run/'PLAN.json', body); C.write(run/'SHARD_branch_0_COMPLETE.json', dict(passed=True))
        C.write(run/'STATE_CHECK.json', dict(passed=True, runs=6, cross_checks=6))
        return run

    def rows(self, folder):
        with gzip.open(folder/'RUNS.jsonl.gz', 'rt') as stream:
            return {r['id'].split('__')[1] + str(r['phase']): r for r in map(json.loads, stream)}

    def test_summarize_and_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.build(Path(tmp)); summary = R.summarize(run, require_tool_ok=True)
            self.assertEqual((summary['tool'], summary['runs'], summary['runs_per_arm']), ('tool_ok', 6, dict(home=2, release=2, native=2)))
            got = self.rows(run); home = got['home0.5']
            self.assertEqual((home['b_done'], home['scripted_steps'], home['b_after'], home['within_budget'], home['reached']), (True, 40, 30, True, True))
            self.assertEqual((home['moved_b'], home['b_at_handover'], home['script_in_range'], home['script_contiguous']), (.05, False, True, True))
            self.assertTrue(home['near_b'])                                 # the hand-over record sits at B's work point
            self.assertEqual((got['native0.5']['scripted_steps'], got['native0.5']['reached'], got['native0.5']['near_b']), (0, None, False))
            self.assertEqual((got['release0.75']['b_done'], got['release0.75']['scripted_steps']), (False, 20))
            u1 = summary['readings']['upper_bound']
            self.assertEqual((u1['reading'], u1['home']['n'], u1['home']['done']), ('high', 2, 2))
            self.assertAlmostEqual(summary['readings']['reported']['gamma_post_auc']['all'] or 0, 0)                 # one class only: no AUC
            self.assertEqual(summary['readings']['release_enough']['reading'], 'insufficient')                      # one state per cell: no failure cell
            states = R.state_rows(list(self.rows(run).values()), *R.census_view(Path(tmp)/'census')[:2])
            self.assertEqual(sorted(round(s['gamma_post'], 6) for s in states), [.9, .9])                           # .36 over the f = 0 gap .4
            self.assertEqual(sorted(s['gamma_switch'] for s in states), [.5, .5])
            self.assertEqual({(s['recheck'], s['native']) for s in states}, {(0, 0)})

    def test_gate_recomputes_into_another_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.build(Path(tmp)); out = Path(tmp)/'gate'
            summary = R.summarize(run, out=out, require_tool_ok=True)
            self.assertEqual(summary['summarized_root'], str(run))
            self.assertTrue((out/'SUMMARY.json').exists() and (out/'RUNS.jsonl.gz').exists()); self.assertFalse((run/'SUMMARY.json').exists())

    def test_gate_refusals(self):
        with tempfile.TemporaryDirectory() as tmp:                           # one of the two home runs did not arrive
            run = self.build(Path(tmp), reached=(True, False))
            self.assertEqual(R.summarize(run)['tool_detail']['home_reach_rate'], .5)
            with self.assertRaises(SystemExit):
                R.summarize(run, require_tool_ok=True)
        with tempfile.TemporaryDirectory() as tmp:                           # a scripted action outside the action range
            self.assertEqual(R.summarize(self.build(Path(tmp), action=1.2))['tool_detail']['script_out_of_range'], 4)
        with tempfile.TemporaryDirectory() as tmp:                           # the smoke ran with other constants than this code's
            run = self.build(Path(tmp)); plan = C.read(run/'PLAN.json'); plan['params']['pos_gain'] = .1; C.write(run/'PLAN.json', plan)
            self.assertEqual(R.summarize(run)['tool'], 'tool_failed')
        with tempfile.TemporaryDirectory() as tmp:                           # the gate only passes a smoke
            with self.assertRaises(SystemExit):
                R.summarize(self.build(Path(tmp), mode='formal'), require_tool_ok=True)
        with tempfile.TemporaryDirectory() as tmp:                           # the census run changed after planning
            run = self.build(Path(tmp)); (Path(tmp)/'census'/'HASHES.json').write_text('{}')
            with self.assertRaises(RuntimeError):
                R.summarize(run)


if __name__ == '__main__':
    unittest.main()
