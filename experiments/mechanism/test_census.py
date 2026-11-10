"""Unit tests for census.py (numpy only): python -m unittest test_census"""
import gzip
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import census as K
import closed_loop as C
from verifier_dataset import hil_suite as S
from verifier_dataset import hil_suite_run as H

K.DRAWS = 300   # the readings' bootstrap at test size (2000 in the summary)

REPO = Path(__file__).resolve().parents[2]
PROMPTS = ['open the middle drawer of the cabinet', 'put the bowl on the stove', 'put the wine bottle on top of the cabinet',
           'open the top drawer and put the bowl inside', 'put the bowl on top of the cabinet', 'push the plate to the front of the stove',
           'put the cream cheese in the bowl', 'turn on the stove', 'put the bowl on the plate', 'put the wine bottle on the rack']
NAMES = {K.SUITE: ['KITCHEN_SCENE1_%s' % p.replace(' ', '_') for p in PROMPTS]}


def prefixes(failed=(), steps=80):
    out = []
    for p in K.prefix_plan(NAMES):
        ok = (p['task'], p['init']) not in failed
        out.append(dict(p, success=ok, policy_steps=steps if ok else None, steps=S.WAIT + (steps if ok else S.SUITES[K.SUITE]), start_eef=[0., 0., 0.]))
    return out


def tape(p, holds=()):
    """A prefix tape: open gripper until step 60, closed (aperture .02) after; B goals false unless (step, B) in holds."""
    n = p['steps']
    return [dict(step=t, physics='%s_h%d' % (p['id'], t), eef=[0., 0., 0.], gripper=[.04, -.04] if t < 60 else [.01, -.01],
                 a=p['success'] and t == n, others={str(b): (t, b) in holds for b in p['others']}, done=p['success'] and t == n) for t in range(n + 1)]


class Plan(unittest.TestCase):
    def test_moved_objects_match_the_bddl_goals(self):
        from verifier_dataset.hil_suite import task_names, language
        names = task_names((REPO/'third_party/LIBERO/libero/libero/benchmark/libero_suite_task_map.py').read_text())[K.SUITE]
        word = dict(akita_black_bowl_1='bowl', wine_bottle_1='wine_bottle', plate_1='plate', cream_cheese_1='cream_cheese',
                    wooden_cabinet_1_middle_region='cabinet_middle_drawer', flat_stove_1='stove')
        for k, n in enumerate(names):
            goal = (REPO/'third_party/LIBERO/libero/libero/bddl_files/libero_goal'/(n + '.bddl')).read_text().split('(:goal')[1]
            first = re.findall(r'\((?:On|In|Open|Turnon)\s+([^\s\)]+)', goal)[0]
            self.assertEqual(K.MOVED[k], word[first], n)
            self.assertEqual(language(n), PROMPTS[k])

    def test_prefix_plan(self):
        p = K.prefix_plan(NAMES)
        self.assertEqual(len(p), 100); self.assertEqual(len({x['id'] for x in p}), 100)
        for x in p:
            self.assertEqual(x['id'], S.prefix_id(K.SUITE, x['task'], x['init'])); self.assertEqual(x['seed'], S.policy_seed(K.SUITE, x['task'], x['init']))
            self.assertEqual(x['others'], [b for b in range(10) if b != x['task']]); self.assertEqual(x['prompt'], PROMPTS[x['task']])
            self.assertEqual(x['max_steps'], 300)

    def test_branch_units_counts_exclusions_and_covariates(self):
        ps = prefixes(failed={(3, 4)})
        holds = {'libero_goal_t8_i02': {(S.checkpoint(next(p for p in ps if p['id'] == 'libero_goal_t8_i02'), .75), 6)}}
        tapes = {p['id']: tape(p, holds.get(p['id'], ())) for p in ps}
        units, excluded = K.branch_units(ps, tapes, 'formal')
        self.assertEqual(len(units), 4500 - 45 - 2)
        self.assertEqual(sum(e['reason'] == 'prefix task failed' for e in excluded), 9)
        self.assertEqual([e for e in excluded if e['reason'] != 'prefix task failed'],
                         [dict(prefix='libero_goal_t8_i02', other=6, phase=.75, reason='B holds at the switch step')])
        self.assertEqual(len({u['id'] for u in units}), len(units))
        for u in units:
            p = next(q for q in ps if q['id'] == u['prefix'])
            self.assertEqual(u['checkpoint'], S.checkpoint(p, u['phase'])); self.assertEqual(u['seed'], p['seed'])
            self.assertEqual(u['arm']['name'] in K.PHASE_ARMS[u['phase']], True); self.assertEqual(u['prompt'], PROMPTS[u['other']])
            self.assertEqual(u['arm']['prompt_a'], PROMPTS[u['task']]); self.assertEqual(u['record'], [u['other']])
            self.assertEqual(u['covariates']['gripper_closed'], u['checkpoint'] >= 60)
            self.assertEqual(u['covariates']['same_object'], K.MOVED[u['task']] == K.MOVED[u['other']])
            self.assertEqual(u['id'], '%s_s%03d_SW_b%d__%s' % (u['prefix'], u['checkpoint'], u['other'], u['arm']['name']))
        self.assertEqual({u['checkpoint'] for u in units}, {10, 50, 70})
        smoke, _ = K.branch_units(ps, tapes, 'smoke')
        self.assertEqual(len(smoke), 50); self.assertEqual({u['init'] for u in smoke}, {0})
        self.assertEqual({(u['task'], u['other']) for u in smoke}, {(a, (a + 3) % 10) for a in range(10)})

    def test_arms_are_the_servers_valid_arms(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'feasibility'))
        import serve_steer
        ps = prefixes(); units, _ = K.branch_units(ps, {p['id']: tape(p) for p in ps}, 'smoke')
        for u in units:
            serve_steer.check_arm(json.loads(json.dumps(u['arm'])), 18)

    def fake_prefix_root(self, root, reading='env_ok'):
        ps = prefixes()
        for p in ps:
            folder = root/'prefix'/p['id']; folder.mkdir(parents=True); H.save(folder, tape(p), p)
        C.write(root/'PLAN.json', dict(mode='formal', shards=2, prefixes=K.prefix_plan(NAMES)))
        C.write(root/'PREFIXES.json', ps); C.write(root/'STAGE0.json', dict(reading=reading))
        C.write(root/'PREFIX_COMPLETE.json', dict(passed=True, mode='formal'))
        C.write(root/'PREFIX_HASHES.json', {n: C.sha(root/n) for n in ('PLAN.json', 'PREFIXES.json', 'STAGE0.json', 'PREFIX_COMPLETE.json')})

    def test_write_plans_and_refusals(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); pre = tmp/'prefix_root'; self.fake_prefix_root(pre)
            stage1 = tmp/'stage1'; stage1.mkdir(); (stage1/'HASHES.json').write_text('{}')
            smoke, formal = tmp/'smoke', tmp/'formal'
            for r in (smoke, formal):
                r.mkdir(); C.write(r/'CAMPAIGN.json', {})
            K.write_plans(pre, smoke, formal, stage1)
            s, f = C.read(smoke/'PLAN.json'), C.read(formal/'PLAN.json')
            self.assertEqual((s['mode'], s['shards'], len(s['branches'])), ('smoke', 2, 50))
            self.assertEqual((f['mode'], f['shards'], len(f['branches'])), ('formal', 6, 4500))
            self.assertEqual(len(f['prefix_order']), 100); self.assertEqual(len(s['prefix_order']), 10)
            self.assertEqual(f['prefix_hashes'], C.read(pre/'PREFIX_HASHES.json'))
            with self.assertRaises(RuntimeError):
                K.write_plans(pre, smoke, formal, stage1)
            C.write(pre/'STAGE0.json', dict(reading='env_suspect'))
            with self.assertRaises(RuntimeError):
                K.plan(pre, 'smoke', stage1)


# ---------------------------------------------------------------- readings on synthetic runs
def noise(*keys):
    return (int.from_bytes(__import__('hashlib').sha256(repr(keys).encode()).digest()[:4], 'big') % 10000) / 10000.


def world(native_rate, amp_rate=lambda a, b, f, cov: .9, gap=lambda a, b, f, i: 1., inits=range(10)):
    """Synthetic run rows for every census state: native / amp2 b_done drawn from the given rates (deterministic noise)."""
    out = []
    for a in range(10):
        for b in range(10):
            if a == b:
                continue
            for i in inits:
                for f in K.PHASES:
                    cov = dict(aperture=.02 if (i + a) % 2 else .08, gripper_closed=bool((i + a) % 2) and f > 0, same_object=K.MOVED[a] == K.MOVED[b])
                    sw = 'libero_goal_t%d_i%02d_s%03d_SW_b%d' % (a, i, 10 + int(80 * f), b)
                    for arm in K.PHASE_ARMS[f]:
                        rate = native_rate(a, b, f, cov) if arm == 'native' else amp_rate(a, b, f, cov)
                        done = noise(sw, arm) < rate
                        out.append(dict(sw=sw, pair='%d->%d' % (a, b), task=a, other=b, init=i, phase=f, arm=arm, kind=K.ARMS[arm][0],
                                        b_done=done, a_done=not done, covariates=cov, gap_first=gap(a, b, f, i) * .3))
    return out


FADING = lambda a, b: (a + 2 * b) % 3 == 0     # a third of the pairs


class Readings(unittest.TestCase):
    def test_fade_world(self):
        """A third of the pairs fade and fail late; amp2 repairs them; physical covariates are noise."""
        native = lambda a, b, f, cov: .1 if f and FADING(a, b) else .97
        gap = lambda a, b, f, i: (.45 + .1 * noise(a, b, f, i) if FADING(a, b) else .95 + .1 * noise(a, b, f, i)) if f else 1.
        r = K.readings(world(native, lambda a, b, f, cov: .85 if f and FADING(a, b) else .97, gap))
        self.assertEqual(r['prevalence']['reading'], 'common')
        self.assertEqual(r['restoration']['reading'], 'restores_switch'); self.assertEqual(r['selectivity']['reading'], 'selective')
        self.assertEqual(r['explanation']['across_cells']['reading'], 'fade_tracks')
        self.assertEqual(r['explanation']['beyond_physics']['reading'], 'beyond_physics')
        rho = r['reported']['rho']['0.75']; self.assertLess(rho['rho'], .8)
        g = r['restoration']['gain']; self.assertGreater(g['pairs'], 20)
        self.assertFalse({k.split(' ')[0] for k, v in r['cells'].items() if v['cls'] == 'failure'} & set(K.PILOT))

    def test_physics_world(self):
        """Failure follows the need to let go of another object; the image gap is noise."""
        native = lambda a, b, f, cov: .1 if f and cov['gripper_closed'] and not cov['same_object'] else .97
        gap = lambda a, b, f, i: (.5 + noise('g', a, b, f, i)) if f else 1.
        r = K.readings(world(native, lambda a, b, f, cov: .97, gap))
        self.assertEqual(r['explanation']['beyond_physics']['reading'], 'physics_explains')
        self.assertIn(r['explanation']['across_cells']['reading'], ('unclear', 'reversed', 'fade_tracks'))

    def test_rare_failures_damage_and_no_restore(self):
        native = lambda a, b, f, cov: .97
        r = K.readings(world(native, lambda a, b, f, cov: .5))
        self.assertEqual(r['prevalence']['reading'], 'rare'); self.assertEqual(r['restoration']['reading'], 'insufficient')
        self.assertEqual(r['selectivity']['reading'], 'damages')
        native = lambda a, b, f, cov: .2 if f == .75 and FADING(a, b) else .97
        r = K.readings(world(native, lambda a, b, f, cov: .2 if f == .75 and FADING(a, b) else .97))
        self.assertEqual(r['restoration']['reading'], 'no_restore'); self.assertEqual(r['prevalence']['reading'], 'common')

    def test_ineligible_and_thin_cells(self):
        native = lambda a, b, f, cov: .1 if b == 5 else .97     # B = 5 fails from the start: its pairs are ineligible
        rows = [r for r in world(native) if not (r['task'] == 0 and r['other'] == 1 and r['phase'] == .5 and r['init'] > 2)]
        r = K.readings(rows)
        self.assertEqual(r['cells']['0->1 f=0.5']['cls'], 'thin'); self.assertEqual(r['cells']['0->5 f=0.75']['cls'], 'ineligible')
        self.assertEqual(r['reported']['eligible_pairs'], 81); self.assertEqual(r['reported']['classes']['thin'], 1)

    def test_paired_diff_and_logistic(self):
        rows = [dict(pair='p%d' % k, phase=.5, x=1, y=int(j % 2 == 0)) for k in range(10) for j in range(10)]
        d = K.paired_diff(rows, 'x', 'y', {('p%d' % k, .5) for k in range(10)})
        self.assertAlmostEqual(d['mean'], .5); self.assertEqual(d['pairs'], 10); self.assertLessEqual(d['lo'], .5); self.assertGreaterEqual(d['hi'], .5)
        rng = np.random.default_rng(0); x = rng.normal(size=500); y = (rng.random(500) < 1 / (1 + np.exp(-(.5 + 2 * x)))).astype(float)
        w = K.logistic(np.stack([np.ones(500), x], 1), y, ridge=0.)
        self.assertAlmostEqual(w[1], 2., delta=.5); self.assertAlmostEqual(w[0], .5, delta=.4)
        self.assertEqual(K.auc([1, 2, 3], [0, 1, 1]), 1.); self.assertEqual(K.spearman([1, 2, 3, 4], [2, 4, 6, 9]), 1.)
        self.assertIsNone(K.spearman([1, 1, 1], [1, 2, 3]))

    def test_tool_reading(self):
        good = dict(physics_equal=True, fidelity_linf=.002, max_delta=0., first_shift_linf=0., scale_error=0., prompt_ok=True, kind='native',
                    b_at_checkpoint=False)
        amp = dict(good, kind='amp', max_delta=1., first_shift_linf=.1, scale_error=1e-6)
        check = dict(passed=True, runs=2, cross_checks=2)
        self.assertEqual(K.tool_reading([good, amp], check)[0], 'tool_ok')
        for bad in (dict(good, physics_equal=False), dict(good, max_delta=1.), dict(amp, first_shift_linf=0.), dict(amp, scale_error=.01),
                    dict(good, b_at_checkpoint=True), dict(good, prompt_ok=False), dict(good, fidelity_linf=.2)):
            self.assertEqual(K.tool_reading([good, amp, bad], check)[0], 'tool_failed', bad)
        self.assertEqual(K.tool_reading([good, amp], {})[0], 'tool_failed')
        self.assertEqual(K.tool_reading([good, amp], dict(check, passed=False))[0], 'tool_failed')


class Summary(unittest.TestCase):
    """run_row and summarize on a fake root: prefix t1 / init 0, B = 8, every phase and arm."""

    def build(self, root, mode='smoke', check=True):
        pre = root/'prefix_root'; every = prefixes(); ps = [p for p in every if p['id'] == 'libero_goal_t1_i00']
        for p in ps:
            folder = pre/'prefix'/p['id']; folder.mkdir(parents=True); H.save(folder, tape(p), p)
        C.write(pre/'PREFIXES.json', ps); hashes = {'PREFIXES.json': C.sha(pre/'PREFIXES.json')}
        units, _ = K.branch_units(every, {p['id']: tape(p) for p in every}, 'formal')
        units = [u for u in units if u['prefix'] == 'libero_goal_t1_i00' and u['other'] == 8]
        for u in units:
            folder = root/'run'/'branch'/u['id']; folder.mkdir(parents=True)
            done = u['arm']['name'] == 'amp2' or u['phase'] == 0
            trace = [dict(step=u['checkpoint'] + k, eef=[0., 0., 0.], a=False, others={'8': done and k == 20}, done=False, physics='x') for k in range(21)]
            scale = u['arm']['lam'] - 1
            requests = [dict(request_index=i + 1, input_sha256='in%d' % i, rng_before='r%d' % i, rng_after='r%d' % (i + 1),
                             native_chunk_sha256='c%d' % i, output_replayed=False, native_output_linf_difference=0.) for i in range(4)]
            gap = 1. if u['phase'] == 0 else .5
            log = [dict(request_index=i + 1, input_sha256='in%d' % i, rng_before='r%d' % i, rng_after='r%d' % (i + 1), chunk_sha256='c%d' % i,
                        delta_k=2. * scale, diff_k=2., norm_k=4., delta_v=gap * scale, diff_v=gap, norm_v=2., seconds=.1) for i in range(4)]
            log[0].update(fidelity_linf=.001, first_shift_linf=0. if scale == 0 else .05)
            run_id = 'job_' + u['id']; (root/'run'/'server').mkdir(exist_ok=True, parents=True)
            (root/'run'/'server'/(run_id + '.jsonl')).write_text(''.join(json.dumps(e) + '\n' for e in log))
            (folder/'POLICY.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in requests))
            row = dict(u, outcome=S.outcome(trace, u, [0., 0., 0.]), start_eef=[0., 0., 0.], rng_reset=dict(run_id=run_id, arm=u['arm']),
                       physics_at_checkpoint='libero_goal_t1_i00_h%d' % u['checkpoint'], first_policy_input_sha256='in0', actions_outside_spec=0,
                       elapsed_seconds=10.)
            H.save(folder, trace, row)
        stage1 = root/'stage1'; stage1.mkdir()
        C.write(stage1/'PLAN.json', dict(branches=[dict(id='s1', suite=K.SUITE, condition='SW', task=1, other=8, init=0, phase=.5)]))
        with gzip.open(stage1/'RUNS.jsonl.gz', 'wt') as stream:
            stream.write(json.dumps(dict(id='s1', b_done=True)) + '\n')
        (stage1/'HASHES.json').write_text('{}')
        run = root/'run'
        C.write(run/'PLAN.json', dict(mode=mode, shards=1, prefix_root=str(pre), prefix_hashes=hashes, prefix_order=ps, branches=units, excluded=[],
                                      layers=K.LAYERS, inits=[0], stage1_root=str(stage1), stage1_hashes_sha256=C.sha(stage1/'HASHES.json')))
        C.write(run/'SHARD_branch_0_COMPLETE.json', dict(passed=True))
        if check:
            C.write(run/'STATE_CHECK.json', dict(passed=True, runs=5, cross_checks=5))
        return run

    def test_summarize_and_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.build(Path(tmp))
            summary = K.summarize(run, require_tool_ok=True)
            self.assertEqual(summary['tool'], 'tool_ok'); self.assertEqual(summary['runs'], 5)
            with gzip.open(run/'RUNS.jsonl.gz', 'rt') as stream:
                got = {r['id'].split('__')[1] + str(r['phase']): r for r in map(json.loads, stream)}
            self.assertTrue(got['amp20.75']['b_done']); self.assertFalse(got['native0.75']['b_done']); self.assertTrue(got['native0']['b_done'])
            self.assertAlmostEqual(got['native0.5']['gap_first'], .25)
            self.assertEqual(summary['stage1_replication'], {'1->8 f=0.5': dict(stage1='1/1', census='0/1')})
            states = K.states([json.loads(x) for x in gzip.open(run/'RUNS.jsonl.gz', 'rt')])
            self.assertEqual(sorted(s['gamma'] for s in states if s['gamma'] is not None), [.5, .5])

    def test_missing_state_check_or_wrong_physics_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.build(Path(tmp), check=False)
            self.assertEqual(K.summarize(run)['tool'], 'tool_failed')
            with self.assertRaises(SystemExit):
                K.summarize(run, require_tool_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            run = self.build(Path(tmp)); plan = C.read(run/'PLAN.json'); u = plan['branches'][2]
            folder = run/'branch'/u['id']; row, trace = H.load(folder); row['physics_at_checkpoint'] = 'other'; H.save(folder, trace, row)
            self.assertEqual(K.summarize(run)['tool_detail']['physics_equal'], 4)
        with tempfile.TemporaryDirectory() as tmp:
            run = self.build(Path(tmp), mode='formal')
            with self.assertRaises(SystemExit):
                K.summarize(run, require_tool_ok=True)


if __name__ == '__main__':
    unittest.main()
