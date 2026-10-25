import math
import unittest

from verifier_dataset import v5_layouts as L


def obj(x, y, half=.03, active=True):
    return dict(lo=[x - half, y - half, .9], hi=[x + half, y + half, 1.], center=[x, y, .95], active=active)


class Sampling(unittest.TestCase):
    def scene(self):
        return dict(bowl_1=obj(0, 0), plate_1=obj(.4, 0, half=.08), cookies_1=obj(.2, .3), ramekin_1=obj(.2, -.3),
                    milk_1=obj(-.2, .25), parked_1=obj(5, 5, active=False))

    def test_constraints_hold_and_sampling_is_seeded(self):
        objects = self.scene(); path = [[x / 100, 0.] for x in range(0, 41)]
        goal = ([.32, -.08], [.48, .08])
        a = L.sample_layout(objects, ['cookies_1', 'milk_1'], path, goal, 'bowl_1', 'ab' * 32)
        self.assertEqual(a, L.sample_layout(objects, ['cookies_1', 'milk_1'], path, goal, 'bowl_1', 'ab' * 32))
        self.assertNotEqual(a['moved'], L.sample_layout(objects, ['cookies_1', 'milk_1'], path, goal, 'bowl_1', 'cd' * 32)['moved'])
        (x0, y0), (x1, y1) = a['region']
        reach = L.radius((objects['bowl_1']['lo'], objects['bowl_1']['hi']))
        for name, p in a['moved'].items():
            r = L.radius((objects[name]['lo'], objects[name]['hi']))
            self.assertTrue(x0 <= p[0] <= x1 and y0 <= p[1] <= y1)
            self.assertGreaterEqual(min(math.hypot(p[0] - q[0], p[1] - q[1]) for q in path), r + reach + L.PATH_GAP)
            self.assertGreaterEqual(L.rect_distance(p, *goal), r + L.GOAL_GAP)
            for other in ('bowl_1', 'plate_1', 'ramekin_1'):
                c = objects[other]['center']
                self.assertGreaterEqual(math.hypot(p[0] - c[0], p[1] - c[1]), r + L.radius((objects[other]['lo'], objects[other]['hi'])) + L.OBJECT_GAP)

    def test_anchor_with_no_room_stays(self):
        objects = dict(bowl_1=obj(0, 0), cookies_1=obj(.05, .05))
        path = [[x / 50, y / 50] for x in range(-6, 9) for y in range(-6, 9)]  # the target sweeps the whole region
        out = L.sample_layout(objects, ['cookies_1'], path, ([.0, .0], [.0, .0]), 'bowl_1', 'ef' * 32)
        self.assertEqual((out['moved'], out['stayed']), ({}, ['cookies_1']))

    def test_thin_keeps_one_point_per_centimetre(self):
        self.assertEqual(L.thin([[0, 0, 1], [.004, 0, 1], [.012, 0, 1], [.015, 0, 1]]), [[0, 0], [.012, 0]])


class Design(unittest.TestCase):
    def test_design_keeps_the_first_accepted_layout_or_reports_failure(self):
        objects = dict(bowl_1=obj(0, 0), plate_1=obj(.4, 0, half=.08), cookies_1=obj(.2, .3), milk_1=obj(-.2, .3))
        path, goal = [[x / 100, 0.] for x in range(41)], ([.32, -.08], [.48, .08])
        below = lambda lay: lay['moved'].get('cookies_1', [0, 1])[1] < 0
        out = L.design_layout(objects, ['cookies_1'], path, goal, 'bowl_1', 'ab' * 32, below)
        self.assertTrue(out['designed'] and below(out))
        never = L.design_layout(objects, ['cookies_1'], path, goal, 'bowl_1', 'ab' * 32, lambda lay: False)
        self.assertEqual((never['designed'], never['attempts']), (False, L.DESIGN_ATTEMPTS))

    def test_resting_twin_needs_a_moved_anchor_rule_of_each_label(self):
        objects = dict(bowl_1=obj(0, 0), plate_1=obj(.4, 0, half=.05), cookies_1=obj(.2, .3), ramekin_1=obj(.2, -.3),
                       milk_1=obj(-.3, .3))
        start, goal = L.g.point([0., 0.]), ([.35, -.05], [.45, .05])
        names = ['cookies_1', 'milk_1', 'ramekin_1']
        # cookies at (.2, .3): its line to the ramekin separates; its line to the milk carton does not.
        self.assertTrue(L.resting_twin(objects, {}, dict(moved={'cookies_1': [.2, .3]}), names, start, goal, 0.))
        self.assertFalse(L.resting_twin(objects, {}, dict(moved={'milk_1': [-.3, .3]}), ['milk_1', 'cookies_1'], start, goal, 0.))


class Screen(unittest.TestCase):
    def test_shortcut_transfer_and_capability(self):
        base = dict(task='t|x', template='line', clearance=.1, lift=.01, family='f')
        natural = [dict(base, source='a', unit=0, rule='line:p|q', label=1, time_bin=0),
                   dict(base, source='a', unit=5, rule='line:p|r', label=0, time_bin=0),
                   dict(base, source='z', unit=0, rule='line:p|q', label=0, time_bin=0)]  # validation source: not in the table
        cf = lambda src, k, rule, label, tb=0: dict(base, source='%s#%d' % (src, k), base_source=src, unit=0, rule=rule,
                                                    label=label, time_bin=tb)
        counterfactual = [cf('b', 0, 'line:p|q', 0), cf('b', 0, 'line:p|r', 1),       # twin; the table gets both wrong
                          cf('c', 0, 'line:p|q', 1), cf('c', 0, 'line:p|s', 0, 3),    # twin; one right, one from the global fallback
                          cf('a', 0, 'line:p|q', 0), cf('a', 0, 'line:p|r', 1)]       # training source: excluded
        split = dict(a='train', b='test', c='validation', z='validation')
        out = L.shortcut_transfer(natural, counterfactual, split)
        self.assertEqual((out['rows'], out['training_rows'], out['matched_text_time']), (4, 2, 3))
        self.assertAlmostEqual(out['accuracy'], .25)
        cap = L.capability([], counterfactual, split)
        self.assertEqual(cap['capable_sources'], ['a', 'b', 'c'])
        self.assertEqual(cap['capable_by_split'], {'train': 1, 'validation': 1, 'test': 1})


class S1(unittest.TestCase):
    def rows(self):
        base = dict(task='t|x', template='line', clearance=.1, lift=.01, family='f', time_bin=0, robot_clearance=.3)
        out = []
        for src, split_rules in (('v1', ((0, 1), (1, 0))), ('v2', ((0, 1), (1, 0))), ('t1', ((0, 1), (1, 0)))):
            for k, (a, b) in enumerate(split_rules):
                for rule, label in (('line:p|q', a), ('line:p|r', b)):
                    out.append(dict(base, source='%s#%d' % (src, k), base_source=src, layout=k, unit=0, rule=rule, label=label,
                                    target_robot_side=1, goal_robot_side=1 - label))
        out.append(dict(out[0], rule='line:p|s', robot_clearance=.05))  # robot base too close: unusable
        out.append(dict(out[0], unit=5, lift=.2))                       # lifted: not a resting twin
        return out

    def test_plan_takes_eight_layouts_of_every_object_source(self):
        import json, tempfile
        from pathlib import Path
        cases = [dict(source='o%d' % i, task=L.S1_TASK, family='safelibero_object') for i in range(3)] + \
                [dict(source='g0', task='safelibero_goal/I/task1', family='safelibero_goal')]
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'CASES.json').write_text(json.dumps(cases))
            entries = L.plan(Path(tmp), 's1')
        self.assertEqual(len(entries), 3 * L.S1_LAYOUTS)
        self.assertEqual({e['spec']['source'] for e in entries}, {'o0', 'o1', 'o2'})
        self.assertEqual(entries[0]['plan']['seed'], L.g.digest('o0', 0, 'v5-layout-s1'))

    def test_supply_counts_strict_twins_per_split(self):
        split = dict(v1='validation', v2='validation', t1='test')
        out = L.s1_supply(self.rows(), split)
        self.assertEqual((out['validation']['strict_twin_units'], out['validation']['strict_twin_sources']), (4, 2))
        self.assertEqual(out['test']['strict_twin_units'], 2)
        self.assertEqual(out['robot_near_rows'], 1)
        self.assertEqual(out['s1_reading'], 's1_supply_low')

    def test_review_units_come_from_validation_sources_round_robin(self):
        split = dict(v1='validation', v2='validation', t1='test')
        units = L.select_review_units(self.rows(), split, n=3)
        self.assertEqual(len(units), 3)
        self.assertEqual({u['base_source'] for u in units}, {'v1', 'v2'})
        self.assertEqual(len({u['base_source'] for u in units[:2]}), 2)
        for u in units:
            self.assertEqual([r['label'] for r in u['rules']], [1, 0])
            self.assertNotIn('line:p|s', u['candidate_rules'])
        self.assertEqual(units, L.select_review_units(self.rows(), split, n=3))


class Stacks(unittest.TestCase):
    def test_obstacle_moves_with_its_base_and_blocked_stacks_stay(self):
        objects = dict(bowl_1=obj(0, 0), moka_pot_obstacle_1=obj(.3, .2), box_base_1=obj(.3, .2, half=.05),
                       cookies_1=obj(-.2, .2), bowl_2=obj(-.2, .2), milk_1=obj(.1, -.3), plate_1=obj(.4, -.1))
        owner = {1: 'moka_pot_obstacle_1', 2: 'box_base_1', 3: 'cookies_1', 4: 'bowl_2', 5: 'milk_1', 6: 'plate_1'}
        table, finger = 100, 200
        contacts = [(1, 2), (2, table), (3, table), (3, 4), (5, table), (5, finger), (6, table)]
        stacks = L.movable_stacks(objects, contacts, owner, {table}, {finger}, free_joint=lambda n: True,
                                  anchors=['moka_pot_obstacle_1', 'cookies_1', 'milk_1'], fixed={'bowl_1', 'plate_1'})
        self.assertEqual(stacks, {'moka_pot_obstacle_1': ['box_base_1', 'moka_pot_obstacle_1'], 'cookies_1': ['bowl_2', 'cookies_1']})
        merged = L.merge_stacks(objects, stacks)
        self.assertNotIn('box_base_1', merged)
        for got, want in zip(merged['moka_pot_obstacle_1']['lo'][:2], [.25, .15]):
            self.assertAlmostEqual(got, want)
        self.assertEqual(L.movable_stacks(objects, contacts, owner, {table}, {finger}, free_joint=lambda n: n != 'box_base_1',
                                          anchors=['moka_pot_obstacle_1'], fixed=set()), {})


class Readouts(unittest.TestCase):
    def test_resting_twins_validity_and_reading(self):
        layouts = [dict(valid=True, success=True, violations=[], moved={'a': 1}, maximum_path_deviation=.001),
                   dict(valid=False, success=True, violations=[dict(kind='anchor_drift')], moved={}, maximum_path_deviation=None)]
        base = dict(task='t|x', template='line', clearance=.1, time_bin=0, base_source='s', lift=.01, family='f')
        rows = [dict(base, source='s#0', unit=0, rule='line:a|b', label=1), dict(base, source='s#0', unit=0, rule='line:a|c', label=0),
                dict(base, source='s#1', unit=0, rule='line:a|b', label=0), dict(base, source='s#1', unit=0, rule='line:a|c', label=1)]
        out = L.readouts(layouts, rows)
        self.assertEqual((out['validity'], out['resting_twins'], out['text_only_bound']), (.5, 2, .5))
        self.assertEqual(out['invalid_reasons'], {'anchor_drift': 1})
        self.assertEqual(out['families_with_twins'], 1)
        self.assertEqual(out['reading'], 'layout_design_insufficient')  # twins from one source and one family only


if __name__ == '__main__':
    unittest.main()
