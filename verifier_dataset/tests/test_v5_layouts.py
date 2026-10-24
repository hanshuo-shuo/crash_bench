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
