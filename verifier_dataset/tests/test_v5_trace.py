import unittest

from verifier_dataset import v5_trace


def obj(x, y, half=.02):
    return dict(lo=[x - half, y - half, .9], hi=[x + half, y + half, 1.], center=[x, y, .95], active=True)


class Rows(unittest.TestCase):
    def trace(self, path):
        objects = dict(bowl_1=obj(0, 0), plate_1=obj(.4, 0, half=.08), cookies_1=obj(.2, .3), ramekin_1=obj(.2, -.3),
                       wooden_cabinet_1=obj(-.3, .3))
        snap = lambda step, x, z: dict(step=step, objects=objects, surface=.9, material=[x, 0, z],
                                       goal=dict(lo=[.32, -.08, .9], hi=[.48, .08, 1.]))
        return dict(source='s', family='f', task='t', student_task='put the bowl on the plate', target='bowl_1',
                    receiver='plate_1', actions=20, checkpoints=[snap(0, 0., .93), snap(5, .3, 1.2)], path=path)

    def test_checkpoint_time_labels_lift_and_witness(self):
        path = [[s, x, 0, 1.] for s, x in [(1, .05), (3, .15), (5, .3), (7, .35), (9, .4)]]
        rows = {(r['unit'], r['rule']): r for r in v5_trace.rows_from_trace(self.trace(path))}
        wall = 'line:cookies|ramekin'
        self.assertEqual(rows[(0, wall)]['label'], 1)
        self.assertTrue(rows[(0, wall)]['suffix_crosses'])
        self.assertEqual(rows[(5, wall)]['label'], 0)
        self.assertTrue(rows[(5, wall)]['witnessed'])
        self.assertAlmostEqual(rows[(5, wall)]['lift'], .3)
        self.assertAlmostEqual(rows[(0, wall)]['lift'], .03)
        self.assertEqual(rows[(5, wall)]['time_bin'], 1)
        self.assertTrue(all(r['template'] == 'line' for r in rows.values()))

    def test_robot_side_facts_only_with_a_robot_and_their_xor_is_the_label(self):
        path = [[s, x, 0, 1.] for s, x in [(1, .05), (3, .15), (5, .3), (7, .35), (9, .4)]]
        plain = v5_trace.rows_from_trace(self.trace(path))
        self.assertTrue(all('target_robot_side' not in r for r in plain))
        trace = self.trace(path)
        for snap in trace['checkpoints']:
            snap['robot'] = [-.6, 0.]
        rows = {(r['unit'], r['rule']): r for r in v5_trace.rows_from_trace(trace)}
        for r in rows.values():
            self.assertEqual(r['label'], int(r['target_robot_side'] != r['goal_robot_side']))
        wall = rows[(0, 'line:cookies|ramekin')]  # the wall x = .2; robot and bowl on one side, plate on the other
        self.assertEqual((wall['target_robot_side'], wall['goal_robot_side']), (1, 0))
        self.assertAlmostEqual(wall['robot_clearance'], .8)

    def test_a_suffix_that_recrosses_is_not_a_witness(self):
        path = [[s, x, 0, 1.] for s, x in [(1, .05), (5, .3), (7, .1), (9, .4)]]
        rows = {(r['unit'], r['rule']): r for r in v5_trace.rows_from_trace(self.trace(path))}
        self.assertFalse(rows[(5, 'line:cookies|ramekin')]['witnessed'])


if __name__ == '__main__':
    unittest.main()
