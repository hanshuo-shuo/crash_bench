import unittest

import census
import census_reach_posthoc as R


def step(x, closed=False, action=None):
    half = .01 if closed else .04
    return dict(eef=[x, 0., 1.], gripper=[half, -half], action=action)


def run(pair, init, phase, arm, b=False, a=False, same=False):
    return dict(id='%s_%s_%s_%s' % (pair, init, phase, arm), pair=pair, init=init, phase=phase, arm=arm, b_done=b, a_done=a,
                covariates=dict(same_object=same))


class Pure(unittest.TestCase):
    def test_threshold_matches_census(self):
        self.assertEqual(R.CLOSED, census.CLOSED)

    def test_work_point_is_first_closed_step(self):
        self.assertEqual(R.work_point([step(0), step(.3, True), step(.5, True)]), [.3, 0., 1.])
        self.assertIsNone(R.work_point([step(0), step(.3)]))

    def test_closest(self):
        self.assertAlmostEqual(R.closest([step(0), step(.2), step(.4)], [.25, 0., 1.]), .05)

    def test_stalled_needs_a_full_still_tail(self):
        still = [step(0)] * (R.TAIL + 1)
        self.assertTrue(R.stalled(still))
        self.assertFalse(R.stalled(still[:-1]))                                    # too short to judge
        self.assertFalse(R.stalled([step(i * .001) for i in range(R.TAIL + 1)]))   # 5 cm of travel

    def test_outside_splits_arm_and_gripper(self):
        steps = [step(0), step(0, action=[0] * 6 + [1.0008]), step(0, action=[0, 0, 1.2, 0, 0, 0, 1.5]), step(0, action=[.5] * 7)]
        self.assertEqual(R.outside(steps), (3, 1, 1, 1.5))


class Tables(unittest.TestCase):
    def setUp(self):
        go = [step(0), step(.2), step(.4)]; stay = [step(0)] * (R.TAIL + 2)
        self.traces = {
            '0->1_0_0_native': [step(0), step(.4, True)],                 # reference: B's work point at x = .4
            '0->1_0_0.5_native': stay, '0->1_0_0.5_amp2': go,             # A -> neither, amp2 reaches B
            '0->1_1_0_native': [step(0), step(.4, True)],
            '0->1_1_0.5_native': [step(.35), step(.4)], '0->1_1_0.5_amp2': go,   # switch step already near B: dropped
            '0->2_0_0_native': [step(0), step(.4)],                       # reference never closes: dropped
            '0->2_0_0.5_native': stay, '0->2_0_0.5_amp2': go,
            '1->3_0_0_native': [step(0), step(.4, True)],
            '1->3_0_0.5_native': stay, '1->3_0_0.5_amp2': go}             # same object: dropped
        self.rows = [run('0->1', 0, 0, 'native', b=True), run('0->1', 0, .5, 'native', a=True), run('0->1', 0, .5, 'amp2'),
                     run('0->1', 1, 0, 'native', b=True), run('0->1', 1, .5, 'native', a=True), run('0->1', 1, .5, 'amp2'),
                     run('0->2', 0, 0, 'native', b=True), run('0->2', 0, .5, 'native', a=True), run('0->2', 0, .5, 'amp2'),
                     run('1->3', 0, 0, 'native', b=True, same=True), run('1->3', 0, .5, 'native', a=True, same=True),
                     run('1->3', 0, .5, 'amp2', same=True)]

    def test_counts(self):
        out = R.tables(self.rows, self.traces.__getitem__)
        self.assertEqual(out['states_read'], 1)
        self.assertEqual(out['dropped'], {'reference never closes the gripper': 1, 'switch step already near B work point': 1,
                                          'no B reference from the start': 1, 'same-object pair': 1})
        native = out['by_outcome']['native A']; amp = out['by_outcome']['amp2 A->neither']
        self.assertEqual((native['runs'], native['reach_5'], native['stalled']), (1, 0, 1))
        self.assertEqual((amp['runs'], amp['reach_3'], amp['reach_5'], amp['stalled'], amp['reach_and_stalled']), (1, 1, 1, 0, 0))


if __name__ == '__main__':
    unittest.main()
