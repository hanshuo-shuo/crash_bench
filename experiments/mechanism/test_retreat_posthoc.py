import unittest

import census
import retreat_posthoc as P


def step(closed=False):
    half = .01 if closed else .04
    return dict(eef=[0., 0., 1.], gripper=[half, -half])


def census_run(sw, pair, init, phase, arm, b, gap=1.):
    return dict(sw=sw, pair=pair, init=init, phase=phase, arm=arm, b_done=b, gap_first=gap)


def run(sw, arm, task, other, b, a=False, same=False, gap=.9, reached=True, moved=None, after=None):
    return dict(id=sw + '__' + arm, sw=sw, arm=arm, pair='%d->%d' % (task, other), task=task, other=other, init=0, phase=.5, b_done=b,
                a_done=a, scripted_steps=2, reached=reached, gap_first=gap, moved_b=moved, covariates=dict(same_object=same), b_after=after)


class Pure(unittest.TestCase):
    def test_threshold_matches_census(self):
        self.assertEqual(P.CLOSED, census.CLOSED)

    def test_longest_closed(self):
        self.assertEqual(P.longest_closed([step(), step(True), step(True), step(), step(True)]), 2)
        self.assertEqual(P.longest_closed([]), 0)

    def test_auc(self):
        self.assertEqual(P.auc([2, 3], [1]), 1.)
        self.assertEqual(P.auc([1], [1]), .5)
        self.assertIsNone(P.auc([], [1]))

    def test_behaviour_reads_only_the_policy_part(self):
        scripted = [step(), step(True), step(True)]                       # switch step and two scripted steps
        r = run('s', 'home', 3, 1, False)
        self.assertEqual(P.behaviour(r, scripted + [step(True)] * 4), 'never_closed')
        self.assertEqual(P.behaviour(r, scripted + [step(True)] * 5), 'closed_neither')
        self.assertEqual(P.behaviour(dict(r, a_done=True), scripted), 'finished_a')


class Tables(unittest.TestCase):
    def setUp(self):
        self.census = [census_run('x', '3->1', 0, .5, 'native', True), census_run('x', '3->1', 0, .5, 'amp2', True),
                       census_run('y', '2->1', 0, .5, 'native', False), census_run('y', '2->1', 0, .5, 'amp2', False),
                       census_run('x0', '3->1', 0, 0, 'native', True, gap=2.), census_run('y0', '2->1', 0, 0, 'native', True, gap=1.)]
        self.runs = [run('x', 'home', 3, 1, False, same=True, gap=1.), run('x', 'release', 3, 1, True, same=True),
                     run('y', 'home', 2, 1, True, gap=.9), run('y', 'release', 2, 1, False)]
        self.traces = {'x__home': [step()] * 3 + [step(True)] * 6}

    def test_exception_and_rest(self):
        t = P.tables(self.runs, self.census, self.traces.__getitem__, old=3)
        self.assertEqual(t['exception']['home'], dict(done=0, n=1))
        self.assertEqual(t['exception']['by_pair']['3->1']['native'], dict(done=1, n=1))
        self.assertEqual(t['exception']['by_pair']['3->1']['start'], dict(done=1, n=1))
        self.assertEqual(t['exception']['behaviour'], dict(closed_neither=1))
        self.assertEqual(t['without']['home'], dict(done=1, n=1))
        self.assertEqual(t['worse'], dict(n=1, by_pair=[('3->1', 1)]))
        self.assertEqual(t['outcomes']['home'], dict(neither=1, B=1))

    def test_gamma_uses_the_start_gap_of_the_same_pair_and_init(self):
        t = P.tables(self.runs, self.census, self.traces.__getitem__, old=3)
        self.assertEqual(t['gamma_auc']['all'], dict(n=2, failures=1, auc=1.))          # .9 / 1 above 1 / 2
        self.assertIsNone(t['gamma_auc']['without']['all']['auc'])


class Teacher(unittest.TestCase):
    def setUp(self):
        native = lambda sw, pair, b, after=None: dict(census_run(sw, pair, 0, .5, 'native', b), b_after=after)
        self.census = [native('x', '3->1', True, 10), native('y', '2->1', True, 30), native('z', '2->1', False),
                       dict(census_run('y0', '2->1', 0, 0, 'native', True), b_after=1)]            # f = 0 is not a mid-task state
        self.runs = [run('x', 'home', 3, 1, False, same=True), run('x', 'release', 3, 1, True, same=True, after=5),
                     run('y', 'home', 2, 1, True, after=20), run('y', 'release', 2, 1, True, after=40),
                     run('z', 'home', 2, 1, False), run('z', 'release', 2, 1, False)]

    def test_retreat_first_then_a_stored_success_without_it(self):
        t = P.teacher(self.runs, self.census)
        self.assertEqual(t['picked'], dict(native=1, home=1, none=1))                 # x: home failed, native before release
        self.assertEqual((t['states'], t['with_teacher']), (3, 2))
        self.assertEqual(t['by_old_task'], {2: dict(home=1, none=1), 3: dict(native=1)})
        self.assertEqual(t['same_object'], dict(native=1))
        self.assertEqual(t['pairs'], dict(n=2, any=2, most=1))

    def test_steps_count_the_script(self):
        t = P.teacher(self.runs, self.census)
        self.assertEqual(P.total(self.runs[2]), 22)
        self.assertEqual(t['steps'], dict(home=dict(n=1, median=22), native=dict(n=2, median=20.), release=dict(n=2, median=24.5)))
        self.assertEqual(t['paired']['native_home'], dict(n=1, native=30, home=22, native_faster=0))
        self.assertEqual(t['fastest'], dict(picked=dict(release=1, home=1, none=1), median=14.5))    # x: release 7 before native 10
        self.assertEqual(t['steps_picked'], 16.)


if __name__ == '__main__':
    unittest.main()
