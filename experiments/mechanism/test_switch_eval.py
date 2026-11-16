import unittest

import switch_eval as E


def census_unit(task, other, init, phase, checkpoint):
    prefix = 'libero_goal_t%d_i%02d' % (task, init); sw = '%s_s%03d_SW_b%d' % (prefix, checkpoint, other)
    return dict(id=sw + '__native', sw=sw, group='1a', stages=[1], suite='libero_goal', task=task, other=other, init=init, phase=phase,
                checkpoint=checkpoint, condition='SW', prompt='do %d' % other, seed=5, budget=300, stop_on='B', record=[other], prefix=prefix,
                pair='%d->%d' % (task, other), arm=dict(name='native', kind='native', lam=1., layers=[12, 13, 14], prompt_a='do %d' % task, direction_seed=1),
                covariates=dict(aperture=.08, gripper_closed=False, same_object=False))


def branches():
    out = []
    for task in range(10):
        for init in (0, 9):
            for other in ((task + 1) % 10, (task + 2) % 10):
                for phase, t in ((0, 10), (.5, 60), (.75, 85)):
                    out.append(census_unit(task, other, init, phase, t))
                    if phase:
                        out.append(dict(out[-1], id=out[-1]['sw'] + '__amp2', arm=dict(out[-1]['arm'], name='amp2', kind='amp', lam=2.)))
    return out


class Units(unittest.TestCase):
    def test_test_split_is_the_held_out_old_tasks(self):
        us = E.units(branches(), 'test', E.KINDS, 'ft')
        self.assertEqual({u['task'] for u in us}, {4, 7})
        self.assertEqual({k: sum(u['kind'] == k for u in us) for k in E.KINDS}, dict(switch=16, start=8, keep=8))
        self.assertTrue(all(u['id'].endswith('__ft') and u['arm']['name'] == 'native' for u in us))
        self.assertEqual(len({u['id'] for u in us}), len(us))

    def test_dev_split_is_the_dev_layout_of_the_other_old_tasks(self):
        us = E.units(branches(), 'dev', ('switch',), 'ft')
        self.assertEqual({u['init'] for u in us}, {9}); self.assertFalse({u['task'] for u in us} & {4, 7})
        self.assertEqual(len(us), 8 * 2 * 2)

    def test_keep_unit_continues_the_old_task(self):
        keep = [u for u in E.units(branches(), 'test', ('keep',), 'base') if u['task'] == 4 and u['init'] == 0]
        self.assertEqual([(u['checkpoint'], u['phase']) for u in keep], [(60, .5), (85, .75)])
        u = keep[0]
        self.assertEqual((u['prompt'], u['stop_on'], u['other'], u['record'], u['condition'], u['pair']), ('do 4', 'A', None, [], 'C0', '4->4'))
        self.assertNotEqual(u['arm']['prompt_a'], u['prompt'])
        self.assertEqual(u['id'], 'libero_goal_t4_i00_s060_KEEP__base')


    def test_instruction_units_use_the_suites_prompts_once_per_state(self):
        us = E.units(branches(), 'test', ('stop', 'return', 'cancel'), 'ft')
        self.assertEqual({k: sum(u['kind'] == k for u in us) for k in E.INSTRUCT}, {'stop': 8, 'return': 8, 'cancel': 8})
        one = {u['kind']: u for u in us if u['task'] == 4 and u['init'] == 0 and u['phase'] == .5}
        self.assertEqual({k: (u['condition'], u['prompt']) for k, u in one.items()},
                         {'stop': ('STOP', E.S.STOP), 'return': ('RETURN', E.S.RETURN), 'cancel': ('CANCEL', 'do not do 4, stop doing it now')})
        u = one['cancel']
        self.assertEqual((u['id'], u['stop_on'], u['other'], u['pair']), ('libero_goal_t4_i00_s060_CANCEL__ft', 'A', None, '4->CANCEL'))
        self.assertNotEqual(u['arm']['prompt_a'], u['prompt'])


def state(pair, phase, init, tuned, native, home, task=4):
    return dict(pair=pair, phase=phase, init=init, task=task, other=int(pair[-1]), same_object=False, tuned=tuned, native=native, home=home,
                release=False, amp2=False, tuned_a=False, tuned_steps=100 if tuned else None, home_steps=120 if home else None, tuned_home=tuned)


class Readings(unittest.TestCase):
    def switch(self, tuned, native, home):
        return [state('4->%d' % b, f, i, tuned(i), native(i), home(i)) for b in (1, 2, 3, 5, 6) for f in (.5, .75) for i in range(10)]

    def test_learns_and_matches(self):
        s = self.switch(lambda i: i < 9, lambda i: i < 2, lambda i: i < 9)
        self.assertEqual(E.learns(s)['reading'], 'learns'); self.assertEqual(E.against_script(s)['reading'], 'matches_script')

    def test_does_not_learn_and_below(self):
        s = self.switch(lambda i: i < 2, lambda i: i < 2, lambda i: i < 9)
        self.assertEqual(E.learns(s)['reading'], 'does_not_learn'); self.assertEqual(E.against_script(s)['reading'], 'below_script')

    def test_between(self):
        s = self.switch(lambda i: i < 8, lambda i: i < 2, lambda i: i < 9)
        self.assertEqual(E.learns(s)['reading'], 'learns'); self.assertEqual(E.against_script(s)['reading'], 'between')

    def test_keeps_and_forgets(self):
        start = [dict(pair='4->1', phase=0, tuned=True, native=True) for _ in range(20)]
        keep = [dict(task=4, phase=f, tuned=True, base=True) for f in (.5, .75) for _ in range(10)]
        self.assertEqual(E.keeps(start, keep)['reading'], 'keeps')
        self.assertEqual(E.keeps(start, [dict(k, tuned=False) for k in keep])['reading'], 'forgets')
        self.assertEqual(E.keeps([dict(s, tuned=False) for s in start], keep)['reading'], 'forgets')
        self.assertEqual(E.keeps(start, [])['reading'], 'insufficient')

    def test_tables_count_per_arm(self):
        s = self.switch(lambda i: i < 7, lambda i: i < 2, lambda i: 1 <= i < 9)
        t = E.tables(s, [], [])['switch']
        self.assertEqual(t['all']['tuned'], dict(done=70, n=100)); self.assertEqual(t['all']['home'], dict(done=80, n=100))
        self.assertEqual(t['script_fails_tuned_succeeds'], 10); self.assertEqual(t['script_succeeds_tuned_fails'], 20)
        self.assertEqual(t['flow'], {'B->B': 20, 'not->B': 50, 'not->not': 30})
        self.assertEqual(t['steps']['both'], dict(n=60, tuned=100., home=120.))


class Instructions(unittest.TestCase):
    def run_(self, sw, kind, condition, **o):
        return dict(dict(id=sw + '__ft', sw=sw, unit_kind=kind, condition=condition, task=4, init=0, phase=.5, b_done=False, a_done=False, stopped=False,
                         home=False, released=True, still=True), **o)

    def test_rows_read_each_instruction_by_its_own_rule(self):
        runs = [self.run_('a', 'stop', 'STOP', stopped=True), self.run_('b', 'return', 'RETURN'), self.run_('c', 'cancel', 'CANCEL', released=False),
                self.run_('k', 'keep', 'C0', a_done=True)]
        base = [self.run_('a', 'stop', 'STOP'), self.run_('c', 'cancel', 'CANCEL', a_done=True)]
        rows = E.instruct_rows(runs, base, dict(a=True, b=True, c=False))
        self.assertEqual([(r['kind'], r['tuned'], r['base'], r['script'], r['put_down']) for r in rows],
                         [('stop', True, False, True, True), ('return', False, None, True, True), ('cancel', True, False, False, False)])

    def test_obeys(self):
        rows = [dict(kind='stop', task=t, phase=f, tuned=True, base=False, script=True, put_down=True, tuned_a=False) for t in (4, 7) for f in (.5, .75)]
        rows += [dict(kind='cancel', task=4, phase=.5, tuned=k < 2, base=False, script=True, put_down=False, tuned_a=k >= 2) for k in range(5)]
        o = E.obeys(rows)
        self.assertEqual((o['stop']['reading'], o['stop']['tuned'], o['stop']['base'], o['stop']['script']),
                         ('obeys', dict(done=4, n=4), dict(done=0, n=4), dict(done=4, n=4)))
        self.assertEqual(o['stop']['tuned_minus_base']['mean'], 1.)
        self.assertEqual((o['cancel']['reading'], o['cancel']['old_task_finished']), ('does_not_obey', dict(done=3, n=5)))
        self.assertNotIn('return', o)


class StateRows(unittest.TestCase):
    def test_rows_join_the_stored_results(self):
        cov = dict(same_object=True)
        runs = [dict(id='s__ft', sw='s', pair='4->1', task=4, other=1, init=0, phase=.5, covariates=cov, unit_kind='switch', b_done=True, a_done=False, b_after=90, home=True),
                dict(id='z__ft', sw='z', pair='4->1', task=4, other=1, init=0, phase=0, covariates=cov, unit_kind='start', b_done=False, a_done=False, b_after=None, home=False),
                dict(id='k__ft', sw='k', pair='4->4', task=4, other=None, init=0, phase=.5, covariates=cov, unit_kind='keep', b_done=False, a_done=True, b_after=None, home=False)]
        census = [dict(sw='s', arm='native', b_done=False), dict(sw='s', arm='amp2', b_done=True), dict(sw='z', arm='native', b_done=True)]
        retreat = [dict(sw='s', arm='home', b_done=True, b_after=70, scripted_steps=40), dict(sw='s', arm='release', b_done=False, b_after=None, scripted_steps=20)]
        switch, start, keep = E.state_rows(runs, census, retreat, [dict(sw='k', a_done=False)])
        self.assertEqual((switch[0]['tuned'], switch[0]['native'], switch[0]['amp2'], switch[0]['home'], switch[0]['release'], switch[0]['home_steps']),
                         (True, False, True, True, False, 110))
        self.assertEqual((start[0]['tuned'], start[0]['native']), (False, True))
        self.assertEqual((keep[0]['tuned'], keep[0]['base']), (True, False))


if __name__ == '__main__':
    unittest.main()
