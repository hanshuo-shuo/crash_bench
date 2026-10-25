import unittest
from collections import Counter, defaultdict

from verifier_dataset import v5_dataset as D


def row(source, unit, rule, label, time_bin=0, lift=.01, base=None, layout=0, t=1):
    return dict(source=source, unit=unit, rule=rule, label=label, time_bin=time_bin, lift=lift, task='t|x', clearance=.2,
                robot_clearance=.3, witnessed=True, target_robot_side=t, goal_robot_side=t ^ label,
                base_source=base or source.split('#')[0], layout=layout)


class Rows(unittest.TestCase):
    def test_kept_needs_margin_robot_room_and_a_witness(self):
        self.assertTrue(D.kept(row('s', 0, 'line:a|b', 0)))
        self.assertFalse(D.kept(dict(row('s', 0, 'line:a|b', 0), witnessed=False)))
        self.assertTrue(D.kept(dict(row('s', 0, 'line:a|b', 1), witnessed=False)))
        self.assertFalse(D.kept(dict(row('s', 0, 'line:a|b', 1), robot_clearance=.05)))
        self.assertFalse(D.kept(dict(row('s', 0, 'line:a|b', 1), clearance=.04)))

    def test_texts(self):
        r = row('s', 0, 'line:alphabet_soup|orange_juice', 1)
        f = D.facts(r)
        self.assertEqual(f, dict(line='alphabet soup can | orange juice carton', target_robot_side='yes', goal_robot_side='no', separated='yes'))
        self.assertEqual(D.facts_text(f).splitlines()[0], 'Line: alphabet soup can | orange juice carton')
        text = D.constraint_text('line:alphabet_soup|orange_juice')
        self.assertIn('through the alphabet soup can and the orange juice carton', text)
        self.assertIn('robot arm itself may pass over', text)
        self.assertEqual((D.DIGIT[1], D.VERDICT[1], D.DIGIT[0], D.VERDICT[0]), ('0', 'infeasible', '1', 'feasible'))


class Sampling(unittest.TestCase):
    def test_natural_sampling_takes_two_steps_per_bin_and_two_rules_per_image(self):
        rows = [row('s#1', step, 'line:a|%s' % r, (step + i) % 2, time_bin=step // 10)
                for step in range(0, 30, 5) for i, r in enumerate('bcd')]
        picks = D.sample_natural(rows, 'salt')
        self.assertEqual(Counter(rs[0]['time_bin'] for _, rs in picks), {0: 2, 1: 2, 2: 2})
        self.assertTrue(all(len(rs) == 2 for _, rs in picks))
        self.assertEqual(picks, D.sample_natural(rows, 'salt'))
        self.assertNotEqual(picks, D.sample_natural(rows, 'other'))

    def test_balanced_twins_make_every_rule_progress_group_even(self):
        rows = []
        for k in range(4):
            rows += [row('v#%d' % k, 0, 'line:a|b', k % 2), row('v#%d' % k, 0, 'line:a|c', 1 - k % 2)]
        rows += [row('v#4', 0, 'line:a|b', 1), row('v#4', 0, 'line:a|c', 0)]  # makes line:a|b 3 vs 2
        images = D.balanced_twins(rows, 'salt')
        flat = [r for _, rs in images for r in rs]
        groups = defaultdict(Counter)
        for r in flat:
            groups[r['rule']][r['label']] += 1
        self.assertTrue(all(c[0] == c[1] for c in groups.values()))
        self.assertEqual(D.lookup([dict(r, split='train') for r in rows], flat, 'label'), .5)


class Readouts(unittest.TestCase):
    def test_reading_needs_enough_twins_and_an_even_lookup(self):
        flat = []
        for part, n in (('train', 4), ('validation', 200), ('test', 200)):
            for i in range(n):
                for lab in (0, 1):
                    flat.append(dict(task='t|x', rule='line:a|%d' % (i % 3), time_bin=0, label=lab, target_robot_side=1,
                                     goal_robot_side=1 - lab, split=part, image='%s%d' % (part, i), base_source='%s%d' % (part, i % 4),
                                     layout=0))
        out = D.readouts(flat)
        self.assertEqual((out['validation']['twin_images'], out['validation']['lookup_from_train']['label']), (200, .5))
        self.assertEqual(out['reading'], 's2_dataset_ready')
        out = D.readouts([r for r in flat if not (r['split'] == 'test' and r['image'] > 'test5')])
        self.assertEqual(out['reading'], 's2_dataset_insufficient')


if __name__ == '__main__':
    unittest.main()
