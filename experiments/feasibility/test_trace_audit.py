import unittest
from trace_audit import compare_common_trace, FIELDS


class TraceAuditTests(unittest.TestCase):
    def records(self):
        return [{field: (i + 1 if field == 'step' else [0., 1.]) for field in FIELDS}
                for i in range(2)]

    def test_larger_budget_can_extend_but_not_change_common_actions(self):
        shorter = self.records()
        longer = self.records() + [dict(shorter[-1], step=3)]
        self.assertEqual(compare_common_trace(shorter, longer, 'budget'), 2)
        longer[1] = dict(longer[1], output=[0., .9])
        with self.assertRaisesRegex(RuntimeError, 'field=output'):
            compare_common_trace(shorter, longer, 'budget')

    def test_larger_budget_early_exit_is_not_treated_as_budget_effect(self):
        with self.assertRaisesRegex(RuntimeError, 'stopped before'):
            compare_common_trace(self.records(), self.records()[:1], 'budget')

    def test_same_commands_with_different_physics_are_not_equal(self):
        left, right = self.records(), self.records()
        right[0]['obstacle_l1_m'] = .0011
        with self.assertRaisesRegex(RuntimeError, 'field=obstacle_l1_m'):
            compare_common_trace(left, right, 'control')


if __name__ == '__main__': unittest.main()
