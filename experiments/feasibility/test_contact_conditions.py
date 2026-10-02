import unittest
from contact_conditions import merge_replays, prefix_eligibility, continuation_contact_counts


class ContactConditionTests(unittest.TestCase):
    def contact(self, name, robot=(), target=(), completed=False, job='1'):
        return dict(source_root='/tmp/diagnostic', run_id=name, robot_contact_actions=list(robot),
                    target_contact_actions=list(target),
                    completed_without_robot_target_protected_contact=completed, slurm_job=job)

    def base(self, repeat):
        return dict(state='spatial_09', repeat=repeat, run_id='base%d' % repeat,
                    condition='identity_geometry', branch_step=None)

    def fork(self, repeat, condition='reference', safe=True):
        return dict(state='spatial_09', repeat=repeat, run_id='fork%d_%s' % (repeat, condition),
                    condition=condition, branch_step=77, extra_budget=77, safe_success=safe)

    def test_target_contact_at_checkpoint_is_prior_contact_and_all_rows_stay(self):
        rows = [self.base(r) for r in range(2)] + [self.fork(r) for r in range(2)]
        contacts = [self.contact('base0', target=[77]), self.contact('base1', robot=[78])]
        gate = prefix_eligibility(rows, contacts, '/tmp/diagnostic', ['spatial_09'], range(2))
        item = next(x for x in gate if x['checkpoint'] == 77)
        self.assertEqual(item['official_proxy_eligible_repeats'], [0, 1])
        self.assertEqual(item['prior_protected_contact_repeats'], [0])
        self.assertEqual(item['no_prior_protected_contact_repeats'], [1])
        self.assertEqual(len(rows), 4)

    def test_missing_baseline_audit_stops_instead_of_assuming_no_contact(self):
        with self.assertRaises(RuntimeError):
            prefix_eligibility([self.base(0)], [], '/tmp/diagnostic', ['spatial_09'], range(1))

    def test_unknown_suffix_is_neither_certified_success_nor_failure(self):
        rows = [self.base(r) for r in range(2)] + [self.fork(r) for r in range(2)]
        contacts = [self.contact('base0'), self.contact('base1')]
        gate = prefix_eligibility(rows, contacts, '/tmp/diagnostic', ['spatial_09'], range(2))
        result = continuation_contact_counts(rows, contacts, '/tmp/diagnostic', gate)[0]
        self.assertEqual(result['unknown_contact_successes'], 2)
        self.assertEqual(result['completed_without_protected_contact'], 0)
        contacts.append(self.contact('fork1_reference', completed=True))
        result = continuation_contact_counts(rows, contacts, '/tmp/diagnostic', gate)[0]
        self.assertEqual(result['unknown_contact_successes'], 1)
        self.assertEqual(result['completed_without_protected_contact'], 1)

    def test_duplicate_replay_is_not_an_additional_trial_and_conflict_stops(self):
        a, b = self.contact('base0', robot=[10]), self.contact('base0', robot=[10], job='2')
        result = merge_replays([a, b])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['replay_jobs'], ['1', '2'])
        with self.assertRaises(RuntimeError):
            merge_replays([a, self.contact('base0', target=[10])])


if __name__ == '__main__':
    unittest.main()
