import unittest
from contact_conditions import merge_replays, prefix_eligibility, continuation_contact_counts, initial_reference_contacts, contact_qualified_view


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

    def test_initial_validation_requires_every_replay_and_keeps_proxy_contact_distinct(self):
        rows = [dict(state='spatial_09', repeat=r, run_id='validation%d' % r, condition='reference',
                     branch_step=None, validation=True, success=True, safe_success=True) for r in range(2)]
        contacts = [dict(self.contact('validation%d' % r, completed=r == 1),
                         source_success=True, source_safe_success=True) for r in range(2)]
        result = initial_reference_contacts(rows, contacts, '/tmp/diagnostic', ['spatial_09'], range(2))[0]
        self.assertEqual(result['task_complete'], 2)
        self.assertEqual(result['official_proxy_safe_complete'], 2)
        self.assertEqual(result['both_safe_complete'], 1)
        with self.assertRaises(RuntimeError):
            initial_reference_contacts(rows, contacts[:1], '/tmp/diagnostic', ['spatial_09'], range(2))
        contacts[0]['source_success'] = False
        with self.assertRaises(RuntimeError):
            initial_reference_contacts(rows, contacts, '/tmp/diagnostic', ['spatial_09'], range(2))

    def test_budget_contact_alias_requires_entire_trace_and_identical_initial_state(self):
        remaining = dict(self.fork(0), run_id='remaining', extra_budget=0,
                         end_step=100, initial_execution_fingerprint='initial')
        extended = dict(remaining, run_id='extended', extra_budget=77)
        contact = dict(self.contact('extended', completed=True), source_success=True, source_safe_success=True)
        gate = [dict(state='spatial_09', checkpoint=77, no_prior_protected_contact_repeats=[0])]
        quality = dict(status='passed', budget_pairs=[dict(remaining='remaining', extended='extended', verified_common_actions=100)])
        result = contact_qualified_view([remaining, extended], [contact], '/tmp/diagnostic', gate, quality)
        self.assertTrue(all(r['safe_success'] for r in result['rows']))
        self.assertEqual(len(result['audit_aliases']), 1)
        self.assertTrue(remaining['safe_success'])
        extended['initial_execution_fingerprint'] = 'different'
        with self.assertRaises(RuntimeError):
            contact_qualified_view([remaining, extended], [contact], '/tmp/diagnostic', gate, quality)
        extended['initial_execution_fingerprint'] = 'initial'
        quality['budget_pairs'][0]['verified_common_actions'] = 99
        with self.assertRaises(RuntimeError):
            contact_qualified_view([remaining, extended], [contact], '/tmp/diagnostic', gate, quality)

    def test_physical_view_preserves_original_proxy_success_and_excludes_touched_prefix(self):
        row = self.fork(0)
        contact = dict(self.contact(row['run_id'], target=[80]), source_success=True, source_safe_success=True)
        gate = [dict(state='spatial_09', checkpoint=77, no_prior_protected_contact_repeats=[0])]
        quality = dict(status='passed', budget_pairs=[])
        result = contact_qualified_view([row], [contact], '/tmp/diagnostic', gate, quality)
        self.assertFalse(result['rows'][0]['safe_success'])
        self.assertTrue(row['safe_success'])
        with self.assertRaises(RuntimeError):
            contact_qualified_view([row], [], '/tmp/diagnostic', gate, quality)
        gate[0]['no_prior_protected_contact_repeats'] = []
        self.assertEqual(contact_qualified_view([row], [], '/tmp/diagnostic', gate, quality)['rows'], [])

    def test_matched_physical_time_comparison_uses_late_untouched_cohort(self):
        from analysis import continuation_comparisons
        from protocol import BRANCH_CONDITIONS
        rows = []; contacts = []
        for step in [0, 77]:
            for repeat in range(2):
                for condition in BRANCH_CONDITIONS:
                    safe = condition == 'reference' and step == 0
                    row = dict(self.fork(repeat, condition, safe), branch_step=step,
                               extra_budget=step, run_id='%d_%d_%s' % (step, repeat, condition))
                    rows.append(row)
                    if safe:
                        contacts.append(dict(self.contact(row['run_id'], completed=True),
                                             source_success=True, source_safe_success=True))
        gates = [dict(state='spatial_09', checkpoint=0, no_prior_protected_contact_repeats=[0,1]),
                 dict(state='spatial_09', checkpoint=77, no_prior_protected_contact_repeats=[1])]
        view = contact_qualified_view(rows, contacts, '/tmp/diagnostic', gates, dict(status='passed', budget_pairs=[]))
        comparison = continuation_comparisons(view['rows'], 'spatial_09')[0]
        self.assertEqual(comparison['same_surviving_prefix_repeats'], [1])
        self.assertEqual(comparison['reference_early_witness'], 1)
        self.assertEqual(comparison['reference_late_witness'], 0)


if __name__ == '__main__':
    unittest.main()
