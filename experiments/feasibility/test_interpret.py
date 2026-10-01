import unittest
from interpret import outcome_counts, finite_library
from protocol import BRANCH_CONDITIONS


class InterpretationTests(unittest.TestCase):
    def row(self, repeat, candidate, safe=False, step=27):
        return {'state': 'object_00', 'repeat': repeat, 'condition': candidate,
                'branch_step': step, 'extra_budget': step, 'safe_success': safe}

    def test_oracle_library_is_distinct_from_one_fixed_candidate(self):
        rows = []
        for repeat in range(2):
            for candidate in BRANCH_CONDITIONS:
                rows.append(self.row(repeat, candidate,
                                     (repeat == 0 and candidate == 'nominal') or
                                     (repeat == 1 and candidate == 'release5')))
        x = finite_library(rows, 'object_00', 27, 27)
        self.assertEqual(x['n'], 2)
        self.assertEqual(x['ordinary_library_any'], 2)
        self.assertEqual(x['best_fixed_candidate_count'], 1)
        self.assertEqual(x['best_fixed_candidates'], ['nominal', 'release5'])

    def test_missing_fork_not_counted_as_failure_and_reference_excluded(self):
        rows = [self.row(0, c, c == 'reference') for c in BRANCH_CONDITIONS]
        rows += [self.row(1, 'nominal', True)]
        x = finite_library(rows, 'object_00', 27, 27)
        self.assertEqual(x['n'], 1)
        self.assertEqual(x['counts']['reference'], 1)
        self.assertEqual(x['ordinary_library_any'], 0)
        self.assertEqual(x['best_fixed_candidates'], [])
        self.assertEqual(finite_library(rows, 'object_00', 77, 77)['n'], 0)

    def test_four_outcomes_exhaust_records_with_separate_method_exit(self):
        rows = [{'success': success, 'collided': collided, 'exited': not success}
                for success in [False, True] for collided in [False, True]]
        x = outcome_counts(rows)
        self.assertEqual(x, {'n': 4, 'safe_complete': 1, 'safe_incomplete': 1,
                             'unsafe_complete': 1, 'unsafe_incomplete': 1, 'method_exits': 2})


if __name__ == '__main__':
    unittest.main()
