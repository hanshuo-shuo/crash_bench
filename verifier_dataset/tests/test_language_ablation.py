from collections import Counter
from pathlib import Path
import unittest
from verifier_dataset import hil_suite as S
from verifier_dataset import language_ablation as A

ROOT = Path(__file__).resolve().parents[2]
NAMES = S.task_names((ROOT/'third_party/LIBERO/libero/libero/benchmark/libero_suite_task_map.py').read_text())


class Design(unittest.TestCase):
    def test_covers_all_tasks_without_selecting_on_success(self):
        units = A.plan(NAMES, 'formal')
        self.assertEqual(len(units), 1240)
        self.assertEqual(len({u['id'] for u in units}), len(units))
        self.assertEqual(Counter(u['condition'] for u in units), dict(C0=400, EMPTY=400, RANDOM=400, C0R=40))
        for suite in S.SUITES:
            self.assertEqual({u['task'] for u in units if u['suite'] == suite}, set(range(10)))
        for group in {u['group'] for u in units}:
            rows = [u for u in units if u['group'] == group]
            self.assertEqual(len({u['seed'] for u in rows}), 1)
            self.assertEqual(len({u['max_steps'] for u in rows}), 1)
            self.assertTrue(all(u['prompt'] == '' for u in rows if u['condition'] == 'EMPTY'))
        self.assertEqual({u['init'] for u in units}, set(range(10)))

    def test_random_bank_cannot_identify_the_task(self):
        units = A.plan(NAMES, 'formal')
        for init in A.INITS:
            random = [u['prompt'] for u in units if u['init'] == init and u['condition'] == 'RANDOM']
            self.assertEqual(len(random), 40); self.assertEqual(set(random), {A.random_prompt(init)})
            self.assertEqual(len(random[0].split()), A.SYMBOLS)
            self.assertTrue(set(random[0].split()) <= set(A.ALPHABET))
        self.assertEqual(len({A.random_prompt(i) for i in A.INITS}), 10)

    def test_smoke_is_disjoint_and_twins_stay_together(self):
        smoke = A.plan(NAMES, 'smoke'); formal = A.plan(NAMES, 'formal')
        self.assertEqual(len(smoke), 16)
        self.assertEqual({u['suite'] for u in smoke}, set(S.SUITES))
        self.assertFalse({u['group'] for u in smoke} & {u['group'] for u in formal})
        shards = [A.shard_indices(formal, k, 2) for k in range(2)]
        self.assertEqual(sorted(shards[0] + shards[1]), list(range(len(formal))))
        self.assertFalse({formal[i]['group'] for i in shards[0]} & {formal[i]['group'] for i in shards[1]})
        self.assertEqual([len(x) for x in shards], [640, 600])

    def test_missing_mislabeled_and_failed_baselines(self):
        units = A.plan(NAMES, 'smoke')
        rows = {u['id']: dict(u, success=u['condition'] == 'EMPTY') for u in units}
        pairs = A.paired_rows(units, rows)
        self.assertEqual(len(pairs), 4)
        self.assertTrue(all(not p['C0'] and p['EMPTY'] for p in pairs))
        with self.assertRaises(RuntimeError):
            A.paired_rows(units, {k: v for k, v in rows.items() if k != units[0]['id']})
        rows[units[0]['id']]['prompt'] = 'wrong instruction'
        with self.assertRaises(RuntimeError):
            A.paired_rows(units, rows)


class Statistics(unittest.TestCase):
    def test_paired_resampling_retains_covariance(self):
        pairs = [dict(suite='s', task=0, init=i, C0=i < 5, EMPTY=i < 5, RANDOM=i >= 5) for i in range(10)]
        stats = A.statistics(pairs, replicates=1000)
        self.assertEqual(stats['success']['C0']['value'], .5)
        self.assertEqual(stats['contrasts']['EMPTY']['difference']['ci'], [0., 0.])
        self.assertEqual(stats['contrasts']['EMPTY']['ratio']['ci'], [1., 1.])
        self.assertEqual(stats['contrasts']['RANDOM']['baseline_only'], 5)
        self.assertEqual(stats['contrasts']['RANDOM']['ablation_only'], 5)
        self.assertEqual(stats['contrasts']['EMPTY']['retained_baseline_successes'], 5)

    def test_task_weights_and_zero_baseline(self):
        pairs = [dict(suite='s', task=0, init=i, C0=True, EMPTY=False, RANDOM=False) for i in range(10)]
        pairs += [dict(suite='s', task=1, init=0, C0=False, EMPTY=True, RANDOM=False)]
        self.assertEqual(A.statistics(pairs, replicates=100)['success']['C0']['value'], .5)
        stats = A.statistics([dict(suite='s', task=0, init=0, C0=False, EMPTY=True, RANDOM=False)], replicates=100)
        self.assertIsNone(stats['contrasts']['EMPTY']['ratio']['value'])
        self.assertIsNone(stats['contrasts']['EMPTY']['ratio']['ci'])
        self.assertEqual(stats['contrasts']['EMPTY']['difference']['value'], 1.)


if __name__ == '__main__':
    unittest.main()
