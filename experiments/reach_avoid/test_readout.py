import unittest
import numpy as np
from readout import check_groups, failure_target, metrics, cluster_interval


class GroupedReadout(unittest.TestCase):
    def test_group_leak_rejected(self):
        rows=[dict(scene_group='A',split='train'),dict(scene_group='A',split='test'),dict(scene_group='B',split='validation')]
        with self.assertRaises(ValueError):check_groups(rows)
    def test_need_three_splits(self):
        with self.assertRaises(ValueError):check_groups([dict(scene_group='A',split='train')])
    def test_failure_is_full_horizon_endpoint(self):
        self.assertEqual(failure_target(dict(outcome='safe_timeout')),1)
        self.assertEqual(failure_target(dict(outcome='safe_completion')),0)
        self.assertIsNone(failure_target(dict(outcome='invalid')))
    def test_single_class_auc_not_fabricated(self):
        self.assertIsNone(metrics([1,1],[.2,.9])['auroc'])
    def test_bootstrap_uses_groups_not_frames(self):
        result=cluster_interval([0,1,0,1],[.1,.9,.2,.8],['A','A','A','A'],repetitions=20)
        self.assertIsNone(result['interval']);self.assertEqual(result['groups'],1)


if __name__=='__main__':unittest.main()
