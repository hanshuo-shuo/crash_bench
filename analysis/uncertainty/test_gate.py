"""Synthetic gate decision checks; never executes a simulator or policy."""
import unittest
from gate import command,schedule
from common import config,full_schedule


class GateTests(unittest.TestCase):
    def test_strict_threshold_and_all_seven_zero_preserves_raw(self):
        raw=[1.,-2.,3.,4.,5.,6.,-1.];saved=raw.copy()
        self.assertEqual(command(raw,.5,.5),(raw,False))
        self.assertEqual(command(raw,.5000001,.5),([0.]*7,True));self.assertEqual(raw,saved)

    def test_invalid_proxy_or_action_rejected(self):
        with self.assertRaises(ValueError):command([0.]*6,1.,.5)
        with self.assertRaises(ValueError):command([0.]*7,float('nan'),.5)

    def test_full_matrix_and_two_scene_two_seed_frontier(self):
        cfg=config('full');cfg['repeats']=list(range(5));all_cases=schedule(cfg)
        self.assertEqual(len(all_cases),300);self.assertEqual(len(schedule(cfg,0)),150);self.assertEqual(len(schedule(cfg,1)),150)
        front=schedule(cfg,0)[:4];self.assertEqual({(x['state_index'],x['repeat']) for x in front},{(0,0),(0,1),(42,0),(42,1)})
        self.assertEqual(len({x['name'] for x in all_cases}),300)
        baseline={(x['state_index'],x['repeat']):x['scene'] for x in full_schedule(config('full')) if x['method']=='nominal' and x['repeat']<5}
        self.assertEqual({(x['state_index'],x['repeat']) for x in all_cases},set(baseline))
        for x in all_cases:self.assertEqual(x['scene'],baseline[(x['state_index'],x['repeat'])])


if __name__=='__main__':unittest.main()
