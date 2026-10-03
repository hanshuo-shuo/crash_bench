import math
from pathlib import Path
import random
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
from contract import bounds, predicate, legal_action, exclusion, decide, witness_at, SMOKE, PILOT, CAPS, canonical_gripper_command

class ContractTests(unittest.TestCase):
    def test_only_sign_preserving_gripper_replay_is_canonicalized(self):
        for v in [-1.014,-.4,0,.7,1.008]:
            raw=[.4]*6+[v]; actual=canonical_gripper_command(raw)
            self.assertTrue(legal_action(actual))
            self.assertEqual(actual[:6],raw[:6])
            self.assertEqual((v>0)-(v<0),(actual[6]>0)-(actual[6]<0))
        for a in [[1.01]*7,[0]*6+[float('nan')]]:
            with self.assertRaises(ValueError): canonical_gripper_command(a)
    def test_strict_boundary_and_native_z_extension(self):
        R=[[1,0,0],[0,1,0],[0,0,1]]
        self.assertFalse(predicate([0,-.1,0],[0,0,0],R,[.1,.1,.1]))
        self.assertTrue(predicate([0,-.1+1e-10,-.105],[0,0,0],R,[.1,.1,.1]))
        self.assertFalse(predicate([0,0,-.11],[0,0,0],R,[.1,.1,.1]))

    def test_moving_rotating_goal_cannot_escape_certificate(self):
        rng=random.Random(701)
        for _ in range(1000):
            s=[rng.uniform(-10,10) for _ in range(3)]
            angle=rng.uniform(-math.pi,math.pi)
            R=[[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1]]
            size=[rng.uniform(.01,.2) for _ in range(3)]
            lo,hi=bounds(s,R,size)
            point=[(lo[i]+hi[i])/2 for i in range(3)]
            self.assertTrue(predicate(point,s,R,size))
            self.assertGreater(point[1]-lo[1],0)
            for cap in [-.005,-.05]: self.assertFalse(point[1]-lo[1]<=cap)

    def test_unsupported_positive_nan_and_small_margin_abstain(self):
        for cap in [None, .1, 0, -.0000001, float('nan'), float('inf')]:
            self.assertIsNone(exclusion(cap,True))
        self.assertIsNone(exclusion(-.1,False))
        self.assertEqual(exclusion(-.1,True)['label'],'infeasible')

    def test_no_success_never_means_infeasible(self):
        self.assertEqual(decide(False,None),'unknown')
        self.assertEqual(decide(True,None),'feasible')
        with self.assertRaises(RuntimeError): decide(True,exclusion(-.1,True))

    def test_history_budget_and_synchronized_success(self):
        records=[dict(step=0,protected_contact_count=0,max_ingress_m=-.2,native_success=False,synchronized_success=False),
                 dict(step=230,protected_contact_count=0,max_ingress_m=.04,native_success=True,synchronized_success=True)]
        self.assertEqual(witness_at(records,.15,300),230)
        self.assertIsNone(witness_at(records,.005,300))
        self.assertIsNone(witness_at(records,.15,150))
        self.assertIsNone(witness_at(records,.15,300,False))
        records[0]['protected_contact_count']=1
        self.assertIsNone(witness_at(records,.15,300))
        records[0]['protected_contact_count']=0
        records[-1]['synchronized_success']=False
        self.assertIsNone(witness_at(records,.15,300))

    def test_action_model_and_grouping(self):
        self.assertTrue(legal_action([0]*6+[-1]))
        for a in [[0]*6,[0]*6+[1.01],[0]*6+[float('nan')]]:self.assertFalse(legal_action(a))
        self.assertEqual(len(PILOT),7)
        self.assertEqual(len({s['layout_group'] for s in PILOT}),6)
        self.assertNotIn(SMOKE[0]['layout_group'],{s['layout_group'] for s in PILOT})
        holdout=[s for s in PILOT if s['split'].endswith('holdout')]
        self.assertEqual(len(holdout),5)
        self.assertEqual(len({s['layout_group'] for s in holdout}),5)
        self.assertEqual(set(CAPS.values()),{-.05,.15,-.005,.005})

if __name__=='__main__': unittest.main()
