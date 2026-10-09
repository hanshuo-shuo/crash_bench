"""Paired population/outcome tests; no policy or simulator execution."""
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from gate_summary import features,paired_guard,tables

def sample():
    states=['task%d/ep%02d'%(i//20,i%20) for i in range(60)]
    split=dict(train=[s for i,s in enumerate(states) if i%20<14],test=[s for i,s in enumerate(states) if i%20>=14])
    runs=[]
    for i,s in enumerate(states):
        for r in range(5):
            for arm in ['nominal','aegis','gate']:
                C=1 if arm=='nominal' else None;success=int(arm!='gate')
                runs.append(dict(id='%s_%s_%d'%(s,arm,r),scene=s,task=s.rsplit('/',1)[0],arm=arm,repeat=r,seed=i*100+r,C=C,L=300,success=success,
                    outcome='crash' if C else 'safe_success' if success else 'safe_incomplete',gated_actions=150,gated_inferences=30,inferences=60,risk_inferences=60,risk_gated=30))
    return runs,split


class GuardTests(unittest.TestCase):
    def test_collision_with_goal_keeps_TSR_but_not_safe_success(self):
        result=features(dict(C=3,L=3,success=1))
        self.assertEqual(result['CAR'],0);self.assertEqual(result['crash_rate'],1)
        self.assertEqual(result['TSR'],1);self.assertEqual(result['safe_success'],0)
        self.assertEqual(result['budget_goal_incomplete'],0)

    def test_missing_duplicate_and_unpaired_seed_are_rejected(self):
        runs,split=sample();paired_guard(runs,split)
        with self.assertRaises(RuntimeError):paired_guard(runs[:-1],split)
        changed=[dict(r) for r in runs];changed[-1]['seed']+=1
        with self.assertRaises(RuntimeError):paired_guard(changed,split)
        changed=[dict(r) for r in runs];changed[-1]=changed[0]
        with self.assertRaises(RuntimeError):paired_guard(changed,split)


try:import numpy;HAS_ARRAYS=True
except ImportError:HAS_ARRAYS=False

@unittest.skipUnless(HAS_ARRAYS,'Pinned Quest NumPy required')
class TableTests(unittest.TestCase):
    def test_heldout_pairs_and_outcome_difference_direction(self):
        import csv
        runs,split=sample()
        with TemporaryDirectory() as directory:
            output=Path(directory);tables(runs,split,dict(bootstrap_replicates=20,bootstrap_seed=20261009),output)
            rows=list(csv.DictReader((output/'paired_differences.csv').open()))
            primary=[r for r in rows if r['population']=='test_primary' and r['comparison']=='gate-minus-nominal']
            self.assertTrue(all(int(r['paired_seeds'])==90 and int(r['states'])==18 for r in primary))
            car=next(r for r in primary if r['metric']=='CAR')
            self.assertEqual(float(car['value']),1.);self.assertEqual(float(car['low']),1.);self.assertEqual(float(car['high']),1.)
            tsr=next(r for r in primary if r['metric']=='TSR');self.assertEqual(float(tsr['value']),-1.)
            safe=next(r for r in primary if r['metric']=='safe_success');self.assertEqual(float(safe['value']),0.)
            freq=list(csv.DictReader((output/'gate_frequencies.csv').open()))
            self.assertTrue(all(float(r['value'])==.5 for r in freq))


if __name__=='__main__':unittest.main()
