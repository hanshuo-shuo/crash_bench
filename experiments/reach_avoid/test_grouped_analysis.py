import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from grouped_analysis import fit_fixed,probability,paired_interval,validate_outcome,nested_feasibility_select
class GroupedIncremental(unittest.TestCase):
    def test_constant_failure_fold_is_explicit_prior(self):
        x=np.arange(12).reshape(6,2)
        for value in (0,1):
            model=fit_fixed(x,np.full(6,value),LogisticRegression())
            np.testing.assert_array_equal(probability(model,x),np.full(6,value))
    def test_paired_interval_zero_for_identical_predictions(self):
        y=np.array([0,1,0,1]);p=np.array([.2,.8,.4,.9])
        result=paired_interval(y,p,p,['a','a','b','b'],30)
        self.assertEqual(result['delta_auroc_interval'],[0.,0.])
    def test_partial_timeout_is_not_failure(self):
        s=dict(outcome='safe_timeout',steps=1,execution_horizon_T=300,safe_history=True,safe_success=False,initial_valid=True,initial_label='feasible')
        with self.assertRaises(ValueError):validate_outcome(s)
        s['steps']=300;self.assertEqual(validate_outcome(s),1)
        s.update(outcome='invalid',steps=1);self.assertIsNone(validate_outcome(s))
        s.update(outcome='collision',safe_history=False);self.assertEqual(validate_outcome(s),1)
    def test_nested_selection_ignores_held_group_labels(self):
        rng=np.random.default_rng(24);features={'a':rng.normal(size=(14,3)),'b':rng.normal(size=(14,3))}
        y=np.tile([0,1],7);fitting=np.arange(8);val=np.arange(8,12);held=np.arange(12,14)
        one,records=nested_feasibility_select(features,['a','b'],fitting,val,y)
        y[held]=1-y[held]
        two,repeated=nested_feasibility_select(features,['a','b'],fitting,val,y)
        self.assertEqual(records,repeated);self.assertEqual(one[1],two[1])
        np.testing.assert_array_equal(probability(one[2],features[one[1]][held]),probability(two[2],features[two[1]][held]))
if __name__=='__main__':unittest.main()

