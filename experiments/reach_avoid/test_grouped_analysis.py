import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from grouped_analysis import fit_fixed,probability,paired_interval
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
if __name__=='__main__':unittest.main()
