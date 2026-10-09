"""Small synthetic checks only; no experimental outcome is loaded or fitted."""
from pathlib import Path
import importlib.util
import json
import math
import unittest
from platt import fit,predict

AVAILABLE=importlib.util.find_spec('numpy') is not None and importlib.util.find_spec('scipy') is not None
PARAMETERS=json.loads((Path(__file__).parent/'config.yaml').read_text())['task2']['platt']


@unittest.skipUnless(AVAILABLE,'NumPy/SciPy supplied by the pinned Quest analysis environment')
class PlattTests(unittest.TestCase):
    def test_complete_separation_has_finite_regularized_solution(self):
        m=fit([-1.,1.],[0,1],[.5,.5],PARAMETERS)
        p=predict(m,[-1.,1.]).tolist()
        self.assertFalse(m['fallback']);self.assertTrue(math.isfinite(m['slope']))
        self.assertGreater(m['slope'],0.);self.assertGreater(p[0],0.);self.assertLess(p[1],1.)
        self.assertAlmostEqual(p[0]+p[1],1.,places=9)

    def test_single_class_fallback_is_frozen_and_explicit(self):
        m=fit([1.,2.,3.],[1,1,1],[.2,.3,.5],PARAMETERS)
        self.assertEqual(m['status'],'single_class');self.assertEqual(m['slope'],0.)
        self.assertEqual(predict(m,[4.,5.]).tolist(),[.999999,.999999])

    def test_constant_input_uses_weighted_prevalence(self):
        m=fit([3.,3.],[0,1],[.8,.2],PARAMETERS)
        self.assertEqual(m['status'],'constant_input')
        self.assertEqual(predict(m,[1.,9.]).tolist(),[.2,.2])

    def test_standardization_and_weight_scale_do_not_change_predictions(self):
        x=[-2.,-1.,1.,2.];y=[0,1,0,1];w=[1.,2.,3.,4.]
        a=fit(x,y,w,PARAMETERS)
        b=fit([10.+5.*v for v in x],y,[10.*v for v in w],PARAMETERS)
        for p,q in zip(predict(a,x),predict(b,[10.+5.*v for v in x])):self.assertAlmostEqual(p,q,places=9)

    def test_invalid_labels_do_not_fit(self):
        with self.assertRaisesRegex(ValueError,'Binary'):fit([1.,2.],[0,-1],[1.,1.],PARAMETERS)


if __name__=='__main__':unittest.main()
