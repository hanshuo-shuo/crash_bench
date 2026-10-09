"""Independent scalar/pairwise bootstrap checks on synthetic inputs only."""
import importlib.util
import unittest
from stat_rules import bootstrap_weights,classification_metrics

AVAILABLE=importlib.util.find_spec('numpy') is not None


@unittest.skipUnless(AVAILABLE,'NumPy supplied by the pinned Quest analysis environment')
class ClusterStatsTests(unittest.TestCase):
    def test_repeated_clusters_keep_multiplicity(self):
        import numpy as np
        from cluster_stats import mean
        value,ci=mean([0.,0.,1.],['A','A','B'],[.25,.25,.5],['A','B'],np.array([[2,1]]))
        self.assertEqual(value,.5);self.assertAlmostEqual(ci['low'],1./3.)

    def test_pair_matrix_matches_independent_reweighted_tied_pairs(self):
        import numpy as np
        from cluster_stats import classification,interval
        rng=np.random.default_rng(4109)
        states=['A','B','C'];ids=['A','B','C']*10
        for _ in range(30):
            y=rng.integers(0,2,len(ids));x=rng.integers(-2,3,len(ids));p=rng.random(len(ids));w=rng.random(len(ids));w=w/w.sum()
            m=rng.integers(0,5,(20,3));fast=classification(y,x,p,ids,w,states,m)
            scalar={k:[] for k in fast}
            point=classification_metrics(y.tolist(),x.tolist(),p.tolist(),w.tolist())
            for counts in m:
                sampled=[s for s,c in zip(states,counts) for _ in range(c)]
                reweighted=bootstrap_weights(ids,w,sampled)
                check=classification_metrics(y,x,p,reweighted) if reweighted else dict.fromkeys(fast)
                for k in fast:scalar[k].append(np.nan if check[k] is None else check[k])
            for key in fast:
                self.assertAlmostEqual(fast[key]['value'],point[key],places=12)
                expected=interval(scalar[key])
                self.assertEqual(fast[key]['ci']['undefined'],expected['undefined'])
                self.assertAlmostEqual(fast[key]['ci']['low'],expected['low'],places=12)
                self.assertAlmostEqual(fast[key]['ci']['high'],expected['high'],places=12)

    def test_single_class_and_constant_ties(self):
        import numpy as np
        from cluster_stats import classification
        m=np.array([[2,1],[1,2],[0,0]])
        a=classification([1,1],[0,0],[.3,.3],['A','B'],[.5,.5],['A','B'],m)
        self.assertIsNone(a['AUROC']['value']);self.assertEqual(a['AUROC']['ci']['undefined'],3)
        b=classification([0,1],[0,0],[.3,.3],['A','B'],[.5,.5],['A','B'],m)
        self.assertEqual(b['AUROC']['value'],.5);self.assertEqual(b['AUROC']['ci']['low'],.5)

    def test_stratified_draws_repeat_and_preserve_task_sizes(self):
        import numpy as np
        from cluster_stats import draws
        a=draws(['A','B','C','D'],['T1','T1','T2','T2'],25,20261009)
        self.assertTrue(np.array_equal(a,draws(['A','B','C','D'],['T1','T1','T2','T2'],25,20261009)))
        self.assertTrue((a[:,:2].sum(axis=1)==2).all());self.assertTrue((a[:,2:].sum(axis=1)==2).all())


if __name__=='__main__':unittest.main()
