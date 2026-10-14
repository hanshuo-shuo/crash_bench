import copy
import unittest
from verifier_training.fd1000_metrics import paired_comparison


def example():
    rows=[]
    for group in ('source_0','source_1'):
        for assignment in (0,1):
            for boundary in ('a','b'):
                truth=int(boundary=='b')
                rows.append(dict(sample_id=f'{group}_{assignment}_{boundary}',unit_id=group+'_unit',
                    source_group=group,source_id=group,assignment=assignment,protected_boundary=boundary,
                    student_input_sha256=f'pixels_{group}_{assignment}_{boundary}',truth=truth,predicted=truth,
                    gold_rationale_prefix_used=False))
    return rows


class PairedSourceComparison(unittest.TestCase):
    def test_order_does_not_change_alignment_and_invalid_is_wrong(self):
        a=example();b=copy.deepcopy(a);b[0]['predicted']=None
        result=paired_comparison(a,list(reversed(b)))
        self.assertEqual(result['generative_minus_direct_accuracy'],-.125)
        self.assertEqual(result['direct_only_correct'],1)
        self.assertEqual(result['physical_checkpoint_units'],2)
        self.assertEqual(result['source_paired_bootstrap_95pct'],[-.25,0.])

    def test_source_resampling_preserves_a_whole_correlated_group(self):
        a=example();b=copy.deepcopy(a)
        for row in a[:4]:row['predicted']=1-row['truth']
        for row in b[4:]:row['predicted']=1-row['truth']
        result=paired_comparison(a,b)
        self.assertEqual(result['generative_minus_direct_accuracy'],0.)
        self.assertEqual(result['source_paired_bootstrap_95pct'],[-1.,1.])

    def test_refuses_misaligned_evidence_and_incomplete_units(self):
        for key,value in [('truth',1),('source_group','other'),('student_input_sha256','other'),('gold_rationale_prefix_used',True)]:
            a=example();b=copy.deepcopy(a);b[0][key]=value
            with self.assertRaises(RuntimeError):paired_comparison(a,b)
        with self.assertRaises(RuntimeError):paired_comparison(example()[:-1],example()[:-1])
        with self.assertRaises(RuntimeError):paired_comparison(example(),example()+example()[:1])


if __name__=='__main__':unittest.main()
