import unittest
from verifier_dataset.replay import checkpoint_steps, assert_trace


class ReplayEvidence(unittest.TestCase):
    def test_nonterminal_source_inference_boundaries(self):
        steps=checkpoint_steps(160)
        self.assertEqual(len(steps),10)
        self.assertEqual(len(set(steps)),10)
        self.assertTrue(all(t%5==0 and 160-t>10 for t in steps))
        self.assertEqual(checkpoint_steps(5),[])

    def test_missing_or_nonfinite_action_cannot_be_a_witness(self):
        with self.assertRaises(RuntimeError): assert_trace([{'step':2,'applied':[0.]*7}],1)
        with self.assertRaises(RuntimeError): assert_trace([{'step':1,'applied':[float('nan')]*7}],1)
        with self.assertRaises(RuntimeError): assert_trace([],1)


if __name__ == '__main__': unittest.main()
