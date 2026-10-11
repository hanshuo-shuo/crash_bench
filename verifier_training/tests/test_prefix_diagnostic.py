import unittest
import torch
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
from verifier_training.prefix_diagnostic import ForcedPrefix

class RepetitionEffect(unittest.TestCase):
    def test_repeated_negative_prefix_can_flip_class(self):
        # An ordinary token already seen in the question can be penalized at the
        # class decision, independently of feasibility evidence or model weights.
        logits=torch.tensor([[0.,20.,20.5]])
        prefix=torch.tensor([[0,2,0]])
        adjusted=RepetitionPenaltyLogitsProcessor(1.05)(prefix,logits.clone())
        self.assertEqual(int(logits.argmax()),2)
        self.assertEqual(int(adjusted.argmax()),1)
        self.assertEqual(float(adjusted[0,1]),20.)
        self.assertAlmostEqual(float(adjusted[0,2]),20.5/1.05,places=5)
    def test_native_prefix_forcing_preserves_inspected_scores(self):
        processor=ForcedPrefix(2,[3,4],1)
        z=torch.tensor([[1.,2.,3.,4.,5.]])
        first=processor(torch.tensor([[0,1]]),z)
        self.assertEqual(int(first.argmax()),3);self.assertEqual(len(processor.inspected),0)
        second=processor(torch.tensor([[0,1,3]]),z)
        self.assertEqual(int(second.argmax()),4)
        self.assertTrue(torch.equal(processor.inspected[0],z))
        self.assertTrue(torch.equal(processor(torch.tensor([[0,1,3,4]]),z),z))

if __name__=='__main__':unittest.main()
