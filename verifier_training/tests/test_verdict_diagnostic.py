import unittest
import torch
from verifier_training.verdict_diagnostic import partition, token_losses, GROUPS

class Diagnostic(unittest.TestCase):
    def test_disjoint_class_continuation_eos(self):
        labels=torch.tensor([[-100,-100,2,3,4,5,9,-100]])
        rm=torch.tensor([[0,0,1,1,0,0,0,0]],dtype=torch.bool)
        vm=torch.tensor([[0,0,0,0,1,1,1,0]],dtype=torch.bool)
        masks=partition(labels,rm,vm,9)
        self.assertEqual([int(m.sum()) for m in masks.values()],[2,1,1,1])
        logits=torch.randn(1,8,10,requires_grad=True)
        token=token_losses(logits,labels)
        parts=[token[m[:,1:]].sum()/5 for m in masks.values()]
        self.assertTrue(torch.allclose(sum(parts),token.sum()/5))
        grads=[torch.autograd.grad(p,logits,retain_graph=True)[0] for p in parts]
        total=torch.autograd.grad(token.sum()/5,logits)[0]
        self.assertTrue(torch.allclose(sum(grads),total))
        self.assertEqual(total[:,0].abs().sum(),0)
        self.assertEqual(total[:,6:].abs().sum(),0)
    def test_missing_partition_fails(self):
        a=torch.tensor([[-100,1,2,9]])
        with self.assertRaises(ValueError):partition(a,torch.zeros_like(a,dtype=torch.bool),a.eq(2)|a.eq(9),9)

if __name__=='__main__':unittest.main()
