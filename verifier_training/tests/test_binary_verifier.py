import unittest
import torch
from verifier_training.binary_verifier import losses, parse, LABEL_TO_DIGIT
from verifier_training.loss import gradient_probe

class BinaryLoss(unittest.TestCase):
    def test_equal_weights_and_no_eos_dilution(self):
        for length in (5,25):
            z=torch.zeros(1,length,7,requires_grad=True)
            labels=torch.ones(1,length,dtype=torch.long);labels[:,:2]=-100
            labels[:,-1]=6  # supervised EOS, outside the binary component
            vm=torch.zeros_like(labels,dtype=torch.bool);vm[:,-2]=True
            rm=labels.ne(-100)&~vm
            loss=losses(z,labels,rm,vm);loss['total'].backward()
            self.assertAlmostEqual(float(z.grad[0,-3,1]),-.5*6/7,places=6)
            self.assertGreater(float(z.grad[0,-2].abs().sum()),0)
            self.assertEqual(float(z.grad[0,0].abs().sum()),0)
            self.assertAlmostEqual(float(loss['total']),float(.5*loss['rationale']+.5*loss['verdict']))
    def test_both_components_backpropagate_to_lora(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.lora_A=torch.nn.Parameter(torch.randn(3,2));self.lora_B=torch.nn.Parameter(torch.zeros(2,7));self.visual=torch.nn.Linear(3,3);self.visual.requires_grad_(False)
            def forward(self,x):return x@self.lora_A@self.lora_B
        m=Toy();z=m(torch.randn(1,6,3));labels=torch.tensor([[-100,-100,2,3,4,6]])
        vm=torch.tensor([[0,0,0,0,1,0]],dtype=torch.bool);rm=labels.ne(-100)&~vm
        values=losses(z,labels,rm,vm);probe=gradient_probe(m,values,'B')
        self.assertTrue(all(v['lora_gradient_norm']>0 for v in probe.values()))
        values['total'].backward();self.assertIsNone(m.visual.weight.grad)
    def test_padding_ignored_and_real_eos_kept(self):
        z=torch.zeros(1,7,7,requires_grad=True);labels=torch.tensor([[-100,-100,2,3,6,-100,-100]])
        vm=labels.eq(3);rm=labels.ne(-100)&~vm
        losses(z,labels,rm,vm)['total'].backward()
        self.assertGreater(float(z.grad[0,3].abs().sum()),0)
        self.assertEqual(float(z.grad[0,4:].abs().sum()),0)
    def test_two_digit_or_missing_partition_rejected(self):
        z=torch.zeros(1,5,7);labels=torch.tensor([[-100,-100,2,3,6]])
        vm=labels.eq(2)|labels.eq(3);rm=labels.ne(-100)&~vm
        with self.assertRaises(ValueError):losses(z,labels,rm,vm)
    def test_mapping_and_strict_output(self):
        self.assertEqual(LABEL_TO_DIGIT,{'infeasible':'0','feasible':'1'})
        self.assertEqual(parse('Rationale: Visible.\nVerdict: 0'),'infeasible')
        self.assertEqual(parse('Rationale: Visible.\nVerdict: 1'),'feasible')
        self.assertIsNone(parse('Rationale: Visible.\nVerdict: 10'))
        self.assertIsNone(parse('Rationale: Visible.\nVerdict: feasible'))

if __name__=='__main__':unittest.main()
