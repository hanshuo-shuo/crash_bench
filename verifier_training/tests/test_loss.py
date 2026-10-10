import unittest
import torch
from verifier_training.loss import masked_losses, gradient_probe
from verifier_training.data import collate

class Loss(unittest.TestCase):
    def test_causal_shift_prompt_and_padding_mask(self):
        logits=torch.zeros(1,6,5,requires_grad=True)
        labels=torch.tensor([[-100,-100,2,3,4,-100]])
        rm=torch.tensor([[False,False,True,True,False,False]])
        vm=torch.tensor([[False,False,False,False,True,False]])
        loss=masked_losses(logits,labels,rm,vm);loss['total'].backward()
        self.assertEqual(loss['rationale_tokens'],2);self.assertEqual(loss['verdict_tokens'],1)
        self.assertEqual(logits.grad[0,0].abs().sum(),0)
        self.assertEqual(logits.grad[0,4:].abs().sum(),0)
        self.assertGreater(logits.grad[0,1:4].abs().sum(),0)
        self.assertAlmostEqual(loss['total'].item(),torch.log(torch.tensor(5.)).item())
    def test_rationale_and_verdict_independent_gradients(self):
        # Actual gradients through a learnable low-rank branch, not loss scalars.
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.lora_A=torch.nn.Parameter(torch.randn(3,2));self.lora_B=torch.nn.Parameter(torch.zeros(2,7));self.visual=torch.nn.Linear(3,3);self.visual.requires_grad_(False)
            def forward(self,x):return x@self.lora_A@self.lora_B
        m=Toy();logits=m(torch.randn(1,5,3));labels=torch.tensor([[-100,-100,2,3,4]])
        rm=torch.tensor([[False,False,True,True,False]]);vm=torch.tensor([[False,False,False,False,True]])
        losses=masked_losses(logits,labels,rm,vm);probe=gradient_probe(m,losses,'B')
        self.assertGreater(probe['rationale']['lora_gradient_norm'],0);self.assertGreater(probe['verdict']['lora_gradient_norm'],0)
    def test_padding_does_not_mask_real_eos(self):
        def item(ids):
            a=torch.tensor([ids]);mask=torch.ones_like(a,dtype=torch.bool)
            return ({'input_ids':a,'attention_mask':torch.ones_like(a),'pixel_values':torch.zeros(2,8),'image_grid_thw':torch.ones(2,3,dtype=torch.long)},a.clone(),mask,mask,1)
        batch,labels,rm,vm=collate([item([1,2,0]),item([1,0])],0)
        self.assertEqual(labels.tolist(),[[1,2,0],[1,0,-100]])
        self.assertFalse(vm[1,-1]);self.assertTrue(vm[1,1])
    def test_nonzero_verdict_required(self):
        m=torch.nn.Linear(2,5);z=m(torch.ones(1,3,2));labels=torch.full((1,3),-100)
        masks=torch.zeros(1,3,dtype=torch.bool)
        with self.assertRaises(RuntimeError):gradient_probe(m,masked_losses(z,labels,masks,masks),'A')

if __name__=='__main__':unittest.main()
