import unittest
from verifier_training.rank_contract import PROFILE,LOW_LR_PROFILE,apply_profile,lr_difference


class LowerLearningRate(unittest.TestCase):
    def base(self,lr):
        return {'lora_rank':16,'lora_alpha':32,'learning_rate':lr,'max_steps':240,
                'lora_dropout':0.,'vision_encoder':'frozen','projector':'frozen',
                'component_weights':{'rationale':.5,'verdict':.5}}

    def test_only_fixed_quarter_lr_changes(self):
        for before,after in ((2e-4,5e-5),(1e-4,2.5e-5)):
            high=apply_profile(self.base(before),PROFILE)
            low=apply_profile(self.base(before),LOW_LR_PROFILE)
            self.assertEqual(lr_difference(high,low),{'learning_rate':{'original_lr':before,'quarter_lr':after}})
            self.assertEqual((low['lora_rank'],low['lora_alpha']),(64,128))

    def test_budget_loss_and_freeze_changes_rejected(self):
        high=apply_profile(self.base(2e-4),PROFILE)
        for key,value in [('max_steps',480),('lora_dropout',.05),('lora_rank',128),
                          ('vision_encoder','trainable'),('projector','trainable'),
                          ('component_weights',{'rationale':.2,'verdict':.8})]:
            low=apply_profile(self.base(2e-4),LOW_LR_PROFILE);low[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):lr_difference(high,low)

    def test_arbitrary_lr_search_not_allowed(self):
        high=apply_profile(self.base(2e-4),PROFILE)
        low=apply_profile(self.base(2e-4),LOW_LR_PROFILE);low['learning_rate']=1e-5
        with self.assertRaises(ValueError):lr_difference(high,low)
        with self.assertRaises(ValueError):apply_profile(self.base(3e-4),LOW_LR_PROFILE)
