import unittest
from verifier_training.rank_contract import apply_profile, config_difference, PROFILE


class RankContract(unittest.TestCase):
    def setUp(self):
        self.base = {'lora_rank':16, 'lora_alpha':32, 'learning_rate':1e-4,
                     'component_weights':{'rationale':.5,'verdict':.5},
                     'vision_encoder':'frozen', 'projector':'frozen'}

    def test_rank_and_scale_only(self):
        candidate = apply_profile(self.base, PROFILE)
        self.assertEqual(set(config_difference(self.base,candidate)), {'lora_rank','lora_alpha'})
        self.assertEqual(candidate['lora_alpha']/candidate['lora_rank'], 2)
        self.assertEqual(self.base['lora_rank'],16)

    def test_old_default_unchanged(self):
        self.assertEqual(apply_profile(self.base),self.base)
        self.assertIsNot(apply_profile(self.base),self.base)

    def test_confounded_comparison_rejected(self):
        for key,value in [('learning_rate',2e-4),('vision_encoder','trainable'),
                          ('projector','trainable'),('component_weights',{'rationale':.2,'verdict':.8})]:
            candidate=apply_profile(self.base,PROFILE);candidate[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                config_difference(self.base,candidate)

    def test_wrong_profile_and_baseline_rejected(self):
        with self.assertRaises(ValueError):apply_profile(self.base,'rank128')
        with self.assertRaises(ValueError):apply_profile(dict(self.base,lora_rank=32),PROFILE)

    def test_nested_baseline_not_mutated(self):
        candidate=apply_profile(self.base,PROFILE)
        candidate['component_weights']['verdict']=.9
        self.assertEqual(self.base['component_weights']['verdict'],.5)
