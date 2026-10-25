import inspect
import unittest
from verifier_training.launch_verdict_diagnostic import ASSETS
from verifier_training.v5_contract import DATASET,FACT_KEYS,FORCED_PREFIX,assistant_target,configuration,validate


@unittest.skipUnless((ASSETS/'model').exists() and (DATASET/'dataset.jsonl').exists(),'Staged model or S2 dataset not mounted')
class Encoding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from verifier_training.fd1000_run import processor_for
        cls.processor=processor_for(ASSETS);rows=validate()
        cls.rows=[next(r for r in rows if r['split']==s) for s in ('train','validation','validation_natural')]

    def test_v5_targets_keep_token_boundaries_and_identical_inputs(self):
        from verifier_training.fd1000_run import encode,input_digest,vision_inputs
        config=configuration()
        for r in self.rows:
            vision=vision_inputs(self.processor,r,DATASET);items={}
            for kind in ('direct','facts'):
                items[kind]=encode(self.processor,r,DATASET,kind,config['max_length'],vision,verify=True,answer=assistant_target(r,kind),
                    keys=FACT_KEYS if kind=='facts' else (),forced=FORCED_PREFIX[kind])
                self.assertEqual(int(items[kind]['vm'].sum()),1)
            self.assertEqual(input_digest(items['direct']),input_digest(items['facts']))
            f=items['facts']
            for key in FACT_KEYS:self.assertGreater(int(f['detail']['key_'+key].sum()),0,key)
            self.assertLessEqual(f['fields']['input_ids'].shape[1],config['max_length'])
            with self.assertRaisesRegex(ValueError,'Not a token boundary'):
                encode(self.processor,r,DATASET,'facts',config['max_length'],vision,answer=assistant_target(r,'facts'),keys=FACT_KEYS)

    def test_v4_default_is_unchanged(self):
        from verifier_training.fd1000_run import encode
        self.assertIsNone(inspect.signature(encode).parameters['forced'].default)


if __name__=='__main__':unittest.main()
