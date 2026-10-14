from pathlib import Path
import tempfile
import unittest
from verifier_training.common import write
from verifier_training.fd1000_contract import validate,configuration


class ScientificDataGate(unittest.TestCase):
    def test_engineering_or_unreviewed_data_cannot_start_training(self):
        for gate in ({'passed':True,'engineering_supervision_only':True},
                     {'passed':True,'scientific_training_allowed':False},
                     {'passed':False,'scientific_training_allowed':True}):
            with tempfile.TemporaryDirectory() as name:
                root=Path(name);write(root/'ACCEPTANCE.json',gate)
                with self.assertRaisesRegex(ValueError,'has not authorized'):validate(root)

    def test_changed_dataset_fails_before_any_model_load(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);(root/'dataset.jsonl').write_text('tampered')
            write(root/'ACCEPTANCE.json',dict(passed=True,scientific_training_allowed=True,dataset_sha256='not-the-file'))
            with self.assertRaisesRegex(ValueError,'evidence changed'):validate(root)

    def test_primary_generation_has_no_gold_rationale_prefix(self):
        config=configuration()
        self.assertIn('no gold rationale prefix',config['evaluation'])
        self.assertFalse(config['hyperparameter_search'])
        self.assertTrue(config['inference_inputs_identical'])


if __name__=='__main__':unittest.main()
