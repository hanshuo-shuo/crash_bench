import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from baseline import prompt_check


class FakeClient:
    def infer(self,request):
        n=len(request['prompt'])
        return dict(prefix_final=np.array([n,2.]),image_embedding=np.array([1.,2.]),
            image_prefix_final=np.array([n,4.]),valid_tokens=np.array(10),valid_image_tokens=np.array(8),
            actions=np.ones((5,7))*n,tokenized_prompt=np.array([1,n,2]),tokenized_prompt_mask=np.array([True]*3),
            metadata=dict(prompt=request['prompt']),server_timing=dict(infer_ms=12.))


class PromptTransport(unittest.TestCase):
    def test_transport_dict_is_not_a_feature_array(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_check(FakeClient(),{},Path(tmp));folder=Path(tmp)/'prompt_check'
            with np.load(folder/'original.npz',allow_pickle=False) as arrays:
                self.assertNotIn('server_timing',arrays.files)
                for key in arrays.files:self.assertNotEqual(arrays[key].dtype.kind,'O')
            meta=json.loads((folder/'original.json').read_text())
            self.assertIn('server_timing',meta['transport_metadata'])
            contrasts=json.loads((folder/'CONTRASTS.json').read_text())['contrasts']
            self.assertEqual(contrasts['duplicate_minus_original']['actions']['linf'],0.)
            self.assertGreater(contrasts['safety_minus_original']['actions']['linf'],0.)


if __name__=='__main__':unittest.main()
