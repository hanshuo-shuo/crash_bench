import tempfile
import unittest
from pathlib import Path
try:
    import numpy as np
    from PIL import Image
except ImportError:
    np=None
from scripts.openrouter_perception import export_request
from verifier_dataset.frozen_perception import check_inputs
from verifier_dataset.replay import ARRAYS


@unittest.skipIf(np is None,'NumPy/Pillow required; run with the bundled or Quest Python')
class FrozenInputEvidence(unittest.TestCase):
    def prepare(self,root,delta=1):
        import io
        def png(pixel):
            output=io.BytesIO();x=np.zeros((4,4,3),dtype=np.uint8);x[0,0,0]=pixel
            Image.fromarray(x).save(output,format='PNG');return output.getvalue()
        expected=export_request(png(0),'task','suite',root/'old',budgeted=True)
        actual=export_request(png(delta),'task','suite',root/'new',budgeted=True)
        for name in ('a.npz','b.npz'):np.savez(root/name,**{k:np.zeros(2) for k in ARRAYS})
        return actual,expected,root/'a.npz',root/'b.npz'

    def test_only_quantization_drift_with_equal_physical_arrays_is_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.prepare(root)
            result=check_inputs(*args)
            self.assertEqual(result['changed_pixels'],1)
            self.assertFalse(result['full_model_camera_identity_claim'])
            self.assertFalse((args[0]/'response.json').exists())
            np.savez(root/'a.npz',**{k:np.ones(2) for k in ARRAYS})
            with self.assertRaisesRegex(RuntimeError,'physical arrays differ'):check_inputs(*args)

    def test_larger_image_difference_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,'quantization level'):
                check_inputs(*self.prepare(Path(tmp),delta=2))


if __name__=='__main__':unittest.main()
