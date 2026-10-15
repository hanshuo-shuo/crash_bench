import unittest
import numpy as np
from verifier_dataset.constraint_shortcuts import color_interaction_features


class ColorOnlyReadoutColumns(unittest.TestCase):
    def test_centroid_readout_contains_no_rgb_tiles_or_area(self):
        x=np.arange(261)[None,:]
        actual=color_interaction_features(x,'color_centroids')
        base=[111,112,116,117,121,122,126,127]
        self.assertEqual(actual.tolist(),[base+[130+i for i in base]+[260]])
        self.assertEqual(actual.shape,(1,17))
        self.assertEqual(color_interaction_features(x,'color_statistics').shape,(1,41))
        with self.assertRaises(RuntimeError):color_interaction_features(np.zeros((1,260)),'color_centroids')


if __name__=='__main__':unittest.main()
