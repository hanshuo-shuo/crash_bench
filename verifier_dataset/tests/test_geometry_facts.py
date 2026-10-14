import unittest
from verifier_dataset.geometry_facts import box_material_radius


class MaterialGeometry(unittest.TestCase):
    def test_hollow_root_outside_every_collision_box_has_no_material_ball(self):
        self.assertEqual(box_material_radius([.01,.10,.10],[.06,0.,0.]),0.)
        self.assertEqual(box_material_radius([.10,.01,.10],[0.,.06,0.]),0.)
        self.assertAlmostEqual(box_material_radius([.03,.02,.05],[.001,.003,0.]),.017)

    def test_nonfinite_geometry_is_rejected(self):
        with self.assertRaises(ValueError):box_material_radius([float('nan'),1.,1.],[0.,0.,0.])


if __name__=='__main__':unittest.main()
