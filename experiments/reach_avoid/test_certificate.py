import math
import unittest
from certificate import certify, classify, inscribed_radius, rectangle_covered


def walls():
    out = []
    for axis in range(3):
        for side in (-1, 1):
            lo = [-1.1]*3; hi = [1.1]*3
            if side == -1: hi[axis] = -1
            else: lo[axis] = 1
            out.append(dict(lower=lo, upper=hi))
    return out


class CertificateSafety(unittest.TestCase):
    def cert(self, boxes, **kwargs):
        args = dict(boxes=boxes, lower=[-1.]*3, upper=[1.]*3, initial_root=[0.]*3,
            fixed_goal=[[2.]*3,[3.]*3], radius=.03, initial_safe=True, all_static=True,
            material_verified=True, guard_active=True)
        args.update(kwargs); return certify(**args)

    def test_complete_separator_and_missing_face(self):
        self.assertEqual(self.cert(walls())['label'], 'infeasible')
        self.assertEqual(self.cert(walls()[:-1])['label'], 'unknown')

    def test_narrow_slit_without_gripper_assumption(self):
        for gap, label in [(.02, 'infeasible'), (.04, 'unknown')]:
            boxes = walls()[:-1]
            boxes += [dict(lower=[-1.1,-1.1,1.], upper=[-gap/2,1.1,1.1]),
                      dict(lower=[gap/2,-1.1,1.], upper=[1.1,1.1,1.1])]
            self.assertEqual(self.cert(boxes)['label'], label)

    def test_guard_and_material_are_necessary(self):
        for key in ['guard_active', 'material_verified', 'all_static', 'initial_safe']:
            self.assertEqual(self.cert(walls(), **{key: False})['label'], 'unknown')
        self.assertEqual(self.cert(walls(), radius=0)['label'], 'unknown')

    def test_union_gap_cannot_hide_between_grid_points(self):
        bounds = [0,0,1,1]
        self.assertTrue(rectangle_covered(bounds, [[0,0,.501,1],[.501,0,1,1]]))
        self.assertFalse(rectangle_covered(bounds, [[0,0,.501,1],[.50100001,0,1,1]]))

    def test_material_radius_in_actual_box_coordinates(self):
        self.assertAlmostEqual(inscribed_radius([.02625,.02625,.05475],[-.00037,0,-.015]), .02588)
        self.assertEqual(inscribed_radius([.02]*3,[.03,0,0]), 0)

    def test_solver_failure_never_becomes_negative(self):
        unknown = dict(label='unknown')
        self.assertEqual(classify(False,300,300,unknown),'unknown')
        self.assertEqual(classify(True,301,300,unknown),'unknown')
        self.assertEqual(classify(True,300,300,unknown),'feasible')
        with self.assertRaises(RuntimeError): classify(True,10,300,dict(label='infeasible'))


if __name__ == '__main__': unittest.main()
