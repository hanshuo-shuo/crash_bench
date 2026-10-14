import math
import unittest
from verifier_dataset.fixture_design import walls
from verifier_dataset.paired_geometry import expanded_point_goal,point_certificate


class MaterialPointProof(unittest.TestCase):
    def test_offset_goal_covers_every_rotation(self):
        box=[[.4,-.1,.2],[.5,.1,.3]];offset=[.03,.04,0.]
        enlarged=expanded_point_goal(box,offset)
        self.assertAlmostEqual(enlarged[0][0],.35)
        self.assertAlmostEqual(enlarged[1][2],.35)
        for angle in range(360):
            x=.05*math.cos(math.radians(angle));y=.05*math.sin(math.radians(angle))
            self.assertTrue(enlarged[0][0]<=.4+x<=enlarged[1][0])
            self.assertTrue(enlarged[0][1]<=-.1+y<=enlarged[1][1])

    def test_missing_face_and_goal_overlap_cannot_certify(self):
        lo=[-.1,-.1,0.];hi=[.1,.1,.2];boxes=walls(lo,hi)
        material=dict(radius=.01,center_in_root=[0.,0.,0.],initial_world_center=[0.,0.,.1],rigid_primitive_verified=True)
        far=[[.4,-.01,.1],[.5,.01,.2]]
        self.assertEqual(point_certificate(boxes,lo,hi,material,far,True)['label'],'conditionally_infeasible')
        self.assertEqual(point_certificate(boxes[:-1],lo,hi,material,far,True)['label'],'unknown')
        material['center_in_root']=[.6,0.,0.]
        self.assertEqual(point_certificate(boxes,lo,hi,material,far,True)['label'],'unknown')

    def test_unsafe_initial_state_is_not_an_infeasibility_case(self):
        material=dict(radius=.01,center_in_root=[0.,0.,0.],initial_world_center=[0.,0.,.1],rigid_primitive_verified=True)
        self.assertEqual(point_certificate(walls([-.1,-.1,0.],[.1,.1,.2]),[-.1,-.1,0.],[.1,.1,.2],
            material,[[.4,-.01,.1],[.5,.01,.2]],False)['label'],'unknown')


if __name__=='__main__':unittest.main()
