import copy
import unittest
import xml.etree.ElementTree as ET
from verifier_dataset.constraint_evidence import CONTRACT,verify_trace,verify_layout,rows
from verifier_dataset.constraint_visuals import add,edge_boxes


class ConstraintEvidence(unittest.TestCase):
    def fixture(self):
        material=dict(geom_type=6,primitive_size=[.01,.01,.01],radius=.01,center_in_root=[0.,0.,0.],rigid_primitive_verified=True)
        layout=dict(contract=CONTRACT,a=[[-.1,-.1,0.],[.1,.1,.2]],b=[[-.1,.4,0.],[.1,.6,.2]],
            goal=[[.45,-.05,0.],[.55,.05,.2]],target=[[-.01,-.01,.09],[.01,.01,.11]],
            material=material,native_goal=['on','object','plate_1'],native_obstacle='bottle',
            checkpoint=0,sample_index=0,remaining_horizon=10)
        geometry=dict(protected=[100],actors=[10,11],goal_definition=dict(goal=layout['native_goal'],kind='object',
            target_contact_geoms=[11],receiver_contact_geoms=[20]))
        samples=[]
        for phase,step,substep,x,done in [('initial',0,0,0.,False),('integration',1,1,.3,False),('endpoint',1,1,.5,True)]:
            row=dict(phase=phase,step=step,substep=substep,native_predicate=done,safe_history=True)
            for mode in ('native','synchronized'):
                row[mode+'_point']=[x,0.,.1];row[mode+'_root']=[x,0.,.1]
                row[mode+'_goal_state']=dict(target_position=[x,0.,.1],receiver_position=[.5,0.,0.])
                row[mode+'_goal_contacts']=[[11,20,0.]] if done else []
                row[mode+'_goal_success']=done;row[mode+'_protected_contacts']=[]
            samples.append(row)
        actions=[dict(step=1,action=[0.]*7,source_physics_equal=True)]
        return layout,geometry,samples,actions

    def test_same_trajectory_witness_and_opposite_rule_certificate(self):
        layout,geometry,samples,actions=self.fixture()
        self.assertTrue(verify_trace(samples,actions,geometry)['native_contact_safe'])
        result=verify_layout(layout,samples,actions)
        self.assertTrue(result['matched_pair']);self.assertEqual(result['witness_commands'],1)
        self.assertTrue(result['blocked_boundary_crossed_by_positive_witness'])

    def test_policy_failure_never_becomes_a_negative(self):
        layout,geometry,samples,actions=self.fixture()
        layout['b']=[[.2,-.1,0.],[.4,.1,.2]]
        result=verify_layout(layout,samples,actions)
        self.assertFalse(result['positive_witness']);self.assertFalse(result['matched_pair'])
        self.assertTrue(result['conditional_negative'])  # Independent a certificate still exists.
        layout['goal']=[[-.05,-.05,.05],[.05,.05,.15]]
        result=verify_layout(layout,samples,actions)
        self.assertFalse(result['conditional_negative'])

    def test_missing_samples_contact_and_fabricated_success_fail(self):
        layout,geometry,samples,actions=self.fixture()
        with self.assertRaises(RuntimeError):verify_trace([samples[0],samples[-1]],actions,geometry)
        bad=copy.deepcopy(samples);bad[1]['native_protected_contacts']=[[100,11,-.0001]]
        with self.assertRaisesRegex(RuntimeError,'history'):verify_trace(bad,actions,geometry)
        bad=copy.deepcopy(samples);bad[-1]['synchronized_goal_contacts']=[]
        with self.assertRaisesRegex(RuntimeError,'formula'):verify_trace(bad,actions,geometry)
        actions[0]['source_physics_equal']=False
        with self.assertRaisesRegex(RuntimeError,'physically different'):verify_trace(samples,actions,geometry)

    def test_same_image_text_and_object_category_balanced(self):
        layout,geometry,samples,actions=self.fixture();proof=verify_layout(layout,samples,actions)
        source=dict(source_id='s',task_group='t',family='f',method='aegis',seed=7)
        data=rows(layout,proof,source,[[{'sha256':'image0'}],[{'sha256':'image1'}]])
        self.assertEqual(len(data),4)
        for i in (0,2):
            self.assertEqual(data[i]['student']['images'],data[i+1]['student']['images'])
            self.assertEqual(data[i]['student']['task'],data[i+1]['student']['task'])
            self.assertNotEqual(data[i]['evidence']['feasibility'],data[i+1]['evidence']['feasibility'])
        self.assertEqual(data[0]['student']['constraints'],data[3]['student']['constraints'])
        self.assertNotEqual(data[0]['evidence']['feasibility'],data[3]['evidence']['feasibility'])
        self.assertTrue(all(r['metadata']['category_identical_across_labels'] for r in data))

    def test_only_nonphysical_world_annotations_are_added(self):
        tree=ET.fromstring(add('<mujoco><worldbody><body name="object"/></worldbody></mujoco>'))
        geoms=tree.findall('worldbody/geom')
        self.assertEqual(len(geoms),48)
        self.assertTrue(all(g.get('contype')=='0' and g.get('conaffinity')=='0' for g in geoms))
        self.assertEqual(len(edge_boxes([[-1.,-1.,-1.],[1.,1.,1.]])),12)


if __name__=='__main__':unittest.main()
