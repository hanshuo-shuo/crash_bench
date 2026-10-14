import unittest
from verifier_dataset.paired_goal import evaluate


class NativeGoalFormula(unittest.TestCase):
    def test_on_object_requires_contact_and_horizontal_proximity(self):
        spec=dict(goal=['on','target','plate'],kind='object',target_contact_geoms=[1],receiver_contact_geoms=[2])
        state=dict(target_position=[0.,0.,.1],receiver_position=[0.,0.,0.])
        self.assertFalse(evaluate(spec,state,[]))
        self.assertTrue(evaluate(spec,state,[[1,2,.0001]]))
        state['target_position'][0]=.03
        self.assertFalse(evaluate(spec,state,[[1,2,0.]]))

    def test_site_in_is_strict_and_uses_pinned_lower_z_allowance(self):
        spec=dict(goal=['in','target','site'],kind='site',size=[.1,.1,.1],
            target_contact_geoms=[1],receiver_contact_geoms=[],contact_required=False)
        state=dict(target_position=[0.,0.,-.105],receiver_position=[0.,0.,0.],receiver_rotation=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        self.assertTrue(evaluate(spec,state,[]))
        state['target_position']=[.1,0.,0.]
        self.assertFalse(evaluate(spec,state,[]))

    def test_site_on_parent_contact_cannot_be_omitted(self):
        spec=dict(goal=['on','target','site'],kind='site',size=[.1,.1,0.],
            target_contact_geoms=[1],receiver_contact_geoms=[2],contact_required=True)
        state=dict(target_position=[0.,0.,.02],receiver_position=[0.,0.,0.],receiver_rotation=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        self.assertFalse(evaluate(spec,state,[]))
        self.assertTrue(evaluate(spec,state,[[1,2,0.]]))
        spec['contact_required']=False
        self.assertTrue(evaluate(spec,state,[]))


if __name__=='__main__':unittest.main()
