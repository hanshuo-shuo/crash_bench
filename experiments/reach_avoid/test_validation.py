import copy
import unittest
from validation import verify_steps


def records():
    pose=dict(target=[0,0,0],site_position=[0,0,0],site_rotation=[[1,0,0],[0,1,0],[0,0,1]],site_size=[1,1,1])
    step=dict(step=0,action=None,native_pose=pose,synchronized_pose=pose,
        native_success=True,synchronized_success=True,fixed_goal_success=True,
        contract_success=True,safe_success=True,safe_history=True)
    summary=dict(steps=0,execution_horizon_T=300,safe_success=True,safe_history=True,
        initial_valid=True,label='feasible',outcome='safe_completion')
    return [step],summary


class PersistedEvidence(unittest.TestCase):
    def verify(self,steps,summary):verify_steps(steps,summary,[[-1]*3,[1]*3],dict(label='unknown'),300)
    def test_valid_terminal_witness(self):self.verify(*records())
    def test_outside_goal_rejects_success(self):
        steps,summary=records();steps[0]['synchronized_pose']=copy.deepcopy(steps[0]['synchronized_pose'])
        steps[0]['synchronized_pose']['target']=[2,0,0]
        with self.assertRaises(RuntimeError):self.verify(steps,summary)
    def test_beyond_horizon_rejected(self):
        initial,summary=records();steps=[]
        for i in range(302):
            row=copy.deepcopy(initial[0]);row['step']=i;row['action']=[0]*7;steps.append(row)
        summary['steps']=301
        with self.assertRaises(RuntimeError):self.verify(steps,summary)
    def test_duplicate_initial_step_rejected(self):
        steps,summary=records();steps.append(copy.deepcopy(steps[0]));summary['steps']=1
        with self.assertRaises(RuntimeError):self.verify(steps,summary)
    def test_count_and_label_mismatch_rejected(self):
        for key,value in [('steps',1),('label','infeasible'),('safe_success',False),('outcome','collision')]:
            steps,summary=records();summary[key]=value
            with self.assertRaises(RuntimeError):self.verify(steps,summary)


if __name__=='__main__':unittest.main()
