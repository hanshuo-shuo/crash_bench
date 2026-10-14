import copy
import tempfile
import types
from pathlib import Path
import unittest
from unittest.mock import patch
from verifier_dataset.reciprocal_policy import check_reciprocal
from verifier_dataset.paired_policy import prompt_for


class ReciprocalIdentity(unittest.TestCase):
    def pair(self):
        a=dict(matched_pair=True,primary_goal=['in','milk','basket'],enclosed_goal=['in','butter','basket'],
            fixed_destination=[[0,0,0],[1,1,1]],source=dict(source_id='one'),checkpoint=0,remaining_horizon=300,
            initial_physics_sha256='same',safe_history=dict(primary=True,enclosed=True))
        b=copy.deepcopy(a);b['primary_goal'],b['enclosed_goal']=a['enclosed_goal'],a['primary_goal']
        return a,b

    def test_same_objects_must_succeed_with_both_roles(self):
        a,b=self.pair();self.assertTrue(check_reciprocal(a,b))
        for key,value in [('primary_goal',['in','a_different_object','basket']),('remaining_horizon',301),
                          ('initial_physics_sha256','changed'),('source',dict(source_id='another'))]:
            changed=copy.deepcopy(b);changed[key]=value
            self.assertFalse(check_reciprocal(a,changed))
        b['safe_history']['enclosed']=False
        self.assertFalse(check_reciprocal(a,b))

    def test_policy_prompt_identifies_requested_object_and_known_receiver(self):
        self.assertEqual(prompt_for(['in','butter_1','basket_1_contain_region']),'put the butter in the basket')
        with self.assertRaisesRegex(RuntimeError,'unambiguous'):
            prompt_for(['on','unknown_object_1','plate_1'])


class NativePolicyQueue(unittest.TestCase):
    def test_native_replanning_discards_unused_future_commands(self):
        try:import numpy as np
        except ImportError:self.skipTest('Run with bundled or Quest NumPy')
        from verifier_dataset.paired_policy import Policy
        class Client:
            def __init__(self,*args):self.n=0;self._ws=types.SimpleNamespace(close=lambda:None)
            def infer(self,data):
                if '__uncertainty_reset__' in data:
                    self.run=data['run_id'];return dict(run_id=self.run,seed=data['seed'],requests=0)
                offset=self.n;self.n+=1
                return dict(actions=np.tile(np.arange(10)[:,None]/10+offset,(1,7)),
                    uncertainty=dict(run_id=self.run,infer_index=self.n))
        module=types.ModuleType('openpi_client.websocket_client_policy');module.WebsocketClientPolicy=Client
        data={'observation/image':np.zeros((2,2,3),dtype=np.uint8),
              'observation/wrist_image':np.zeros((2,2,3),dtype=np.uint8),
              'observation/state':np.zeros(8),'prompt':'task'}
        with tempfile.TemporaryDirectory() as tmp, patch.dict('sys.modules',{'openpi_client.websocket_client_policy':module}), \
                patch('verifier_dataset.paired_policy.policy_input',return_value=data):
            env=types.SimpleNamespace(env=types.SimpleNamespace(_get_observations=lambda **kwargs:{}))
            policy=Policy(env,Path(tmp),dict(source_id='source',seed=7),dict(remaining_horizon=7),
                          ['in','butter_1','basket_1_contain_region'],1234,'run')
            try:
                actions=list(policy.commands(7))
                self.assertEqual([round(x['applied'][0],2) for x in actions],[0,.1,.2,.3,.4,1.,1.1])
                self.assertEqual(policy.requests,2)
            finally:policy.close()


if __name__=='__main__':unittest.main()
