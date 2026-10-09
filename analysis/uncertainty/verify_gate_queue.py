"""Execute frozen gate queue/replan code with deterministic mock inputs; no rollout."""
import argparse
import ast
import json
import time
from pathlib import Path
from types import SimpleNamespace
from collections import deque
from common import config,atomic_json,sha
from gate import adapt_gate,install


def verify(upstream,output):
    import numpy as np
    from scipy.spatial.transform import Rotation
    original=upstream.read_text();adapted=adapt_gate(original,'gate')
    loops=[n for n in ast.walk(ast.parse(adapted)) if isinstance(n,ast.While) and isinstance(n.test,ast.Compare) and any(isinstance(x,ast.Name) and x.id=='max_steps' for x in n.test.comparators)]
    if len(loops)!=1:raise RuntimeError('Frozen evaluator action loop ambiguous')
    program=compile(ast.fix_missing_locations(ast.Module(body=loops,type_ignores=[])),str(upstream),'exec')
    runtime=install();GateObserver=runtime.Observer;cutoff=.11627746954699222
    class Observer(GateObserver):
        def before(self,values):self.boundaries.append(values['t'])
        def candidate(self,raw,applied,t,status):
            self.commands.append(dict(step=t+1,raw=raw.copy(),applied=applied.copy(),active=self.gate_active,status=status))
        def after(self,obs,done,t):self.after_steps.append(t+1)
        def caught(self,error,t):raise RuntimeError('Synthetic frozen-loop execution failed') from error
    observer=object.__new__(Observer);observer.cfg=dict(gate_threshold=cutoff);observer.commands=[];observer.boundaries=[];observer.after_steps=[]
    def observation(step):
        return dict(agentview_image=np.full((2,2,3),step,dtype=np.uint8),robot0_eye_in_hand_image=np.full((2,2,3),step,dtype=np.uint8),
            robot0_eef_pos=np.array([step*.001,0.,.3]),robot0_eef_quat=np.array([0.,0.,0.,1.]),robot0_gripper_qpos=np.zeros(2),protected_pos=np.array([0.,0.,.3]))
    class Environment:
        def __init__(self):
            self.steps=[];self.sim=SimpleNamespace(model=SimpleNamespace(body_pos=np.zeros((1,3)),body_quat=np.zeros((1,4))))
        def step(self,action):
            self.steps.append(action);return observation(len(self.steps)),0.,False,{}
    class Client:
        def __init__(self):self.elements=[];self.chunks=[]
        def infer(self,element):
            self.elements.append({k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in element.items()})
            index=len(self.elements);chunk=np.arange(1,71,dtype=np.float32).reshape(10,7)+100*index;self.chunks.append(chunk.copy())
            observer.latest=dict(step=1+5*(index-1),disagreement=cutoff+.01 if index==1 else cutoff)
            return dict(actions=chunk)
    env=Environment();client=Client()
    namespace=dict(np=np,R=Rotation,time=time,observer=observer,env=env,client=client,args=SimpleNamespace(resize_size=2,replan_steps=5),
        image_tools=SimpleNamespace(resize_with_pad=lambda x,*_:x,convert_to_uint8=lambda x:x),_quat2axisangle=lambda q:np.zeros(3),
        obs=observation(0),action_plan=deque(),t=0,max_steps=10,replay_images=[],task_description='synthetic mock; no benchmark execution',
        flag_safety_control=False,collide_flag=False,initial_obstacle_pos=np.array([0.,0.,.3]),obstacle_name='protected',eef_body_id=0,
        task_successes=0,total_successes=0)
    exec(program,namespace)
    assert len(client.elements)==2 and len(env.steps)==10 and namespace['t']==10 and not namespace['action_plan']
    assert observer.after_steps==list(range(1,11))
    for i,item in enumerate(observer.commands):
        raw=client.chunks[i//5][i%5]
        assert np.array_equal(item['raw'],raw)
        if i<5:
            assert item['active'] and item['status']=='gate' and np.array_equal(item['applied'],np.zeros(7)) and env.steps[i]==[0.]*7
        else:
            assert not item['active'] and item['status']=='off' and np.array_equal(item['applied'],raw) and np.array_equal(env.steps[i],raw)
    assert not np.array_equal(client.elements[0]['observation/state'],client.elements[1]['observation/state'])
    assert not np.array_equal(client.elements[0]['observation/image'],client.elements[1]['observation/image'])
    observer.latest=dict(step=6,disagreement=cutoff)
    try:observer.apply_gate(client.chunks[1][0],10)
    except runtime.Failure:pass
    else:raise RuntimeError('Queue offset beyond fifth command was accepted')
    proof=dict(passed=True,scope='deterministic_mock_environment; no simulator/no model/no scientific_rollout/no API',upstream_source_sha256=sha(upstream),
        gate_source_sha256=sha(Path(__file__).parent/'gate.py'),threshold=cutoff,synthetic_actions=10,gated_actions=5,policy_calls='two mocked calls',
        above_threshold_all_seven_commands_zero=True,equality_threshold_not_gated=True,original_queue_consumed_five_each=True,
        replan_at_action6_uses_new_image_and_state=True,raw_chunks_unchanged=True,invalid_sixth_queue_offset_rejected=True,
        physics_claim='No mechanical rest claim; mock observation advances independently of zero commands')
    atomic_json(output,proof);print(json.dumps(proof,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('upstream',type=Path);p.add_argument('output',type=Path);a=p.parse_args();verify(a.upstream,a.output)
