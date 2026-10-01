"""Fixed privileged waypoint reference, only ordinary 7D OSC/gripper actions."""
import numpy as np
from scipy.spatial.transform import Rotation
class Reference:
 def __init__(self,state,variant='center'):
  self.state=state;self.variant=variant;self.phase='clear_height';self.phase_steps=0;self.grasp=None;self.offset=None;self.attached=False;self.high=None;self.clear_orientation=None
 def step(self,env,obs):
  self.phase_steps+=1
  eef=np.asarray(obs['robot0_eef_pos']);obj=np.asarray(obs[self.state['target']+'_pos']);goal=np.asarray(obs[self.state['goal']+'_pos'])
  # Fixed high transit plane; no object/obstacle reset or simulator state writes.
  if self.high is None:
   from geometry import active_obstacle,object_points
   points,_=object_points(env,active_obstacle(env,obs))
   self.clear_orientation=Rotation.from_quat(obs['robot0_eef_quat']).as_matrix()
   self.high=max(float(obj[2])+.26,float(goal[2])+.26,float(points[:,2].max())+.12)
   target_object=env.env.objects_dict[self.state['target']]
   if env.env._check_grasp(env.robots[0].gripper,target_object):
    self.phase='lift';self.phase_steps=0;self.grasp=eef.copy();self.offset=eef-obj
    self.orientation=Rotation.from_quat(obs['robot0_eef_quat']).as_matrix();self.attached=True
  high=self.high
  grasp=obj.copy()
  grasp[2]+= .025 if 'bowl' in self.state['target'] else .015
  if self.variant=='rim' and 'bowl' in self.state['target']:grasp[1]-=.035
  if self.variant=='side':grasp[2]-=.025
  grip=-1.
  if self.phase=='clear_height':target=np.array([eef[0],eef[1],high]);orientation=self.clear_orientation
  elif self.phase=='approach':target=np.array([grasp[0],grasp[1],high]);orientation=Rotation.from_euler('xyz',[np.pi,0,0]).as_matrix()
  elif self.phase=='descend':target=grasp;orientation=Rotation.from_euler('xyz',[np.pi,0,0]).as_matrix()
  elif self.phase=='close':target=self.grasp;grip=1.;orientation=Rotation.from_euler('xyz',[np.pi,0,0]).as_matrix()
  elif self.phase=='lift':target=np.array([eef[0],eef[1],high]);grip=1.;orientation=self.orientation
  elif self.phase=='transit':target=np.array([goal[0]+self.offset[0],goal[1]+self.offset[1],high]);grip=1.;orientation=self.orientation
  elif self.phase=='place':
   target=goal.copy();target[:2]+=self.offset[:2];target[2]+=(.06 if 'plate' in self.state['goal'] else .12)+(self.offset[2] if self.offset is not None else .05);grip=1.;orientation=self.orientation
  elif self.phase=='release':target=self.place.copy();orientation=self.orientation
  else:target=np.array([eef[0],eef[1],high]);orientation=self.orientation
  if self.phase=='clear_height' and (abs(eef[2]-high)<.012 or self.phase_steps>=45):self.phase='approach';self.phase_steps=0
  elif self.phase in ['approach','descend'] and (np.linalg.norm(target-eef)<.012 or self.phase_steps>=60):
   if self.phase=='approach':self.phase='descend'
   else:self.phase='close';self.grasp=eef.copy()
   self.phase_steps=0
  elif self.phase=='close' and self.phase_steps>=12:
   self.phase='lift';self.phase_steps=0;self.offset=eef-obj;self.orientation=Rotation.from_quat(obs['robot0_eef_quat']).as_matrix()
  elif self.phase=='lift' and (abs(eef[2]-high)<.015 or self.phase_steps>=45):self.phase='transit';self.phase_steps=0
  elif self.phase=='transit' and (np.linalg.norm(target-eef)<.015 or self.phase_steps>=60):self.phase='place';self.phase_steps=0
  elif self.phase=='place' and (np.linalg.norm(target-eef)<.015 or self.phase_steps>=45):self.phase='release';self.phase_steps=0;self.place=eef.copy()
  elif self.phase=='release' and self.phase_steps>=15:self.phase='retreat';self.phase_steps=0
  current=Rotation.from_quat(obs['robot0_eef_quat']).as_matrix()
  omega=Rotation.from_matrix(orientation@current.T).as_rotvec()
  action=np.r_[np.clip((target-eef)/.05,-1,1),np.clip(omega/.5,-1,1),grip]
  return action
