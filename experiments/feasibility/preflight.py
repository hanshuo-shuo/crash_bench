"""CPU Slurm audit only: no renderer, no model inference, no benchmark labels."""
import hashlib,json,os,random,sys
from pathlib import Path
import numpy as np
from protocol import STATES,independent_caption
from geometry import active_obstacle,verified_ellipsoid
BASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE/'scripts'))
from api_budget import atomic_json
from serve import ControlledPolicy,digest

def server_check():
 class Random:
  @staticmethod
  def key(seed):return np.array([0,seed],dtype=np.uint32)
  @staticmethod
  def key_data(key):return key
 class FakeJax:random=Random
 class Policy:
  def __init__(self):self._rng=Random.key(0);self.calls=0
  def infer(self,obs):
   self._rng=self._rng+np.array([0,1],dtype=np.uint32);self.calls+=1
   return {'actions':np.ones((5,7))*self.calls}
 p=Policy();server=ControlledPolicy(p,FakeJax)
 obs={'observation/state':np.array([1.,2.]),'prompt':'fixed'}
 server.infer({'__paired_reset_rng__':42,'run_id':'a'});a=server.infer(obs)
 server.infer({'__paired_reset_rng__':42,'run_id':'b'});b=server.infer(obs)
 assert p.calls==2 and np.array_equal(a['actions'],b['actions'])
 assert b['diagnostic']['native_output_linf_difference']==1.
 assert a['diagnostic']['rng_after']==b['diagnostic']['rng_after']
 assert digest(obs)!=digest(dict(obs,**{'observation/state':np.array([1.,3.])}))
 return {'passed':True,'native_calls':p.calls,'controlled_output_exact':True,'native_discrepancy_recorded':True}

def main(root):
 root=Path(root);upstream=Path(os.environ['CB_UPSTREAM']);benchmark_root=upstream/'safelibero/libero/libero'
 config=root/'preflight_libero_config';config.mkdir()
 (config/'config.yaml').write_text(json.dumps({'benchmark_root':str(benchmark_root),'bddl_files':str(benchmark_root/'bddl_files'),'init_states':str(benchmark_root/'init_files'),'assets':str(benchmark_root/'assets'),'datasets':str(upstream/'safelibero/libero/datasets')}))
 os.environ['LIBERO_CONFIG_PATH']=str(config)
 sys.path[:0]=[str(upstream/'main'),str(upstream/'safelibero')]
 from libero.libero import benchmark,get_libero_path
 from libero.libero.envs.env_wrapper import ControlEnv
 result={'job':os.environ['SLURM_JOB_ID'],'code_commit':os.environ['CB_CODE_COMMIT'],'server':server_check(),'states':[]}
 for state in STATES:
  task_suite=benchmark.get_benchmark_dict()[state['suite']](safety_level=state['level']);task=task_suite.get_task(state['task'])
  random.seed(7);np.random.seed(7)
  env=ControlEnv(bddl_file_name=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file,use_camera_obs=False,has_offscreen_renderer=False,camera_depths=False)
  try:
   env.seed(7);env.reset();obs=env.set_init_state(task_suite.get_task_init_states(state['task'])[state['episode']])
   for _ in range(20):obs,_,_,_=env.step([0.]*6+[-1.])
   obstacle=active_obstacle(env,obs)
   if state['target']+'_pos' not in obs or state['goal']+'_pos' not in obs:raise RuntimeError('Missing reference target/goal '+str(state))
   p,R,axes,points,info=verified_ellipsoid(env,obstacle)
   controller=env.robots[0].controller
   output_min=np.asarray(controller.output_min);output_max=np.asarray(controller.output_max)
   if not np.allclose(output_max[:3],.05) or not np.allclose(output_max[3:],.5):raise RuntimeError('Reference action inversion does not match native OSC scale')
   result['states'].append({'id':state['id'],'task_description':task.language,'obstacle':obstacle,'correct_caption':independent_caption(obstacle),'target_pos':obs[state['target']+'_pos'].tolist(),'goal_pos':obs[state['goal']+'_pos'].tolist(),'ellipsoid':dict(info,p=p.tolist(),R=R.tolist(),axes=axes.tolist()),'controller_output_min':output_min.tolist(),'controller_output_max':output_max.tolist(),'controller_control_delta':bool(controller.use_delta) if hasattr(controller,'use_delta') else getattr(controller,'control_delta',None),'action_spec':[np.asarray(x).tolist() for x in env.env.action_spec]})
   atomic_json(root/'PREFLIGHT.json',result)
  finally:env.close()
 result['status']='passed';atomic_json(root/'PREFLIGHT.json',result)
if __name__=='__main__':
 try:main(sys.argv[1])
 except BaseException as e:
  atomic_json(Path(sys.argv[1])/'PREFLIGHT_FAILED.json',{'reason':type(e).__name__+': '+str(e)});raise
