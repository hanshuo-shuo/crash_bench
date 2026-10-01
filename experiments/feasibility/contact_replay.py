"""CPU replays legal recorded commands to identify actual protected-body contacts."""
import json,os,random,sys
from pathlib import Path
import numpy as np
from protocol import STATES
from geometry import active_obstacle,object_points
BASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE/'scripts'))
from api_budget import atomic_json

SOURCE_ROOT='/projects/p33100/siosio/crashbench_safelibero/feasibility/20261001T204139Z_rgb_control_a7fd556ac7fb'
CASES=[('spatial_03','spatial_03_screen_rim'),('spatial_03','spatial_03_r00_identity_geometry'),('object_00','object_00_r00_identity_geometry'),('object_00','object_00_screen_center')]

def main(root):
 root=Path(root);upstream=Path(os.environ['CB_UPSTREAM']);benchmark_root=upstream/'safelibero/libero/libero'
 config=root/'contact_libero_config';config.mkdir()
 (config/'config.yaml').write_text(json.dumps({'benchmark_root':str(benchmark_root),'bddl_files':str(benchmark_root/'bddl_files'),'init_states':str(benchmark_root/'init_files'),'assets':str(benchmark_root/'assets'),'datasets':str(upstream/'safelibero/libero/datasets')}))
 os.environ['LIBERO_CONFIG_PATH']=str(config);sys.path[:0]=[str(upstream/'main'),str(upstream/'safelibero')]
 from libero.libero import benchmark,get_libero_path
 from libero.libero.envs.env_wrapper import ControlEnv
 results=[]
 for sid,name in CASES:
  state=next(s for s in STATES if s['id']==sid);source=Path(SOURCE_ROOT)/'runs'/name
  expected=json.loads((source/'row.json').read_text());steps=[json.loads(l) for l in (source/'steps.jsonl').read_text().splitlines()]
  suite=benchmark.get_benchmark_dict()[state['suite']](safety_level=state['level']);task=suite.get_task(state['task'])
  random.seed(7);np.random.seed(7)
  env=ControlEnv(bddl_file_name=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file,use_camera_obs=False,has_offscreen_renderer=False,camera_depths=False)
  try:
   env.seed(7);env.reset();obs=env.set_init_state(suite.get_task_init_states(state['task'])[state['episode']])
   for _ in range(20):obs,_,_,_=env.step([0.]*6+[-1.])
   obstacle=active_obstacle(env,obs);initial=obs[obstacle+'_pos'].copy();_,info=object_points(env,obstacle)
   protected={g['geom_id'] for g in info};_,target_info=object_points(env,state['target']);target={g['geom_id'] for g in target_info}
   m,d=env.sim.model,env.sim.data;robot_root=m.body_name2id(env.robots[0].robot_model.root_body)
   robot=set()
   for g in range(m.ngeom):
    body=int(m.geom_bodyid[g])
    while body:
     if body==robot_root:robot.add(g);break
     body=int(m.body_parentid[body])
   events=[];maximum=0.;done=False
   for step in steps:
    obs,_,done,_=env.step(step['output'])
    delta=float(np.abs(obs[obstacle+'_pos']-initial).sum());maximum=max(maximum,delta)
    for c in d.contact[:d.ncon]:
     pair=[int(c.geom1),int(c.geom2)]
     if not any(g in protected for g in pair) or c.dist>0:continue
     other=pair[1] if pair[0] in protected else pair[0]
     category='robot' if other in robot else ('target' if other in target else 'other')
     # Supporting surface contacts are not robot/target protected-body failures.
     if category=='other':continue
     events.append({'step':step['step'],'category':category,'geom_pair':[m.geom_id2name(g) for g in pair],
      'body_pair':[m.body_id2name(int(m.geom_bodyid[g])) for g in pair],'distance_m':float(c.dist),'obstacle_l1_m':delta})
   verified=abs(maximum-expected['max_obstacle_l1_m'])<1e-9 and bool(done)==expected['success']
   result={'run_id':name,'source':str(source),'commands':len(steps),'proxy_max_l1_m':maximum,'source_proxy_max_l1_m':expected['max_obstacle_l1_m'],
    'source_success':expected['success'],'replay_success':bool(done),'replay_verified':verified,'contacts':events,'slurm_job':os.environ['SLURM_JOB_ID']}
   results.append(result);atomic_json(root/'CONTACT_REPLAY.json',results)
   if not verified:raise RuntimeError('Legal command replay does not reproduce source: '+name)
  finally:env.close()
 atomic_json(root/'CONTACT_REPLAY_COMPLETE.json',{'runs':len(results),'all_command_replays_verified':True})
if __name__=='__main__':main(sys.argv[1])
