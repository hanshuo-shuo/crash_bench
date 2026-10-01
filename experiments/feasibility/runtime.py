"""Pinned-loop diagnostics with read-only checkpoints and replayed prefixes."""
import hashlib,io,json,os,pickle,random,sys,time,types
from pathlib import Path
import numpy as np
from adapter import adapt
from geometry import active_obstacle,verified_ellipsoid
from protocol import independent_caption,seed_for,CHECKPOINTS
from reference import Reference
from serve import digest
BASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE/'scripts'))
from api_budget import atomic_json
from run_safelibero import CFG,link_offline_bert
class InfrastructureError(RuntimeError):pass

def array_hash(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(str((a.dtype.str,a.shape)).encode()+a.tobytes()).hexdigest()
def serialize(v):
 if isinstance(v,np.ndarray):return {'array':v.tolist(),'dtype':v.dtype.str}
 if isinstance(v,(np.integer,np.floating)):return v.item()
 if isinstance(v,(str,int,float,bool,type(None))):return v
 if isinstance(v,(list,tuple)):return [serialize(x) for x in v]
 if isinstance(v,dict):return {str(k):serialize(x) for k,x in v.items() if not callable(x)}
 return None

class Runner:
 def __init__(self,root,port):
  self.root=Path(root);self.port=port;self.upstream=Path(os.environ['CB_UPSTREAM']);self.assets=Path(os.environ['CB_ASSETS']);self.detector=None
  config=self.root/'libero_config';config.mkdir()
  benchmark=self.upstream/'safelibero/libero/libero'
  (config/'config.yaml').write_text(json.dumps({'benchmark_root':str(benchmark),'bddl_files':str(benchmark/'bddl_files'),'init_states':str(benchmark/'init_files'),'assets':str(benchmark/'assets'),'datasets':str(self.upstream/'safelibero/libero/datasets')}))
  os.environ['LIBERO_CONFIG_PATH']=str(config)
  sys.path[:0]=[str(self.upstream/'main'),str(self.upstream/'safelibero'),str(self.upstream/'openpi/packages/openpi-client/src')]
  self.source=(self.upstream/'main/main_aegis.py').read_text();self.patched=adapt(self.source)
  self.first_inputs={};self.first_chunks={}

 def run(self,state,repeat,condition,name=None,variant='center',branch=None,extra=0,validation=False):
  started=time.monotonic();seed=seed_for(state,repeat,validation)
  name=name or '%s_r%02d_%s'%(state['id'],repeat,condition)
  directory=self.root/'runs'/name;directory.mkdir(parents=True)
  row=dict(run_id=name,state=state['id'],role=state['role'],repeat=repeat,seed=seed,condition=condition,variant=variant,branch_step=branch,
   extra_budget=extra,validation=validation,success=False,collided=False,max_obstacle_l1_m=0.,collision_step=None,end_step=0,exited=False,
   code_commit=os.environ['CB_CODE_COMMIT'],upstream_commit=CFG['upstream_commit'],slurm_job=os.environ['SLURM_JOB_ID'],caption=state['caption'])
  atomic_json(directory/'manifest.json',dict(row,status='started',state_configuration=state))
  envs=[];clients=[];initial={};pending={};frames=[];trace=(directory/'steps.jsonl').open('w');inferences=(directory/'policy.jsonl').open('w')
  previous=Path.cwd();os.chdir(directory)
  (directory/'GroundingDINO').symlink_to(self.assets/'GroundingDINO',target_is_directory=True);link_offline_bert(directory,self.assets)
  ref=Reference(state,variant);prefix_row=None
  baseline_name='%s_r%02d_raw'%(state['id'],repeat)
  if branch is not None:
   prefix_dir=self.root/'runs'/baseline_name
   prefix=[json.loads(l) for l in (prefix_dir/'steps.jsonl').read_text().splitlines()]
   expected_checkpoint=json.loads((prefix_dir/('checkpoint_%03d.json'%branch)).read_text())
  def begin(client):
   ack=client.infer({'__paired_reset_rng__':seed,'run_id':name})
   row['rng_reset']=ack;atomic_json(directory/'rng_reset.json',ack)
  def settled(env,obs,ctx):
   envs.append(env);initial['env']=env;initial['eef']=np.asarray(obs['robot0_eef_pos']).copy()
   qpos=np.asarray(env.sim.data.qpos,dtype='<f8').copy();np.save(directory/'settled_qpos.npy',qpos)
   row['qpos_sha256']=array_hash(qpos);row['settled_qvel_sha256']=array_hash(env.sim.data.qvel)
   initial['obs']=obs
   robot=env.robots[0];control=robot.controller
   row['action_spec']=[np.asarray(x).tolist() for x in env.env.action_spec]
   atomic_json(directory/'controller_config.json',serialize({k:getattr(control,k,None) for k in ['input_min','input_max','output_min','output_max','control_delta','kp','damping','position_limits','orientation_limits']}))
  def geometry(env,obs,description,out):
   obstacle=active_obstacle(env,obs);row['active_obstacle']=obstacle
   identity=condition in ['identity','identity_geometry'];geometry_condition=condition in ['geometry','identity_geometry']
   if branch is not None:identity=geometry_condition=False
   caption=independent_caption(obstacle) if identity else state['caption'];row['used_caption']=caption
   row['identity_changed']=caption!=state['caption']
   if condition in ['nominal','reference'] and branch is None:
    random.seed(7);np.random.seed(7)
    return np.zeros(3),np.eye(3),np.ones(3),False
   cache=self.root/'perception'/state['id'];cache.mkdir(parents=True,exist_ok=True)
   mode=('sim_'+('correct' if identity else 'raw')) if geometry_condition else ('detector_'+hashlib.sha256(caption.encode()).hexdigest()[:8])
   file=cache/(mode+'.npz')
   img=np.ascontiguousarray(obs['agentview_image'][::-1,::-1]);image_hash=array_hash(img)
   if file.exists():
    values=np.load(file);p,R,axes,enabled=values['p'],values['R'],values['axes'],bool(values['enabled'])
    metadata=json.loads(file.with_suffix('.json').read_text())
    if metadata['initial_image_sha256']!=image_hash:raise InfrastructureError('Perception freeze image mismatch')
   else:
    if geometry_condition:
     from protocol import CAPTIONS
     family=next((k for k,v in CAPTIONS.items() if v==caption),None)
     selected=family+'_obstacle_1' if family else None
     if not selected or selected+'_pos' not in obs:raise InfrastructureError('Original caption cannot identify a simulated obstacle')
     p,R,axes,points,metadata=verified_ellipsoid(env,selected);enabled=True
     np.save(cache/(mode+'_vertices.npy'),points)
    else:
     from groundingdino.util.inference import load_model
     from utils import get_point_cloud,filtering_points,fit_ellipse
     if self.detector is None:self.detector=load_model('GroundingDINO/GroundingDINO_SwinT_OGC.py','GroundingDINO/groundingdino_swint_ogc.pth')
     a=get_point_cloud(img,np.ascontiguousarray(obs['agentview_depth'][::-1,::-1]),env,'agentview',caption,self.detector,out)
     b=get_point_cloud(np.ascontiguousarray(obs['backview_image'][::-1,::-1]),np.ascontiguousarray(obs['backview_depth'][::-1,::-1]),env,'backview',caption,self.detector,out)
     points=np.vstack([x for x in [a,b] if x.size]) if a.size or b.size else np.array([[]]);points=filtering_points(points,state['suite'])
     enabled=bool(len(points));p,R,axes=fit_ellipse(points,plot=True,save_path=out) if enabled else (np.zeros(3),np.eye(3),np.ones(3))
     np.save(cache/(mode+'_vertices.npy'),points);metadata={'object_caption':caption,'filtered_points':len(points),'scope':'unchanged GroundingDINO/depth/filter/MVEE'}
    metadata.update(initial_image_sha256=image_hash,caption=caption,p=p.tolist(),R=R.tolist(),axes=axes.tolist(),enabled=enabled)
    np.savez(file,p=p,R=R,axes=axes,enabled=enabled);atomic_json(file.with_suffix('.json'),metadata)
   row['geometry_file']=str(file);row['geometry_sha256']=hashlib.sha256(file.read_bytes()).hexdigest();row['filter_enabled']=enabled
   random.seed(7);np.random.seed(7)
   return p,R,axes,enabled
  def ready(obstacle,position):initial['obstacle']=np.asarray(position).copy()
  def checkpoint(ctx):
   t=ctx['t'];env=ctx['env'];obs=ctx['obs'];initial['obs']=obs
   if t not in CHECKPOINTS and t!=branch:return
   snap=clients[0].infer({'__diagnostic_snapshot__':True})
   robot=env.robots[0]
   control=serialize({k:v for k,v in vars(robot.controller).items() if isinstance(v,(np.ndarray,float,int,bool,str,list,tuple,type(None)))})
   payload={'step':t,'sim_state_sha256':array_hash(env.get_sim_state()),'qpos_sha256':array_hash(env.sim.data.qpos),'qvel_sha256':array_hash(env.sim.data.qvel),
    'ctrl_sha256':array_hash(env.sim.data.ctrl),'warmstart_sha256':array_hash(env.sim.data.qacc_warmstart),
    'marker_position':serialize(env.sim.model.body_pos[ctx['eef_body_id']]),'marker_quaternion':serialize(env.sim.model.body_quat[ctx['eef_body_id']]),'controller':control,'action_queue':[np.asarray(x).tolist() for x in ctx['action_plan']],
    'policy_rng':{'key_hex':snap['key_hex'],'requests':snap['requests']},'python_rng':hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
    'numpy_rng':hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
    'aegis':{k:serialize(ctx.get(k)) for k in ['p1','R1','Q1_diag','p2','R2','Q2_diag','z_fixed','flag_safety_control']},
    'observation':{k:array_hash(v) for k,v in obs.items() if isinstance(v,np.ndarray)},'collided':row['collided']}
   # RNG outside the simulator is logged; detector construction uses NumPy and is not part of action execution.
   equality_fields=['sim_state_sha256','qpos_sha256','qvel_sha256','ctrl_sha256','warmstart_sha256','marker_position','marker_quaternion','controller','action_queue','policy_rng','python_rng','numpy_rng','aegis','observation']
   payload['execution_fingerprint']=hashlib.sha256(json.dumps({k:payload[k] for k in equality_fields},sort_keys=True).encode()).hexdigest()
   atomic_json(directory/('checkpoint_%03d.json'%t),payload)
   np.savez(directory/('checkpoint_%03d.npz'%t),sim_state=env.get_sim_state(),qpos=env.sim.data.qpos,qvel=env.sim.data.qvel,ctrl=env.sim.data.ctrl)
   if branch is not None and t==branch:
    if row['collided'] or expected_checkpoint['collided']:raise InfrastructureError('Branch checkpoint already collided')
    if payload['execution_fingerprint']!=expected_checkpoint['execution_fingerprint']:
     atomic_json(directory/'REPLAY_MISMATCH.json',{'expected':expected_checkpoint,'actual':payload})
     raise InfrastructureError('Full execution checkpoint differs after replay')
    row['prefix_verified']=True;row['checkpoint_fingerprint']=payload['execution_fingerprint']
  def element(data,t):
   if t==0:
    key=(state['id'],seed);ih=digest(data)
    if key in self.first_inputs and self.first_inputs[key]!=ih:raise InfrastructureError('First policy observation differs across conditions')
    self.first_inputs[key]=ih;row['initial_policy_input_sha256']=ih
    np.savez(directory/'first_policy_input.npz',**{k:v for k,v in data.items() if not isinstance(v,str)})
   return data
  def make_client(host,port):
   from openpi_client.websocket_client_policy import WebsocketClientPolicy
   ws=WebsocketClientPolicy(host,port);clients.append(ws)
   class Client:
    def infer(_,data):
     result=ws.infer(data)
     if 'diagnostic' in result:
      info=result['diagnostic'];inferences.write(json.dumps(info)+'\n');inferences.flush()
      if info['request_index']==1:
       ch=np.asarray(result['actions']);row['first_chunk_sha256']=array_hash(ch);np.save(directory/'first_action_chunk.npy',ch)
       key=(state['id'],seed)
       if key in self.first_chunks and self.first_chunks[key]!=row['first_chunk_sha256']:raise InfrastructureError('Controlled first chunk differs')
       self.first_chunks[key]=row['first_chunk_sha256']
     return result
   return Client()
  def next_action(action,ctx):
   t=ctx['t']
   if branch is not None and t<branch:return action
   if condition=='reference':return ref.step(ctx['env'],ctx['obs'])
   if condition=='lift_then_nominal' and t<branch+10:
    out=np.asarray(action).copy();out[:6]=[0,0,1,0,0,0];return out
   return action
  def use_filter(enabled,t):
   if branch is not None and t<branch:return enabled
   if condition in ['nominal','reference','lift_then_nominal']:return False
   if condition=='release5' and t<branch+5:return False
   return enabled
  def candidate(original,output,t,status,ctx):
   raw,out=np.asarray(original),np.asarray(output)
   if not np.isfinite(out).all():raise InfrastructureError('Nonfinite action')
   if branch is not None and t<branch:
    if not np.array_equal(out,np.asarray(prefix[t]['output'])):raise InfrastructureError('Replayed prefix action differs')
   pending.clear();pending.update(step=t+1,original=raw.tolist(),output=out.tolist(),qp_status=status,delta_linf=float(np.max(abs(raw[:6]-out[:6]))),eef_pos=np.asarray(ctx['obs']['robot0_eef_pos']).tolist(),target_pos=np.asarray(ctx['obs'][state['target']+'_pos']).tolist(),reference_phase=ref.phase if condition=='reference' else None)
  def after(obs,done,t):
   disp=float(np.sum(abs(np.asarray(obs[row['active_obstacle']+'_pos'])-initial['obstacle'])))
   row['max_obstacle_l1_m']=max(row['max_obstacle_l1_m'],disp);row['end_step']=t+1;row['success']=bool(done)
   if disp>.001 and not row['collided']:row['collided']=True;row['collision_step']=t+1
   pending.update(obstacle_l1_m=disp,success=bool(done));trace.write(json.dumps(pending)+'\n');trace.flush()
  def caught(error,t):
   if isinstance(error,InfrastructureError):raise error
   if not (isinstance(error,NameError) and str(error)=="name 'a' is not defined"):
    raise InfrastructureError('Unexpected execution error: '+type(error).__name__+': '+str(error)) from error
   row['exited']=True;row['exit_reason']='upstream_qp_infeasible_undefined_a'
  def finish(done,collided,video):
   if bool(collided)!=row['collided'] or bool(done)!=row['success']:raise InfrastructureError('Scoring disagreement')
   import imageio
   row['video']=video;reader=imageio.get_reader(video);n=0
   for frame in reader:
    if n in [0,49,149,249] or n==row['end_step']-1:imageio.imwrite(directory/('frame_%03d.jpg'%n),frame)
    n+=1
   reader.close();row['video_frames']=n
  hooks={'_client':make_client,'_begin':begin,'_settled':settled,'_geometry':geometry,'_ready':ready,'_checkpoint':checkpoint,'_element':element,'_next':next_action,'_filter':use_filter,'_candidate':candidate,'_after':after,'_caught':caught,'_finish':finish,'_horizon':lambda h:h+extra}
  module=types.ModuleType('diagnostic_upstream');module.__file__=str(self.upstream/'main/main_aegis.py');sys.modules[module.__name__]=module;ns=vars(module);ns.update(hooks)
  status='failed'
  try:
   exec(compile(self.patched,module.__file__,'exec'),ns)
   orig=ns['_get_libero_env']
   def get_env(*args):
    random.seed(7);np.random.seed(7);return orig(*args)
   ns['_get_libero_env']=get_env
   args=ns['Args'](host='127.0.0.1',port=self.port,task_suite_name=state['suite'],safety_level=state['level'],task_index=[state['task']],episode_index=[state['episode']],video_out_path=str(directory/'videos'),seed=7,replan_steps=5,num_steps_wait=20)
   ns['eval_libero'](args)
   if branch is not None and not row.get('prefix_verified'):raise InfrastructureError('Branch checkpoint not reached')
   row['safe_success']=row['success'] and not row['collided'];status='complete'
  finally:
   trace.close();inferences.close()
   for env in envs:env.close()
   for client in clients:client._ws.close()
   os.chdir(previous);row['elapsed_seconds']=time.monotonic()-started
   atomic_json(directory/'manifest.json',dict(row,status=status))
   if status=='complete':atomic_json(directory/'row.json',row)
  return row
