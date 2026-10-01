"""Pinned-loop diagnostics with read-only checkpoints and replayed prefixes."""
import hashlib,io,json,os,pickle,random,sys,time,types
from pathlib import Path
import numpy as np
from adapter import adapt
from geometry import active_obstacle,verified_ellipsoid
from protocol import independent_caption,seed_for,CHECKPOINTS,BRANCH_BASELINE
from reference import Reference
from serve import digest
from analysis import execution_digest
from observation_control import RGBControl,render_model_signature,scene_signature
BASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE/'scripts'))
from api_budget import atomic_json
from run_safelibero import CFG,link_offline_bert
class InfrastructureError(RuntimeError):pass

def owned_observation(obs):
 return {key:(value.copy(order='C') if isinstance(value,np.ndarray) else value) for key,value in obs.items()}

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
  config=self.root/'libero_config';config.mkdir(exist_ok=True)
  benchmark=self.upstream/'safelibero/libero/libero'
  (config/'config.yaml').write_text(json.dumps({'benchmark_root':str(benchmark),'bddl_files':str(benchmark/'bddl_files'),'init_states':str(benchmark/'init_files'),'assets':str(benchmark/'assets'),'datasets':str(self.upstream/'safelibero/libero/datasets')}))
  os.environ['LIBERO_CONFIG_PATH']=str(config)
  sys.path[:0]=[str(self.upstream/'main'),str(self.upstream/'safelibero'),str(self.upstream/'openpi/packages/openpi-client/src')]
  self.source=(self.upstream/'main/main_aegis.py').read_text();self.patched=adapt(self.source)
  self.asset_manifest_hash=hashlib.sha256((self.assets/'VERIFIED.json').read_bytes()).hexdigest()
  self.rgb_control=RGBControl(self.root/'controlled_rgb')
  self.first_inputs={};self.first_chunks={};self.first_execution={}
  if (self.root/'rows.json').exists():
   for r in json.loads((self.root/'rows.json').read_text()):
    key=(r['state'],r['seed'])
    for memo,field in [(self.first_inputs,'initial_policy_input_sha256'),(self.first_chunks,'first_chunk_sha256'),(self.first_execution,'initial_execution_fingerprint')]:
     if field in r:
      if key in memo and memo[key]!=r[field]:raise InfrastructureError('Previously recorded pairing mismatch')
      memo[key]=r[field]

 def run(self,state,repeat,condition,name=None,variant='center',branch=None,extra=0,validation=False):
  started=time.monotonic();seed=seed_for(state,repeat,validation)
  name=name or '%s_r%02d_%s'%(state['id'],repeat,condition)
  directory=self.root/'runs'/name;directory.mkdir(parents=True)
  row=dict(run_id=name,state=state['id'],role=state['role'],repeat=repeat,seed=seed,condition=condition,variant=variant,branch_step=branch,
   extra_budget=extra,validation=validation,success=False,collided=False,max_obstacle_l1_m=0.,collision_step=None,end_step=0,exited=False,
   code_commit=os.environ['CB_CODE_COMMIT'],upstream_commit=CFG['upstream_commit'],slurm_job=os.environ['SLURM_JOB_ID'],caption=state['caption'])
  configuration={'state':state,'upstream':CFG,'environment_seed':7,'settling_actions':20,'chunk_actions':5,
   'horizon_actions':300+extra,'policy_seed':seed,'policy':'pi05_libero','condition':condition,'reference_variant':variant,
   'branch_step':branch,'extra_budget':extra,'validation':validation,'asset_manifest_sha256':self.asset_manifest_hash,
   'perception':'frozen prior repeat0 caption; state/condition-frozen initial geometry','api_calls':0,
   'observation_control':'exact physical/render-scene RGB templates; native RGB variants retained',
   'controlled_inference':'native infer with exact-input/exact-RNG common output replay; native differences retained'}
  atomic_json(directory/'manifest.json',dict(row,status='started',configuration=configuration))
  envs=[];clients=[];initial={};pending={};frames=[];trace=(directory/'steps.jsonl').open('w');inferences=(directory/'policy.jsonl').open('w')
  previous=Path.cwd();os.chdir(directory)
  (directory/'GroundingDINO').symlink_to(self.assets/'GroundingDINO',target_is_directory=True);link_offline_bert(directory,self.assets)
  ref=Reference(state,variant);prefix_row=None
  baseline_name='%s_r%02d_%s'%(state['id'],repeat,BRANCH_BASELINE)
  row['prefix_condition']=BRANCH_BASELINE if branch is not None else None
  if branch is not None:
   prefix_dir=self.root/'runs'/baseline_name
   prefix=[json.loads(l) for l in (prefix_dir/'steps.jsonl').read_text().splitlines()]
   expected_checkpoint=json.loads((prefix_dir/('checkpoint_%03d.json'%branch)).read_text())
  def begin(client):
   ack=client.infer({'__paired_reset_rng__':seed,'run_id':name})
   if ack.get('seed')!=seed or ack.get('run_id')!=name or ack.get('requests')!=0:
    raise InfrastructureError('RNG reset acknowledgement mismatch')
   row['policy_requests']=0;row['rng_reset']=ack;atomic_json(directory/'rng_reset.json',ack)
  def settled(env,obs,ctx):
   envs.append(env);initial['env']=env;initial['eef']=np.asarray(obs['robot0_eef_pos']).copy()
   qpos=np.asarray(env.sim.data.qpos,dtype='<f8').copy();np.save(directory/'settled_qpos.npy',qpos)
   row['qpos_sha256']=array_hash(qpos);row['settled_qvel_sha256']=array_hash(env.sim.data.qvel)
   initial['obs']=obs
   before={k:array_hash(v) for k,v in obs.items() if isinstance(v,np.ndarray)}
   control=self.rgb_control.apply(obs,[k for k in obs if k.endswith('_image')],scene_signature(env,obs),'full',directory/'native_initial_rgb.npz')
   row['initial_native_observation_hashes']=before;row['initial_rgb_control']=control
   initial['settled_observation_hashes']={k:array_hash(v) for k,v in obs.items() if isinstance(v,np.ndarray)}
   np.savez_compressed(directory/'settled_observation.npz',**{k:v for k,v in obs.items() if isinstance(v,np.ndarray)})
   robot=env.robots[0];control=robot.controller
   row['action_spec']=[np.asarray(x).tolist() for x in env.env.action_spec]
   atomic_json(directory/'controller_config.json',serialize({k:getattr(control,k,None) for k in ['input_min','input_max','output_min','output_max','control_delta','kp','damping','position_limits','orientation_limits']}))
  def geometry(env,obs,description,out):
   obstacle=active_obstacle(env,obs);row['active_obstacle']=obstacle
   identity=condition in ['identity','identity_geometry'];geometry_condition=condition in ['geometry','identity_geometry']
   if branch is not None:identity=geometry_condition=True
   caption=independent_caption(obstacle) if identity else state['caption'];row['used_caption']=caption
   row['identity_changed']=caption!=state['caption']
   if condition in ['nominal','reference'] and branch is None:
    random.seed(7);np.random.seed(7)
    return np.zeros(3),np.eye(3),np.ones(3),False
   cache=self.root/'perception'/state['id'];cache.mkdir(parents=True,exist_ok=True)
   mode=('sim_'+('correct' if identity else 'raw')) if geometry_condition else ('detector_'+hashlib.sha256(caption.encode()).hexdigest()[:8])
   file=cache/(mode+'.npz')
   img=np.ascontiguousarray(obs['agentview_image'][::-1,::-1]).copy();image_hash=array_hash(img)
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
     a=get_point_cloud(img,np.ascontiguousarray(obs['agentview_depth'][::-1,::-1]).copy(),env,'agentview',caption,self.detector,out)
     b=get_point_cloud(np.ascontiguousarray(obs['backview_image'][::-1,::-1]).copy(),np.ascontiguousarray(obs['backview_depth'][::-1,::-1]).copy(),env,'backview',caption,self.detector,out)
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
   if t:
    control=self.rgb_control.apply(obs,[k for k in obs if k.endswith('_image')],scene_signature(env,obs),'full',directory/('native_checkpoint_rgb_%03d.npz'%t))
    atomic_json(directory/('rgb_control_%03d.json'%t),control)
   snap=clients[0].infer({'__diagnostic_snapshot__':True})
   robot=env.robots[0]
   control=serialize({k:v for k,v in vars(robot.controller).items() if isinstance(v,(np.ndarray,float,int,bool,str,list,tuple,type(None)))})
   payload={'step':t,'sim_state_sha256':array_hash(env.get_sim_state()),'qpos_sha256':array_hash(env.sim.data.qpos),'qvel_sha256':array_hash(env.sim.data.qvel),
    'ctrl_sha256':array_hash(env.sim.data.ctrl),'warmstart_sha256':array_hash(env.sim.data.qacc_warmstart),
    'applied_force_sha256':array_hash(env.sim.data.qfrc_applied),'external_force_sha256':array_hash(env.sim.data.xfrc_applied),
    'mocap_pos_sha256':array_hash(env.sim.data.mocap_pos),'mocap_quat_sha256':array_hash(env.sim.data.mocap_quat),'act_sha256':array_hash(env.sim.data.act),
    'marker_position':serialize(env.sim.model.body_pos[ctx['eef_body_id']]),'marker_quaternion':serialize(env.sim.model.body_quat[ctx['eef_body_id']]),'controller':control,'action_queue':[np.asarray(x).tolist() for x in ctx['action_plan']],
    'policy_rng':{'key_hex':snap['key_hex'],'requests':snap['requests']},'python_rng':hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
    'numpy_rng':hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
    'aegis':{k:serialize(ctx.get(k)) for k in ['p1','R1','Q1_diag','p2','R2','Q2_diag','z_fixed','flag_safety_control']},
    'observation':{k:array_hash(v) for k,v in obs.items() if isinstance(v,np.ndarray)},'collided':row['collided']}
   # Python/NumPy states are reset identically after frozen perception and compared here.
   payload['execution_fingerprint']=execution_digest(payload,include_aegis=True)
   if t==0:
    row['observation_changed_during_perception']=[k for k,value in initial['settled_observation_hashes'].items() if payload['observation'].get(k)!=value]
    if row['observation_changed_during_perception']:
     np.savez_compressed(directory/'post_perception_observation.npz',**{k:v for k,v in obs.items() if isinstance(v,np.ndarray)})
     raise InfrastructureError('Observation mutated during perception: '+','.join(row['observation_changed_during_perception']))
    if payload['action_queue'] or payload['policy_rng']['requests']!=0:
     raise InfrastructureError('Initial action queue or policy stream is not empty')
    key=(state['id'],seed);fingerprint=execution_digest(payload)
    row['initial_execution_fingerprint']=fingerprint
    if key in self.first_execution and self.first_execution[key]!=fingerprint:
     atomic_json(directory/'INITIAL_EXECUTION_MISMATCH.json',payload)
     raise InfrastructureError('Initial simulator/controller/RNG/queue/observation mismatch')
    self.first_execution[key]=fingerprint
   atomic_json(directory/('checkpoint_%03d.json'%t),payload)
   np.savez(directory/('checkpoint_%03d.npz'%t),sim_state=env.get_sim_state(),qpos=env.sim.data.qpos,qvel=env.sim.data.qvel,ctrl=env.sim.data.ctrl)
   if branch is not None and t==branch:
    if row['collided'] or expected_checkpoint['collided']:raise InfrastructureError('Branch checkpoint already collided')
    if payload['execution_fingerprint']!=expected_checkpoint['execution_fingerprint']:
     atomic_json(directory/'REPLAY_MISMATCH.json',{'expected':expected_checkpoint,'actual':payload})
     raise InfrastructureError('Full execution checkpoint differs after replay')
    row['prefix_verified']=True;row['checkpoint_fingerprint']=payload['execution_fingerprint']
  def element(data,t):
   signature=scene_signature(initial['env'],initial['latest_obs'])
   control=self.rgb_control.apply(data,['observation/image','observation/wrist_image'],signature,'policy',directory/('native_policy_rgb_%03d.npz'%t))
   atomic_json(directory/('policy_rgb_control_%03d.json'%t),dict(control,step=t))
   if t==0:
    key=(state['id'],seed);ih=digest(data)
    row['initial_policy_input_sha256']=ih
    np.savez(directory/'first_policy_input.npz',**{k:v for k,v in data.items() if not isinstance(v,str)})
    if key in self.first_inputs and self.first_inputs[key]!=ih:raise InfrastructureError('First policy observation differs across conditions')
    self.first_inputs[key]=ih
   return data
  def make_client(host,port):
   from openpi_client.websocket_client_policy import WebsocketClientPolicy
   ws=WebsocketClientPolicy(host,port);clients.append(ws)
   class Client:
    def infer(_,data):
     result=ws.infer(data)
     if 'diagnostic' in result:
      info=result['diagnostic'];row['policy_requests']+=1
      if info['run_id']!=name or info['request_index']!=row['policy_requests'] or info['input_sha256']!=digest(data):
       raise InfrastructureError('Policy request identity/input/sequence mismatch')
      inferences.write(json.dumps(info)+'\n');inferences.flush()
      if info['request_index']==1:
       row['native_first_chunk_sha256']=info['native_chunk_sha256']
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
   if condition=='reference' and (branch is None or t>=branch):
    lower,upper=map(np.asarray,row['action_spec'])
    if (out<lower).any() or (out>upper).any():raise InfrastructureError('Reference action exceeds native robot capability')
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
    if frame.shape!=(1024,1024,3):raise InfrastructureError('Unexpected video frame shape')
    if n in [0,49,149,249] or n==row['end_step']-1:imageio.imwrite(directory/('frame_%03d.jpg'%n),frame)
    n+=1
   reader.close();row['video_frames']=n
   if n!=row['end_step']+int(row['exited']):raise InfrastructureError('Video/action count mismatch')
   if condition=='reference':
    executed=[json.loads(l)['output'] for l in (directory/'steps.jsonl').read_text().splitlines()]
    np.save(directory/'reference_actions.npy',np.asarray(executed),allow_pickle=False)
    row['reference_action_tape']=str(directory/'reference_actions.npy')
  hooks={'_client':make_client,'_begin':begin,'_settled':settled,'_geometry':geometry,'_ready':ready,'_checkpoint':checkpoint,'_element':element,'_next':next_action,'_filter':use_filter,'_candidate':candidate,'_after':after,'_caught':caught,'_finish':finish,'_horizon':lambda h:h+extra}
  module=types.ModuleType('diagnostic_upstream');module.__file__=str(self.upstream/'main/main_aegis.py');sys.modules[module.__name__]=module;ns=vars(module);ns.update(hooks)
  status='failed'
  try:
   exec(compile(self.patched,module.__file__,'exec'),ns)
   orig=ns['_get_libero_env']
   def get_env(*args):
    random.seed(7);np.random.seed(7);env,description=orig(*args)
    # Own every returned array before later GPU/library work can touch camera buffers.
    original_step=env.step;original_init=env.set_init_state
    def step(action):
     signature=render_model_signature(env.sim.model)
     observation,reward,done,info=original_step(action)
     observation=owned_observation(observation);observation['_render_model_signature']=signature
     initial['latest_obs']=observation
     return observation,reward,done,info
    def initialize(value):
     signature=render_model_signature(env.sim.model)
     observation=owned_observation(original_init(value));observation['_render_model_signature']=signature
     initial['latest_obs']=observation
     return observation
    env.step=step;env.set_init_state=initialize
    return env,description
   ns['_get_libero_env']=get_env
   args=ns['Args'](host='127.0.0.1',port=self.port,task_suite_name=state['suite'],safety_level=state['level'],task_index=[state['task']],episode_index=[state['episode']],video_out_path=str(directory/'videos'),seed=7,replan_steps=5,num_steps_wait=20)
   ns['eval_libero'](args)
   if branch is not None and not row.get('prefix_verified'):raise InfrastructureError('Branch checkpoint not reached')
   row['safe_success']=row['success'] and not row['collided'];status='complete'
  except BaseException as error:
   row['infrastructure_error']=type(error).__name__+': '+str(error);raise
  finally:
   trace.close();inferences.close()
   for env in envs:env.close()
   for client in clients:client._ws.close()
   os.chdir(previous);row['elapsed_seconds']=time.monotonic()-started
   atomic_json(directory/'manifest.json',dict(row,status=status,configuration=configuration))
   if status=='complete':atomic_json(directory/'row.json',row)
  return row
