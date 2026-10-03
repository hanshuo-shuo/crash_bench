"""FR-1C: fixed render-only intervention, with stepping prohibited."""
import argparse,hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import diagnostic as d
from visibility import body_geoms
r=d.r
P=json.loads((Path(__file__).parent/'opacity_protocol.json').read_text())
KEYS=('prefix_final','image_embedding','image_prefix_final','valid_tokens','valid_image_tokens','proposal_actions','action_seed7','proprio','risk_scores')
STEP_ATTEMPTS=[]

def blocked_step(*args,**kwargs):
    STEP_ATTEMPTS.append('step');raise RuntimeError('FR-1C prohibits every environment/integration step')

def full_state_hash(env):
    return hashlib.sha256(json.dumps(r.clean(r.capture(env)),sort_keys=True).encode()).hexdigest()

def model_fingerprint(env):
    native=env.sim.model._model;arrays={};numbers={}
    for prefix,obj in [('',native),('opt.',native.opt),('stat.',native.stat)]:
        for key in dir(obj):
            if key.startswith('_'):continue
            value=getattr(obj,key)
            if isinstance(value,np.ndarray):
                a=np.ascontiguousarray(value);arrays[prefix+key]=dict(shape=a.shape,dtype=a.dtype.str,sha256=hashlib.sha256(a.tobytes()).hexdigest())
            elif isinstance(value,(int,float,bool,np.number)):numbers[prefix+key]=r.clean(value)
    if len(arrays)<100:raise RuntimeError('Incomplete model-array coverage')
    return r.clean(dict(arrays=arrays,numbers=numbers))

def source_paths(episode,variant):
    decoy=variant.startswith('decoy_');cpu=Path(P['decoy_cpu_parent'] if decoy else P['cpu_parent'])/('e%d'%episode)
    gpu=Path(P['decoy_gpu_parent'] if decoy else P['gpu_parent'])/('e%d'%episode)/variant
    return cpu,cpu/variant,gpu

def make(root,episode,variant):
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    parent,label_source,old_gpu=source_paths(episode,variant);geo=json.loads((parent/'fixture.json').read_text());d.setup_layout(episode,geo,variant)
    folder=root/('e%d'%episode)/variant;folder.mkdir(parents=True)
    r.SEALED=variant in ('sealed','decoy_sealed')
    suite=benchmark.get_benchmark_dict()[r.STATE['suite']](safety_level=r.STATE['level']);task=suite.get_task(r.STATE['task'])
    bddl=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    random.seed(7);np.random.seed(7)
    env=ControlEnv(bddl_file_name=bddl,camera_heights=1024,camera_widths=1024,use_camera_obs=not r.HEADLESS,has_offscreen_renderer=not r.HEADLESS,camera_names=['agentview','robot0_eye_in_hand'],camera_depths=False)
    env.seed(7);env.reset()
    official=suite.get_task_init_states(r.STATE['task'])[episode]
    assert np.array_equal(official,np.load(label_source/'official_init.npy'))
    env.set_init_state(official)  # forward / observation refresh only; step is blocked
    np.save(folder/'official_init.npy',official);(folder/'task.bddl').write_bytes(bddl.read_bytes())
    canonical=json.loads((label_source/'initial_restore.json').read_text())
    ctl=env.robots[0].controller
    for k,v in canonical['controller'].items():
        if not hasattr(ctl,k):setattr(ctl,k,np.asarray(v) if isinstance(v,list) else v)
    obs=r.restore(env,canonical)
    assert r.statehash(r.capture(env))==r.statehash(canonical)
    goal=[[str(v).lower() if i==0 else v for i,v in enumerate(x)] for x in env.env.parsed_problem['goal_state']]
    assert goal==[['in',r.STATE['target'],r.STATE['goal_site']]]
    red=sorted(env.sim.model.geom_name2id(x['name']) for x in geo['variants'][variant])
    for box in geo['variants'][variant]:
        i=env.sim.model.geom_name2id(box['name']);lo=np.asarray(box['lower']);hi=np.asarray(box['upper'])
        assert np.max(np.abs(env.sim.data.geom_xpos[i]-(lo+hi)/2))<1e-12
        assert np.max(np.abs(env.sim.model.geom_size[i]-(hi-lo)/2))<1e-12
        assert int(env.sim.model.geom_bodyid[i])==0 and env.sim.model.geom_contype[i] and env.sim.model.geom_conaffinity[i]
        assert np.array_equal(env.sim.model.geom_rgba[i],np.asarray([.85,.025,.035,1],dtype=env.sim.model.geom_rgba.dtype))
    target=body_geoms(env.sim.model,env.env.obj_body_id[r.STATE['target']])
    label=json.loads((label_source/'summary.json').read_text())['label'];assert label in ('feasible','infeasible')
    r.write(folder/'SOURCE_LABEL.json',dict(label=label,label_root=str(label_source),source_files={f:r.sha(label_source/f) for f in ('summary.json','GATE.json','geometry.json','initial_restore.json')},fixture_sha256=r.sha(parent/'fixture.json'),opaque_source=str(old_gpu),scope='Inherited independent RE-1 witness/certificate; no new physical experiment.'))
    r.write(folder/'INITIAL_RESTORE.json',r.capture(env));np.savez_compressed(folder/'RENDER_MODEL_BEFORE.npz',geom_rgba=env.sim.model.geom_rgba.copy(),geom_matid=env.sim.model.geom_matid.copy(),red=np.asarray(red),target=np.asarray(target))
    return folder,env,obs,red,target,old_gpu

def alpha_checks(env,red,alpha,before):
    env.sim.model.geom_rgba[red,3]=alpha;after=model_fingerprint(env)
    changed=[k for k in before['arrays'] if before['arrays'][k]!=after['arrays'][k]]
    expected=[] if alpha==1. else ['geom_rgba']
    assert changed==expected and before['numbers']==after['numbers'],changed
    return dict(alpha=alpha,changed_arrays=changed,model_fingerprint=after,full_state_hash=full_state_hash(env))

def cpu(root):
    rows=[]
    for episode,variant in P['states']:
        folder,env,obs,red,target,old=make(root,episode,variant)
        try:
            state=full_state_hash(env);base=model_fingerprint(env);r.write(folder/'MODEL_BEFORE.json',base);checks=[]
            for name,alpha in P['conditions_in_order']:
                row=alpha_checks(env,red,alpha,base);assert row['full_state_hash']==state
                row['condition']=name;checks.append(row)
            env.sim.model.geom_rgba[red,3]=1.;assert model_fingerprint(env)==base and full_state_hash(env)==state
            r.write(folder/'CPU_PHYSICS_CHECK.json',dict(passed=True,checks=checks,zero_step_attempts=len(STEP_ATTEMPTS),label_inherited=True));rows.append(dict(episode=episode,variant=variant,passed=True))
        finally:env.close()
    r.write(root/'CPU_GATE.json',dict(passed=len(rows)==8 and not STEP_ATTEMPTS,rows=rows,environment_actions=0,step_attempts=STEP_ATTEMPTS))
    return rows

def render_data(env):
    env.env._update_observables(force=True);obs=env.env._get_observations();return r.policy_input(obs,d.P['policy_prompt'])

def render_rgb(env,camera):
    from openpi_client import image_tools
    native=np.ascontiguousarray(np.asarray(env.sim.render(width=1024,height=1024,camera_name=camera))[::-1,::-1])
    return native,image_tools.convert_to_uint8(image_tools.resize_with_pad(native,224,224))

def target_roi(env,target,camera):
    m=env.sim.model;sd=env.sim.data;cam=m.camera_name2id(camera);R=np.asarray(sd.cam_xmat[cam]).reshape(3,3);pos=sd.cam_xpos[cam]
    points=np.concatenate([r.geom_points(m,sd,g) for g in target]);q=(points-pos)@R;q=q[-q[:,2]>0]
    mask=np.zeros((224,224),bool)
    if not len(q):return mask,None
    f=112/np.tan(np.deg2rad(float(m.cam_fovy[cam]))/2);uv=np.c_[223-(112+f*q[:,0]/(-q[:,2])),223-(112+f*q[:,1]/(-q[:,2]))]
    lo=np.floor(uv.min(0)).astype(int)-2;hi=np.ceil(uv.max(0)).astype(int)+2
    lo=np.maximum(lo,0);hi=np.minimum(hi,223)
    if (lo<=hi).all():mask[lo[1]:hi[1]+1,lo[0]:hi[0]+1]=True
    return mask,[int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])]

def influence(env,target,folder,data):
    from PIL import Image
    before=full_state_hash(env);fp=model_fingerprint(env);m=env.sim.model;rgba=m.geom_rgba.copy();matid=m.geom_matid.copy();views=[];baselines=[]
    for camera,key in [('agentview','observation/image'),('robot0_eye_in_hand','observation/wrist_image')]:
        native,normal=render_rgb(env,camera);assert np.array_equal(normal,data[key])
        Image.fromarray(native).save(folder/(camera+'_normal_1024.png'));Image.fromarray(normal).save(folder/(camera+'_policy_224.png'))
        colors=[]
        for name,color in [('cyan',[0.,1.,1.]),('magenta',[1.,0.,1.])]:
            m.geom_matid[target]=-1;m.geom_rgba[target,:3]=color
            raw,small=render_rgb(env,camera);colors.append(small)
            Image.fromarray(raw).save(folder/(camera+'_'+name+'_1024.png'));Image.fromarray(small).save(folder/(camera+'_'+name+'_224.png'))
        m.geom_rgba[:]=rgba;m.geom_matid[:]=matid
        _,repeat=render_rgb(env,camera);assert np.array_equal(repeat,normal)
        roi,bbox=target_roi(env,target,camera);diff=np.max(np.abs(colors[0].astype(int)-colors[1].astype(int)),axis=2).astype(np.uint8);changed=(diff>=1)&roi
        Image.fromarray(diff).save(folder/(camera+'_target_influence_224.png'));Image.fromarray(roi.astype(np.uint8)*255).save(folder/(camera+'_target_roi_224.png'))
        Image.fromarray(np.where(changed[:,:,None],np.array([0,255,255],dtype=np.uint8),normal)).save(folder/(camera+'_influence_overlay_224.png'))
        views.append(dict(camera=camera,target_influence_pixels=int(changed.sum()),all_changed_pixels=int((diff>=1).sum()),max_channel_difference=int(diff.max()),roi_bbox=bbox,exact_policy_preprocessing=True,normal_repeat_exact=True))
        rgb=normal.astype(float)/255.;red=(rgb[:,:,0]>.4)&(rgb[:,:,0]>1.6*rgb[:,:,1])&(rgb[:,:,0]>1.6*rgb[:,:,2]);yy,xx=np.nonzero(red)
        baselines.append(np.r_[rgb.reshape(8,28,8,28,3).mean((1,3)).ravel(),red.mean(),xx.mean()/224 if len(xx) else 0.,yy.mean()/224 if len(yy) else 0.])
    assert full_state_hash(env)==before and model_fingerprint(env)==fp
    np.savez_compressed(folder/'cue_baselines.npz',rgb_pooled_and_red=np.concatenate(baselines))
    result=dict(views=views,target_influences_any_policy_view=any(x['target_influence_pixels']>0 for x in views),full_state_unchanged=True,model_restored_exact=True,marker_images_sent_to_policy=False)
    r.write(folder/'TARGET_INFLUENCE.json',result);return result

def gpu(root,port):
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    client=WebsocketClientPolicy('127.0.0.1',port);rows=[];anchor=None;anchor_input=None;extractions=0
    def extract(folder,data):
        nonlocal extractions
        client.infer({'__paired_reset_rng__':7,'run_id':str(folder.relative_to(root))});f=client.infer(dict(data,__extract_initial_features__=True));extractions+=1
        np.savez_compressed(folder/'initial_policy_input.npz',**{k:v for k,v in data.items() if k!='prompt'});np.savez_compressed(folder/'initial_features.npz',**{k:np.asarray(f[k]) for k in KEYS});r.write(folder/'FEATURE_AUDIT.json',f['metadata'])
        assert f['metadata']['passed'] and f['metadata']['feature_repeat_linf']==f['metadata']['action_before_after_linf']==0
        assert f['metadata']['input_sha256']==r.digest(data)
        return f
    for episode,variant in P['states']:
        folder,env,obs,red,target,old=make(root,episode,variant)
        try:
            state=full_state_hash(env);base=model_fingerprint(env);r.write(folder/'MODEL_BEFORE.json',base)
            for name,alpha in P['conditions_in_order']:
                sub=folder/name;sub.mkdir();checks=alpha_checks(env,red,alpha,base);data=render_data(env);assert full_state_hash(env)==state
                if name=='opaque':
                    with np.load(old/'initial_policy_input.npz') as z:assert all(np.array_equal(z[k],data[k]) for k in z.files)
                    assert r.digest(data)==json.loads((old/'FEATURE_AUDIT.json').read_text())['input_sha256']
                audit=influence(env,target,sub,data);f=extract(sub,data)
                if anchor is None:anchor=f;anchor_input=data
                assert full_state_hash(env)==state
                checks.update(passed=True,condition=name,environment_actions=0);r.write(sub/'PHYSICS_CHECK.json',checks)
                rows.append(dict(episode=episode,variant=variant,condition=name,alpha=alpha,target_influences_any_policy_view=audit['target_influences_any_policy_view'],views=audit['views'],input_sha256=r.digest(data)))
            env.sim.model.geom_rgba[red,3]=1.;assert model_fingerprint(env)==base and full_state_hash(env)==state
        finally:env.close()
    folder=root/'end_anchor';folder.mkdir();repeat=extract(folder,anchor_input);differences={k:float(np.max(np.abs(np.asarray(anchor[k])-np.asarray(repeat[k])))) for k in KEYS}
    r.write(root/'END_ANCHOR_REPEAT.json',dict(passed=all(x==0 for x in differences.values()),max_absolute=differences))
    assert not any(differences.values()) and extractions==17 and not STEP_ATTEMPTS
    r.write(root/'VISIBILITY_GATE.json',dict(passed=all(x['target_influences_any_policy_view'] for x in rows if x['condition']=='transparent'),rows=rows,scope=P['scope']))
    r.write(root/'COUNTS.json',dict(inputs=17,feature_forward_passes=34,unexecuted_action_inferences=102,environment_actions=0,step_attempts=STEP_ATTEMPTS,api_calls=0))
    return rows

def main(root,stage,port):
    r.configure(root)
    from robosuite.utils.binding_utils import MjSim
    from robosuite.environments.base import MujocoEnv
    from libero.libero.envs.env_wrapper import ControlEnv
    MjSim.step=blocked_step;MujocoEnv.step=blocked_step;ControlEnv.step=blocked_step
    r.write(root/'protocol.json',P);rows=cpu(root) if stage=='cpu' else gpu(root,port)
    r.write(root/'COMPLETE.json',dict(stage=stage,rows=rows,step_attempts=STEP_ATTEMPTS,environment_actions=0,source_commit=os.environ['CB_CODE_COMMIT'],job_id=os.environ['SLURM_JOB_ID'],api_calls=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('stage',choices=['cpu','gpu']);p.add_argument('--port',type=int);a=p.parse_args()
    try:main(a.root,a.stage,a.port)
    except BaseException as e:r.write(a.root/'STOP.json',dict(error=type(e).__name__,reason=str(e),step_attempts=STEP_ATTEMPTS));raise
