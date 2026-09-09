#!/usr/bin/env python3
"""48 prefixes, <=384 branches. Freeze all gates using A before any B rollout."""
import argparse, gzip, hashlib, importlib.metadata, json, os, pickle, socket, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from crashbench.detour_benefit import HORIZONS, Opportunity, PendingAction, terminal, readout, validate_panel, validate_phase, freeze
from scripts.expansion.hash_tree_manifest import file_sha256, resolve_git_head

def write(path,data):
    with Path(path).open('x') as f:json.dump(data,f,indent=2,sort_keys=True);f.write('\n')
def dump(path,data):
    with Path(path).open('xb') as f:pickle.dump(data,f,protocol=5)
def load(path):
    with Path(path).open('rb') as f:return pickle.load(f)
def measured_act(env,policy,obs,instruction):
    import torch
    po=env.policy_observation(obs,policy.resize_size)
    torch.cuda.synchronize();start=time.perf_counter();action=np.asarray(policy.act(po,instruction)).copy();torch.cuda.synchronize()
    return action,po,time.perf_counter()-start

def step(env,obs,action,crash,glasses,index,calls,control,stream,policy_input=None,latency=0):
    from scripts.collect_glass_recovery_pairs import _glass_force
    before=env.flat_state().copy();new,_,done,_=env.step(action.tolist());acc=bool(crash and crash(env.sim_view))
    event={'step':index+1,'reason':terminal(done,acc,env.episode_terminated()),'calls':calls,'controller_steps':int(control),'inference_seconds':latency}
    pickle.dump({'event':event,'state_before':before,'state_after':env.flat_state().copy(),'action':action,'policy_input':policy_input,'force':_glass_force(env.sim_view,glasses) if glasses else 0.},stream,protocol=5)
    return new,event

def make_crash(env,glasses):
    from crashbench.predicates import build_any
    from scripts.collect_glass_recovery_pairs import _glass_predicate_specs,_prime_glass_predicates
    if not glasses:return None
    crash=build_any(_glass_predicate_specs(glasses));_prime_glass_predicates(crash,env.sim_view);return crash

def generate(env,policy,router,row,config,folder):
    from crashbench.branching.state import capture_exact_state
    from crashbench.glass_recovery_data import array_sha256
    from scripts.capture_glass_detector_placements import _condition_glasses,_seed_everything
    from scripts.collect_glass_recovery_pairs import CandidateRejected
    _seed_everything(config['seed']);env.seed(config['seed']);policy.reset()
    placement=SimpleNamespace(**row['placement']);glasses=_condition_glasses(placement,row['condition'])
    source_path=ROOT/row['placement_manifest'];source_path=source_path.parent/placement.source_state_path
    state=np.load(source_path,allow_pickle=False)
    if array_sha256(state)!=row['source']:raise ValueError('source state mismatch')
    obs=env.reset_to(state,movable_objects=glasses or None)
    raw_horizon=int(env._raw_env().horizon)
    if raw_horizon<config['settle_steps']+config['long_H']:raise ValueError('environment cannot support 2H')
    events=[];opportunity=Opportunity();start=time.monotonic();calls=0;risk_times=[]
    anchor={k:v for k,v in row.items() if k!='placement'};anchor.update(triggered=False,raw_environment_horizon=raw_horizon)
    with gzip.open(folder/'prefix.pkl.gz','xb',compresslevel=1) as stream:
        try:crash=make_crash(env,glasses)
        except CandidateRejected:
            events.append({'step':0,'reason':'accident','calls':0,'controller_steps':0});crash=None
        for warm in range(config['settle_steps'] if not events else 0):
            obs,_,done,_=env.step(env.dummy_action());acc=bool(crash and crash(env.sim_view));reason=terminal(done,acc,env.episode_terminated())
            pickle.dump({'warmup':warm,'obs':obs,'state':env.flat_state().copy(),'reason':reason},stream,protocol=5)
            if reason:events.append({'step':0,'reason':reason,'calls':0,'controller_steps':0});break
        if not events:
            for index in range(config['long_H']):
                action,po,latency=measured_act(env,policy,obs,placement.instruction);calls+=1
                risk=0.
                if index<=config['candidate_last_action']:
                    tick=time.perf_counter();pred=router.predict(policy.last_hidden,po['state'],action);risk=pred['base_catastrophe_probability'];risk_times.append(time.perf_counter()-tick)
                if opportunity.observe(risk,config['risk_threshold'],index,config['candidate_last_action']):
                    anchor.update(triggered=True,anchor_step=index,risk=float(risk),hidden=np.asarray(policy.last_hidden,float).tolist(),robot_state=np.asarray(po['state'],float).tolist(),nominal_action=action.tolist(),prefix_inference_calls=calls,proposal_inference_seconds=latency)
                    bundle=capture_exact_state(env,policy,identity={'source':row['source'],'episode':row['episode_id']},provenance={'commit':resolve_git_head(ROOT),'job':os.environ['SLURM_JOB_ID']},declared_branch_seed=config['seed'])
                    context={'bundle':bundle,'pending_action':action,'policy_input':po,'instruction':placement.instruction,'glasses':glasses,'prefix_events':events,'proposal_latency':latency,'risk_history_seconds':risk_times,'queue_executed':0,'queue_remaining':1,'post_proposal_queue':0}
                    dump(folder/'bundle.pkl',context);anchor.update(bundle_sha256=file_sha256(folder/'bundle.pkl'),bundle_id=bundle.bundle_id)
                    break
                obs,event=step(env,obs,action,crash,glasses,index,1,False,stream,po,latency);events.append(event)
                if event['reason']:break
    anchor['prefix_elapsed_seconds']=time.monotonic()-start;anchor['prefix_calls']=calls;anchor['risk_call_count']=len(risk_times);anchor['risk_elapsed_seconds']=sum(risk_times)
    if not anchor['triggered']:anchor['horizons']={str(h):readout(events,h) for h in HORIZONS};anchor['events']=events
    anchor['prefix_sha256']=file_sha256(folder/'prefix.pkl.gz');write(folder/'anchor.json',anchor)
    return anchor

def branch(env,policy,row,anchor,config,folder,repeat,option):
    from crashbench.branching.state import restore_exact_state
    from crashbench.recovery import DetourComplete
    from scripts.collect_glass_recovery_pairs import _controller_glass,TARGET,PLATE
    import torch
    if file_sha256(folder/'bundle.pkl')!=anchor['bundle_sha256']:raise ValueError('bundle changed')
    context=load(folder/'bundle.pkl');obs=restore_exact_state(context['bundle'],env,policy)
    override=config['policy_seed_override'][str(repeat)]
    if override is not None:torch.manual_seed(override);torch.cuda.manual_seed_all(override)
    crash=make_crash(env,context['glasses']);pending=PendingAction(context['pending_action']);controller=None
    if option:
        pending.accept_intervention();glass=row['placement']['on_path_glass'] if not context['glasses'] else context['glasses'][0]
        d=config['detour'];bowl=np.asarray(obs[TARGET+'_pos']);plate=np.asarray(obs[PLATE+'_pos'])
        controller=DetourComplete(_controller_glass(glass),bowl,plate,side=d['side'],lane_margin=d['lane_margin'],transit_z=float(bowl[2]+d['lift_offset']),descend_off=d['descend_offset'],leg_cap=d['leg_cap'],target_name=TARGET,orientation_target=None,path_aligned=True,grasp_xy_offset=d['grasp_xy_offset'],departure_clearance=d['departure_clearance']);controller.engage(obs)
    events=list(context['prefix_events']);start=time.monotonic();name=f'r{repeat}_o{option}'
    # Proposal at the candidate was paid for by both arms, including when R discards it.
    proposal_cost_due=True;resume_step=None
    with gzip.open(folder/(name+'.pkl.gz'),'xb',compresslevel=1) as stream:
        for index in range(anchor['anchor_step'],config['long_H']):
            control=controller is not None and controller.i<len(controller.legs)
            if controller is not None and not control and resume_step is None:resume_step=index
            calls=int(proposal_cost_due);latency=context['proposal_latency'] if proposal_cost_due else 0.;proposal_cost_due=False;po=None
            if control:action=np.asarray(controller.step(obs),dtype=np.float32)
            else:
                action=pending.take()
                if action is None:
                    action,po,elapsed=measured_act(env,policy,obs,context['instruction']);calls+=1;latency+=elapsed
                else:po=context['policy_input']
            obs,event=step(env,obs,action,crash,context['glasses'],index,calls,control,stream,po,latency);events.append(event)
            if event['reason']:break
    record={'episode_id':row['episode_id'],'source':row['source'],'condition':row['condition'],'role':row['role'],'repeat':repeat,'phase':'A' if repeat<2 else 'B','option':option,'policy_seed_override':override,'bundle_id':anchor['bundle_id'],'bundle_sha256':anchor['bundle_sha256'],'events':events,'horizons':{str(h):readout(events,h) for h in HORIZONS},'branch_elapsed_seconds':time.monotonic()-start,'controller_resume_base_step':resume_step,'trace_sha256':file_sha256(folder/(name+'.pkl.gz'))}
    write(folder/(name+'.json'),record);return record

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=ROOT/'configs/detour_benefit/development_v1.json');parser.add_argument('--output',type=Path,required=True);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    config=json.loads(args.config.read_text());validate_panel(config['panel']);out=args.output;out.mkdir(parents=True,exist_ok=args.resume)
    if args.resume:
        if json.loads((out/'config.json').read_text())!=config:raise ValueError('resume config changed')
        if (out/'complete.json').exists():raise ValueError('already complete')
    for rel,expected in config['input_sha256'].items():
        if file_sha256(ROOT/rel)!=expected:raise ValueError('input hash mismatch: '+rel)
    if not args.resume:write(out/'config.json',config)
    from crashbench.policies import OpenVLAPolicy
    from crashbench.envs import LiberoEnv
    from crashbench.counterfactual_router import FrozenOutcomeRouter
    checkpoint=Path(os.environ['CB_CHECKPOINT_PATH']);hashes=[]
    for p in sorted(checkpoint.iterdir()):
        if p.is_file():hashes.append({'file':p.name,'bytes':p.stat().st_size,'sha256':file_sha256(p)})
    if args.resume:
        if json.loads((out/'checkpoint_hashes.json').read_text())!=hashes:raise ValueError('checkpoint changed')
    else:write(out/'checkpoint_hashes.json',hashes)
    versions={}
    for name in ('numpy','torch','transformers','mujoco','robosuite','libero'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    write(out/(('resume_'+os.environ['SLURM_JOB_ID']+'.json') if args.resume else 'provenance.json'),{'commit':resolve_git_head(ROOT),'job_id':os.environ.get('SLURM_JOB_ID'),'host':socket.gethostname(),'versions':versions,'config_sha256':file_sha256(args.config),'max_prefixes':48,'max_branches':384,'new_confirmation':False})
    policy=OpenVLAPolicy(pretrained_checkpoint=config['checkpoint'],checkpoint_revision=config['checkpoint_revision'],capture_hidden=True)
    if not args.resume:write(out/'policy_identity.json',policy.checkpoint_identity)
    router=FrozenOutcomeRouter.load(ROOT/config['risk_path']);anchors=[];a=[];b=[]
    env=LiberoEnv('libero_spatial',0,seed=config['seed'])
    try:
        for i,row in enumerate(config['panel']):
            folder=out/row['episode_id']
            if (folder/'anchor.json').exists():anchor=json.loads((folder/'anchor.json').read_text())
            else:
                if folder.exists():raise ValueError('partial prefix exists; requires explicit recovery without new prefix')
                folder.mkdir();anchor=generate(env,policy,router,row,config,folder)
            anchors.append(anchor)
            if anchor['triggered']:
                for repeat in config['A_repeats']:
                    for option in ((0,1) if (i+repeat)%2==0 else (1,0)):
                        record=folder/f'r{repeat}_o{option}.json'
                        if not record.exists() and (folder/f'r{repeat}_o{option}.pkl.gz').exists():raise ValueError('partial branch requires explicit budget accounting')
                        a.append(json.loads(record.read_text()) if record.exists() else branch(env,policy,row,anchor,config,folder,repeat,option))
            print('A',row['episode_id'],'triggered',anchor['triggered'],'branches',len(a),flush=True)
        if not (out/'A.json').exists():write(out/'anchors.json',anchors);write(out/'A.json',a)
        elif json.loads((out/'A.json').read_text())!=a:raise ValueError('saved A changed')
        frozen=freeze(anchors,a,config);models=frozen.pop('models')
        if not (out/'models.pkl').exists():dump(out/'models.pkl',models)
        frozen.update(A_sha256=file_sha256(out/'A.json'),anchors_sha256=file_sha256(out/'anchors.json'),models_sha256=file_sha256(out/'models.pkl'))
        if not (out/'freeze.json').exists():
            write(out/'freeze.json',frozen);(out/'freeze.sha256').write_text(file_sha256(out/'freeze.json')+'\n')
        elif json.loads((out/'freeze.json').read_text())['choices']!=frozen['choices']:raise ValueError('frozen decisions changed')
        for i,(row,anchor) in enumerate(zip(config['panel'],anchors)):
            if anchor['triggered']:
                for repeat in config['B_repeats']:
                    for option in ((0,1) if (i+repeat)%2==0 else (1,0)):
                        folder=out/row['episode_id'];record=folder/f'r{repeat}_o{option}.json'
                        if not record.exists() and (folder/f'r{repeat}_o{option}.pkl.gz').exists():raise ValueError('partial branch requires explicit budget accounting')
                        b.append(json.loads(record.read_text()) if record.exists() else branch(env,policy,row,anchor,config,folder,repeat,option))
            print('B',row['episode_id'],'branches',len(b),flush=True)
        validate_phase(b,anchors,'B');write(out/'B.json',b)
        if len(a)+len(b)>config['max_branches']:raise ValueError('budget exceeded')
        write(out/'complete.json',{'status':'COMPLETE','prefixes':len(anchors),'scored_branches':len(a)+len(b),'freeze_sha256':file_sha256(out/'freeze.json')})
    finally:env.env.close()
if __name__=='__main__':main()
