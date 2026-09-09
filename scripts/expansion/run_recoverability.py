"""Four-parent bounded physical-state probe; no fitting and no new sources."""
import argparse,copy,gzip,json,os,pickle,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.expansion.run_detour_benefit import write,dump,load,measured_act,step,make_crash,branch
from scripts.expansion.hash_tree_manifest import file_sha256,resolve_git_head
from crashbench.detour_benefit import readout

def suffix(events,anchor,h):return readout([dict(e,step=e['step']-anchor) for e in events if e['step']>anchor],h)

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/recoverability/probe_v1.json');p.add_argument('--output',type=Path,required=True);args=p.parse_args();config=json.loads(args.config.read_text());out=args.output;out.mkdir(parents=True,exist_ok=False)
 for rel,h in config['input_sha256'].items():
  if file_sha256(ROOT/rel)!=h:raise ValueError('parent input changed')
 parent=ROOT/config['parent_run'];old=json.loads((parent/'config.json').read_text());oldanchors={e['episode_id']:e for e in json.loads((parent/'anchors.json').read_text())};rows={e['episode_id']:e for e in old['panel']}
 write(out/'config.json',config);write(out/'provenance.json',{'commit':resolve_git_head(ROOT),'job_id':os.environ['SLURM_JOB_ID'],'new_physical_sources':0,'max_scored_branches':48,'max_prefixes':4,'config_sha256':file_sha256(args.config)})
 # Verify weights again on the compute node against the parent's measured bytes.
 checkpoint=Path(os.environ['CB_CHECKPOINT_PATH'])
 for r in json.loads((parent/'checkpoint_hashes.json').read_text()):
  if file_sha256(checkpoint/r['file'])!=r['sha256']:raise ValueError('checkpoint changed')
 from crashbench.policies import OpenVLAPolicy
 from crashbench.envs import LiberoEnv
 from crashbench.branching.state import restore_exact_state,capture_exact_state
 policy=OpenVLAPolicy(pretrained_checkpoint=old['checkpoint'],checkpoint_revision=old['checkpoint_revision'],capture_hidden=True)
 env=LiberoEnv('libero_spatial',0,seed=old['seed']);children=[];unavailable=[];records=[];prefix_cost=[]
 try:
  # Generate all reachable child bundles before scoring any new branch outcomes.
  for eid in config['parents']:
   if file_sha256(parent/eid/'bundle.pkl')!=config['parent_bundle_sha256'][eid]:raise ValueError('parent bundle changed')
   ctx=load(parent/eid/'bundle.pkl');obs=restore_exact_state(ctx['bundle'],env,policy);base=oldanchors[eid];crash=make_crash(env,ctx['glasses']);events=list(ctx['prefix_events']);newcalls=0;start=time.monotonic();terminal_event=None
   with gzip.open(out/(eid+'_extension.pkl.gz'),'xb',compresslevel=1) as stream:
    for offset in range(max(config['offsets'])+1):
     absolute=base['anchor_step']+offset
     if offset==0:action=ctx['pending_action'];po=ctx['policy_input'];latency=ctx['proposal_latency']
     else:action,po,latency=measured_act(env,policy,obs,ctx['instruction']);newcalls+=1
     if offset in config['offsets']:
      cid=f'{eid}_d{offset:02d}';folder=out/cid;folder.mkdir()
      if int(env._raw_env().horizon)<absolute+max(config['suffix_horizons'])+old['settle_steps']:raise ValueError('insufficient environment horizon')
      bundle=capture_exact_state(env,policy,identity={'source':base['source'],'parent':eid,'offset':offset,'episode':cid},provenance={'commit':resolve_git_head(ROOT),'parent_bundle_sha256':config['parent_bundle_sha256'][eid]},declared_branch_seed=ctx['bundle'].rng.declared_branch_seed)
      childctx={'bundle':bundle,'pending_action':np.asarray(action).copy(),'policy_input':po,'instruction':ctx['instruction'],'glasses':ctx['glasses'],'prefix_events':list(events),'proposal_latency':latency,'queue_executed':0,'queue_remaining':1,'post_proposal_queue':0}
      dump(folder/'bundle.pkl',childctx)
      child={k:v for k,v in base.items() if k not in ('bundle_sha256','bundle_id','hidden','robot_state','nominal_action')};child.update(episode_id=cid,parent=eid,offset=offset,anchor_step=absolute,bundle_sha256=file_sha256(folder/'bundle.pkl'),bundle_id=bundle.bundle_id,robot_state=np.asarray(po['state']).tolist(),nominal_action=np.asarray(action).tolist(),bowl=np.asarray(obs['akita_black_bowl_1_pos']).tolist(),plate=np.asarray(obs['plate_1_pos']).tolist(),glass=ctx['glasses'][0],qvel_norm=float(np.linalg.norm(bundle.flat_state[-env._raw_env().sim.model.nv:])))
      children.append(child);write(folder/'anchor.json',child)
     if offset==max(config['offsets']):break
     obs,event=step(env,obs,np.asarray(action),crash,ctx['glasses'],absolute,1,False,stream,po,latency);events.append(event)
     if event['reason']:
      terminal_event=event
      for delayed in config['offsets']:
       if delayed>offset:unavailable.append({'parent':eid,'offset':delayed,'reason':'prefix_'+event['reason'],'terminal_action':event['step']})
      break
   prefix_cost.append({'parent':eid,'new_policy_calls':newcalls,'elapsed_seconds':time.monotonic()-start,'terminal':terminal_event})
   print('children',eid,[c['offset'] for c in children if c['parent']==eid],flush=True)
  write(out/'children.json',children);write(out/'unavailable.json',unavailable);write(out/'prefix_cost.json',prefix_cost)
  for i,child in enumerate(children):
   row=dict(rows[child['parent']],episode_id=child['episode_id']);branchconfig=dict(old,long_H=child['anchor_step']+max(config['suffix_horizons']))
   for repeat in config['repeats']:
    for option in ((0,1) if (config['parents'].index(child['parent'])+child['offset']+repeat)%2==0 else (1,0)):
     r=branch(env,policy,row,child,branchconfig,out/child['episode_id'],repeat,option,audit_controller=True)
     r.update(parent=child['parent'],offset=child['offset'],suffix_horizons={str(h):suffix(r['events'],child['anchor_step'],h) for h in config['suffix_horizons']});records.append(r)
   print('scored',child['episode_id'],len(records),flush=True)
  assert len(records)<=48 and len(children)+len(unavailable)==12
  write(out/'records.json',records);write(out/'complete.json',{'status':'COMPLETE','prefixes':len(prefix_cost),'scored_branches':len(records),'available_children':len(children),'unavailable_children':len(unavailable)})
 finally:env.env.close()
if __name__=='__main__':main()
