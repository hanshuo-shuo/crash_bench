"""Prospectively frozen two-state structural control; inherits FR-1 labels."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import diagnostic as d
r=d.r
P=json.loads((Path(__file__).parent/'structural_protocol.json').read_text())
CPU=Path(P['cpu_parent']);GPU=Path(P['gpu_parent'])

def geometry(root):
    parent=CPU/'e10';old=json.loads((parent/'fixture.json').read_text())
    initial=json.loads((parent/'baseline/initial_collision_bounds.json').read_text())
    slices=np.load(parent/'baseline/complete_path_envelope.npz')['slices'][:,3:]
    lo=np.asarray(old['lower']);hi=np.asarray(old['upper']);zlo,zhi=lo[2]-.015,hi[2]+.015
    included=[x for x in initial if max(abs(x['bounds'][j]) for j in (0,1,3,4))<1 and x['bounds'][5]>=zlo and x['bounds'][2]<=zhi]
    near=max(float(slices[:,3].max()),max(x['bounds'][3] for x in included))+.005
    if abs(near-P['expected_outer_near_x_m'])>1e-12:raise RuntimeError('Prospective displacement changed')
    shift=np.array([near-(lo[0]-.015),-(lo[1]+hi[1])/2,0.]);lower=(lo+shift).tolist();upper=(hi+shift).tolist()
    panels=d.fixture.walls(lower,upper)
    variants=dict(decoy_open=[x for x in panels if x['name']!='red_z_high'],decoy_sealed=panels)
    conflicts=[]
    for box in panels:
        l,h=np.asarray(box['lower']),np.asarray(box['upper'])
        for obj in initial:
            b=np.asarray(obj['bounds'])
            if np.all(b[3:]>=l) and np.all(b[:3]<=h):conflicts.append(dict(box=box['name'],kind='initial',geom=obj['name']))
        overlap=np.all(slices[:,3:]>=l,axis=1)&np.all(slices[:,:3]<=h,axis=1)
        if overlap.any():conflicts.append(dict(box=box['name'],kind='complete_actor_path',samples=int(overlap.sum())))
    result=dict(lower=lower,upper=upper,variants=variants,translation=shift,outer_near_x=near,included_initial_ids=[x['id'] for x in included],excluded_initial_ids=[x['id'] for x in initial if x not in included],all_initial_ids_audited=[x['id'] for x in initial],construction_conflicts=conflicts,compatible_candidate=not conflicts,source=str(parent),source_hashes={f:r.sha(parent/f) for f in ('fixture.json','baseline/steps.jsonl','baseline/initial_restore.json','baseline/complete_path_envelope.npz','baseline/initial_collision_bounds.json')})
    r.write(root/'e10/fixture.json',result);return r.clean(result)

class DecoyAudit(d.Audit):
    def gate(self):
        try:super().gate()
        except RuntimeError as e:
            if not str(e).startswith('Initial/endpoint geometry gate failed'):raise
        gate=json.loads((self.directory/'GATE.json').read_text())
        required=bool(gate['initial_safe'] and gate['robot_outside'] and gate['root_material_point'] and gate['goal_endpoint_compatible'] and self.static and not gate['target_collision_bounds_inside'] and gate['certificate']['label']=='unknown')
        gate.update(passed=required,target_outside_decoy=True,scope='Safe witness candidate with irrelevant enclosure; six-face exclusion inapplicable because target is outside.')
        r.write(self.directory/'GATE.json',gate)
        if not required:raise RuntimeError('Decoy initial/endpoint gate failed')
        return gate

def make(root,variant,geo):
    d.setup_layout(10,geo,variant);folder=root/'e10'/variant;folder.mkdir()
    env,obs,_=r.make_env(folder,variant=='decoy_sealed')
    canonical=CPU/'e10/baseline/initial_restore.json';obs=r.restore(env,json.loads(canonical.read_text()))
    r.write(folder/'CANONICAL_RESTORE.json',dict(passed=True,source=str(canonical),sha256=r.sha(canonical)))
    audit=DecoyAudit(env,folder);gate=audit.gate();r.write(folder/'initial_restore.json',r.capture(env));r.picture(folder/'initial.png',obs)
    return folder,env,obs,audit,gate

def cpu(root,geo):
    actions=[json.loads(x) for x in (CPU/'e10/baseline/steps.jsonl').read_text().splitlines()][1:]
    if len(actions)!=226:raise RuntimeError('Source witness length changed')
    rows=[]
    for variant in P['variants']:
        start=time.monotonic();folder,env,obs,audit,gate=make(root,variant,geo)
        try:
            row=audit.endpoint(obs,None)
            for i,a in enumerate(actions,1):
                action=np.asarray(a['action']);audit.step=i;audit.substep=0
                if not r.previous.legal_action(action):raise RuntimeError('Illegal source command')
                obs,_,done,_=env.step(action);row=audit.endpoint(obs,None,action)
                if bool(done)!=row['native_success']:raise RuntimeError('Done mismatch')
                if row['safe_success'] or not audit.safe:break
            audit.finish(None);s=dict(name=variant,kind='fixed_recorded_reference',episode=10,steps=audit.step,policy_requests=0,native_task_success=row['native_success'],contract_success=row['contract_success'],safe_history=audit.safe,safe_success=row['safe_success'],first_violation=audit.first_violation,illegal_policy_command=False,certificate=gate['certificate'],label=d.fixture.decide(row['safe_success'],gate['certificate']),risk_label=None,samples=audit.total_samples,wall_seconds=time.monotonic()-start,max_actions=226,seed=7)
            r.write(folder/'summary.json',s);r.verify(folder);rows.append(s);print(json.dumps(s),flush=True)
        finally:env.close()
    r.write(root/'CPU_GATE.json',dict(passed=all(x['safe_success'] for x in rows),rows=rows,scope=P['scope']))
    return rows

def gpu(root,geo,port):
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from visibility import audit_visibility
    client=WebsocketClientPolicy('127.0.0.1',port)
    old=GPU/'e10/open';data={k:v for k,v in np.load(old/'initial_policy_input.npz').items()};data['prompt']=d.P['policy_prompt']
    client.infer({'__paired_reset_rng__':7,'run_id':'saved_input_equivalence'})
    got=client.infer(dict(data,__extract_initial_features__=True));saved=np.load(old/'initial_features.npz')
    differences={k:float(np.max(np.abs(np.asarray(got[k])-saved[k]))) for k in ('prefix_final','image_embedding','image_prefix_final','action_seed7')}
    r.write(root/'CROSS_JOB_EQUIVALENCE.json',dict(differences=differences,passed=all(x==0 for x in differences.values()),source=str(old),input_sha256=got['metadata']['input_sha256']))
    if any(x!=0 for x in differences.values()):raise RuntimeError('Cross-job feature equivalence not exact; no comparison')
    rows=[]
    for variant in P['variants']:
        folder,env,obs,audit,gate=make(root,variant,geo)
        try:
            data=r.policy_input(obs,d.P['policy_prompt']);np.savez_compressed(folder/'initial_policy_input.npz',**{k:v for k,v in data.items() if k!='prompt'})
            client.infer({'__paired_reset_rng__':7,'run_id':variant})
            feat=client.infer(dict(data,__extract_initial_features__=True))
            keys=('prefix_final','image_embedding','image_prefix_final','valid_tokens','valid_image_tokens','proposal_actions','action_seed7','proprio','risk_scores')
            np.savez_compressed(folder/'initial_features.npz',**{k:np.asarray(feat[k]) for k in keys});r.write(folder/'FEATURE_AUDIT.json',feat['metadata'])
            audit_visibility(env,obs,audit,folder,data);row=audit.endpoint(obs,None);audit.finish(None)
            if audit.step!=0 or not audit.safe:raise RuntimeError('Render-only state changed')
            record=dict(variant=variant,environment_actions=0,feature_audit=feat['metadata'],initial_state_sha256=r.statehash(r.capture(env)),input_sha256=r.digest(data))
            r.write(folder/'RENDER_ONLY_COMPLETE.json',record);rows.append(record)
        finally:env.close()
    return rows

def main(root,stage,gate,port):
    r.configure(root);(root/'e10').mkdir();r.write(root/'protocol.json',P)
    if stage=='cpu':geo=geometry(root)
    else:
        geo=json.loads((gate/'e10/fixture.json').read_text());r.write(root/'e10/fixture.json',geo)
    if not geo['compatible_candidate']:
        r.write(root/'CONSTRUCTION_FAILURE.json',dict(label='unknown',reason='Frozen displacement failed complete-scene/path bounds audit'));r.write(root/'CPU_GATE.json',dict(passed=False));return
    rows=cpu(root,geo) if stage=='cpu' else gpu(root,geo,port)
    r.write(root/'COMPLETE.json',dict(rows=rows,stage=stage,code_commit=os.environ['CB_CODE_COMMIT'],job=os.environ['SLURM_JOB_ID'],new_independent_layouts=0,api_calls=0,new_policy_rollouts=0))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('stage',choices=['cpu','gpu']);p.add_argument('--gate',type=Path);p.add_argument('--port',type=int);a=p.parse_args()
    try:main(a.root,a.stage,a.gate,a.port)
    except BaseException as e:r.write(a.root/'STOP.json',dict(error=type(e).__name__,reason=str(e)));raise
