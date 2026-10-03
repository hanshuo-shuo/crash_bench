"""Initial readouts and fixed native-policy outcomes on independently labeled cells."""
import argparse,collections,json,os,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import baseline as b
import mechanisms as m
import observe as o
import matrix_cues
r=b.r
P=json.loads((Path(__file__).parent/'matrix_protocol.json').read_text())


def request_arrays(client,folder,data,key,name):
    result=client.infer(dict(data,**{key:True}))
    numeric={k:np.asarray(v) for k,v in result.items() if isinstance(v,(np.ndarray,list,tuple))}
    np.savez_compressed(folder/(name+'.npz'),**numeric)
    r.write(folder/(name+'.json'),{k:v for k,v in result.items() if k not in numeric})
    return numeric,result['metadata']


def execute(root,source,case,client,rollout=True):
    o.P['label_root']=str(source);folder,env,obs,_=o.restore_case(root,case,allow_actions=rollout)
    start=time.monotonic();queue=collections.deque();ref=SimpleNamespace(phase='pi05');requests=0;illegal=False
    try:
        label=json.loads((folder/'INHERITED.json').read_text())['label']
        source_summary=json.loads((source/case/'summary.json').read_text())
        fixture=json.loads((source/case/'fixture.json').read_text())
        for name in ('fixture.json','CERTIFICATE.json','SETTLED_POSE.json'):
            (folder/name).write_bytes((source/case/name).read_bytes())
        audit=m.Audit(env,folder);audit.sample(audit.forward(),'initial_synchronized',True)
        initial_valid=bool(audit.safe and source_summary['initial_valid'])
        r.write(folder/'initial_restore.json',r.capture(env));r.picture(folder/'initial.png',obs)
        data=r.policy_input(obs,b.P['prompt_safe']);np.savez_compressed(folder/'actual_policy_input.npz',**data)
        before=r.statehash(r.capture(env));o.audit_visibility(env,obs,audit,folder,data)
        client.infer({'__paired_reset_rng__':P['seed'],'run_id':case})
        layers,meta=o.save_extraction(client,folder,data,'layer_features')
        attention,ameta=request_arrays(client,folder,data,'__ra_attention__','attention_features')
        generic,gmeta=request_arrays(client,folder,data,'__ra_general_vision__','general_vision')
        matrix_cues.extract(env,audit,folder,attention['action_vision_attention'],ameta,fixture)
        r.write(folder/'FEATURE_AUDIT.json',dict(physics_unchanged=before==r.statehash(r.capture(env)),
            layers=meta,attention=ameta,layer_accepted=meta['trace_native_relative_l2']<=.002,
            attention_accepted=ameta['trace_native_relative_l2']<=.002))
        if before!=r.statehash(r.capture(env)):raise RuntimeError('Feature observation changed physics')
        # Native comparison arrays were already saved. A failed new feature hook
        # does not erase other readouts or relabel the independently labeled state.
        rng=client.infer({'__paired_reset_rng__':P['seed'],'run_id':case})
        row=audit.endpoint(obs,ref,queue=queue,rng=rng)
        do_rollout=bool(rollout and initial_valid and label!='unknown')
        with (folder/'policy.jsonl').open('w') as log:
            if do_rollout:
                for step in range(1,P['execution_horizon_T']+1):
                    if not queue:
                        data=r.policy_input(obs,b.P['prompt_safe']);out=client.infer(data);requests+=1
                        if out['diagnostic']['input_sha256']!=r.digest(data):raise RuntimeError('Policy input provenance mismatch')
                        np.savez_compressed(folder/('input_%03d.npz'%requests),**data)
                        log.write(json.dumps(r.clean(dict(step=step,**out)))+'\n');log.flush()
                        queue.extend(np.asarray(out['actions'])[:P['action_chunk']]);rng=out['diagnostic']['rng_after']
                    raw=queue.popleft().copy();action=raw.copy();action[6]=np.clip(action[6],-1,1)
                    if not r.previous.legal_action(action):
                        illegal=True;r.write(folder/'ILLEGAL.json',dict(step=step,raw=raw,action=action));break
                    audit.step=step;audit.substep=0;obs,_,done,_=env.step(action)
                    row=audit.endpoint(obs,ref,action,queue,rng)
                    if bool(done)!=row['native_success']:raise RuntimeError('Goal bookkeeping mismatch')
                    if step%50==0:r.picture(folder/('frame_%03d.png'%step),obs)
                    if row['safe_success'] or not audit.safe:break
        audit.finish(ref,queue,rng);r.picture(folder/'final.png',obs)
        certificate=json.loads((folder/'CERTIFICATE.json').read_text())
        current_label=m.classify(row['safe_success'] and initial_valid,audit.step,P['execution_horizon_T'],certificate)
        if not initial_valid:current_label='unknown'
        summary=dict(case=case,layout=fixture['layout']['id'],mechanism=fixture['mechanism'],variant=fixture['variant'],
            initial_label=label,label=current_label,outcome=b.outcome(audit.safe,row['safe_success'],illegal or not initial_valid) if do_rollout else 'not_run',
            initial_valid=initial_valid,steps=audit.step,samples=audit.total_samples,safe_history=audit.safe,
            safe_success=row['safe_success'],first_violation=audit.first_violation,policy_requests=requests,
            rollout=do_rollout,execution_horizon_T=P['execution_horizon_T'],seed=P['seed'],wall_seconds=time.monotonic()-start,
            layer_accepted=meta['trace_native_relative_l2']<=.002,attention_accepted=ameta['trace_native_relative_l2']<=.002)
        r.write(folder/'summary.json',summary)
        if do_rollout and not illegal:m.verify(folder)
        print(json.dumps(summary),flush=True);return summary
    finally:env.close()


def main(root,labels,shard,port,pilot):
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    root.mkdir(exist_ok=True);r.configure(root);m.P=P;r.write(root/'protocol.json',P)
    client=WebsocketClientPolicy('127.0.0.1',port)
    if pilot:
        source=labels;cases=['A_slit_open','A_cage_.020']
    else:
        source=labels/('shard_'+str(shard))
        if not (source/'COMPLETE.json').exists():raise RuntimeError('Shard labels not complete')
        cases=[layout['id']+'_'+mechanism+'_'+variant for layout in P['layouts'][2*shard:2*shard+2] for mechanism in P['mechanisms'] for variant in P['variants']]
    rows=[]
    for case in cases:
        rows.append(execute(root,source,case,client,rollout=not pilot));r.write(root/'PROGRESS.json',dict(rows=rows))
    r.write(root/'COMPLETE.json',dict(rows=rows,code_commit=os.environ['CB_CODE_COMMIT'],job=os.environ['SLURM_JOB_ID'],api_calls=0,
        phase='initial-only adapter pilot' if pilot else 'grouped matrix',shard=shard))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--port',type=int,required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    try:main(a.root,a.labels,a.shard,a.port,a.pilot)
    except BaseException as e:
        a.root.mkdir(exist_ok=True);r.write(a.root/'STOP.json',dict(error=type(e).__name__,reason=str(e),job=os.environ.get('SLURM_JOB_ID')));raise
