"""Authorized fixed train-q90 zero-command gate around the nominal evaluator."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import time
from common import atomic_json,scene_id,seed_for,noise_seed
from adapter import adapt,replace_once


def command(raw,disagreement,threshold):
    if len(raw)!=7 or not all(math.isfinite(float(x)) for x in raw) or not math.isfinite(disagreement) or not math.isfinite(threshold):raise ValueError('Finite seven-axis gate inputs required')
    active=disagreement>threshold
    return [0.]*7 if active else [float(x) for x in raw],active


def adapt_gate(source,method):
    if method!='gate':raise ValueError('Gate runner only accepts nominal zero-command arm')
    nominal=adapt(source,'nominal')
    old='                        observer.candidate(action, action, t, "off")\n                        obs, reward, done, info = env.step(action.tolist()) # Crucial step\n'
    new='                        gated_command = observer.apply_gate(action, t)\n                        observer.candidate(action, gated_command, t, "gate" if observer.gate_active else "off")\n                        obs, reward, done, info = env.step(gated_command.tolist()) # Crucial step\n'
    return replace_once(nominal,old,new)


def schedule(cfg,shard=None):
    manifest=json.loads((Path(__file__).parent/'states.json').read_text())['states']
    if len(manifest)!=60 or cfg['repeats']!=list(range(5)):raise ValueError('Fixed60×5 gate matrix required')
    specs=[dict(name='gate_s%02d_r%02d'%(i,r),scene=scene,repeat=r,method='gate',diagnostics=True,state_index=i)
           for i,scene in enumerate(manifest) if shard is None or i%2==shard for r in cfg['repeats']]
    front=[x for x in specs if x['state_index'] in [0,42] and x['repeat'] in [0,1]]
    return front+[x for x in specs if x not in front]


def install():
    import inspect
    import numpy as np
    import runtime
    from preflight import stream_sha
    class GateObserver(runtime.Observer):
        def perception(self,*args):raise runtime.Failure('Gate nominal arm must never request a VLM')

        def ready(self,env,obs,obstacle,enabled):
            if enabled:raise runtime.Failure('Gate arm unexpectedly enabled AEGIS')
            super().ready(env,obs,obstacle,enabled)
            controller=env.robots[0].controller;gripper=env.robots[0].gripper
            if controller.control_dim!=6 or not controller.use_delta or not controller.use_ori or controller.impedance_mode!='fixed':raise runtime.Failure('Unexpected OSC command convention')
            if not np.array_equal(np.asarray(controller.input_min),-np.asarray(controller.input_max)) or not np.array_equal(np.asarray(controller.output_min),-np.asarray(controller.output_max)):
                raise runtime.Failure('Zero command does not map to zero pose delta')
            if type(gripper).__name__!='PandaGripper':raise runtime.Failure('Unexpected gripper zero convention')
            expected={x['sha256'] for x in self.cfg['controller_files']}
            sources=[inspect.getsourcefile(type(x)) for x in [controller,gripper]]
            if any(stream_sha(Path(p)) not in expected for p in sources):raise runtime.Failure('Actual controller/gripper implementation differs')
            atomic_json(self.directory/'ZERO_COMMAND_SEMANTICS.json',dict(controller=type(controller).__name__,use_delta=controller.use_delta,impedance_mode=controller.impedance_mode,
                input_min=np.asarray(controller.input_min).tolist(),input_max=np.asarray(controller.input_max).tolist(),output_min=np.asarray(controller.output_min).tolist(),output_max=np.asarray(controller.output_max).tolist(),
                source_sha256={p:stream_sha(Path(p)) for p in sources},gripper=type(gripper).__name__,
                interpretation='Zero position delta tracks current EEF position; zero orientation delta retains prior orientation goal; zero gripper retains accumulated actuator target. Physics/torques continue; no mechanical rest guarantee.'))

        def apply_gate(self,raw,t):
            if not 0<=t-(self.latest['step']-1)<5:raise runtime.Failure('Gate command escaped its five-action infer queue')
            values,self.gate_active=command(raw,self.latest['disagreement'],self.cfg['gate_threshold'])
            return np.asarray(values,dtype=np.asarray(raw).dtype)

        def candidate(self,raw,applied,t,status):
            super().candidate(raw,applied,t,status)
            self.pending.update(proposed=np.asarray(raw).tolist(),gate_active=self.gate_active,gate_threshold=self.cfg['gate_threshold'],gate_disagreement=self.latest['disagreement'])

    runtime.Observer=GateObserver;runtime.adapt=adapt_gate
    return runtime


def check(root,results):
    import numpy as np
    from common import norm,outcome,time_to_crash
    total=0;gated_actions=0;gated_inferences=0;rows=[];risk_inferences=0;risk_gated=0
    cfg=json.loads((root/'plan.json').read_text())['configuration']
    for result in results:
        p=root/'runs'/result['spec']['name'];infos=[json.loads(x) for x in (p/'inferences.jsonl').read_text().splitlines()]
        spec=result['spec'];scene=scene_id(spec['scene'])
        if result['status']!='complete' or result['seed']!=seed_for(spec['scene'],spec['repeat']) or len(infos)!=result['inferences']:raise RuntimeError('Gate identity/seed/infer count differs')
        baseline=Path(cfg['collection_root'])/'shards'/str(spec['state_index']%2)/'runs'/('full_s%02d_r%02d_nominal'%(spec['state_index'],spec['repeat']))
        with np.load(p/'settled_state.npz',allow_pickle=False) as actual,np.load(baseline/'settled_state.npz',allow_pickle=False) as original:
            if actual.files!=original.files or any(not np.array_equal(actual[k],original[k]) for k in actual.files):raise RuntimeError('Gate/nominal paired settled physical arrays differ')
        actions=[json.loads(x) for x in (p/'steps.jsonl').read_text().splitlines()];r=[json.loads(x) for x in (p/'rows.jsonl').read_text().splitlines()]
        if len(actions)!=result['actions'] or len(r)!=len(actions) or not json.loads((p/'CONDITIONAL_ACTION_EQUALITY.json').read_text())['passed']:raise RuntimeError('Gate evidence incomplete or native invariance failed')
        chunks={};clock={};previous=None;last_rng=None;fixed_scales=None
        for info in infos:
            if info['infer_index']!=len(clock)+1 or info['step']!=1+5*len(clock) or (last_rng is not None and info['rng_before']!=last_rng):raise RuntimeError('Gate infer/action/native RNG clock differs')
            last_rng=info['rng_after']
            with np.load(p/('infer_%03d.npz'%info['infer_index']),allow_pickle=False) as values:
                samples=values['diagnostic_actions'];scales=values['normalization_scales'];chunk=values['actions']
                if samples.shape!=(8,10,7) or not np.isfinite(samples).all() or not info['native_action_array_equal'] or not info['rng_unchanged']:raise RuntimeError('Gate diagnostic/native proof differs')
                expected_seeds=[noise_seed(scene,result['seed'],info['infer_index'],j) for j in range(8)]
                if values['sample_seeds'].tolist()!=expected_seeds or info['sample_seeds']!=expected_seeds or scales.shape!=(7,) or not np.isfinite(scales).all() or (scales<=0).any():raise RuntimeError('Gate seed/scales differ')
                if not np.array_equal(scales,info['normalization_scales']) or (fixed_scales is not None and not np.array_equal(scales,fixed_scales)):raise RuntimeError('Gate normalization changed')
                fixed_scales=scales.copy()
                if abs(float(np.std(samples/scales[None,None,:],axis=0,ddof=1).mean())-info['disagreement'])>1e-12:raise RuntimeError('Gate independent diagnostic reduction differs')
                churn=None if previous is None else float(np.mean(np.abs(chunk[:5]-previous[5:])/scales))
                if (churn is None and info['churn'] is not None) or (churn is not None and abs(churn-info['churn'])>1e-12):raise RuntimeError('Gate shifted churn differs')
                chunks[info['infer_index']]=chunk.copy();clock[info['infer_index']]=info;previous=chunk.copy()
            gated_inferences+=int(info['disagreement']>cfg['gate_threshold']);total+=1
        C=next((a['step'] for a in actions if a['obstacle_l1_m']>.001),None)
        if C!=result['collision_step'] or result['outcome']!=outcome(result['success'],C):raise RuntimeError('Gate official outcome differs')
        risk=[x for x in infos if C is None or x['step']<=C];risk_inferences+=len(risk);risk_gated+=sum(x['disagreement']>cfg['gate_threshold'] for x in risk)
        for action,row in zip(actions,r):
            t=action['step'];info=clock[action['infer_index']];raw=chunks[action['infer_index']][(t-1)%5]
            expected,active=command(raw,info['disagreement'],cfg['gate_threshold'])
            if not np.array_equal(raw,action['raw']) or not np.array_equal(raw,action['proposed']) or not np.array_equal(np.asarray(expected,dtype=raw.dtype),action['applied']) or action['gate_active']!=active:
                raise RuntimeError('Gate raw/proposed/applied/decision differs')
            if not action['queue_array_equal'] or not action['physical_before_action_unchanged'] or abs(norm(expected,scales)-row['act_norm'])>1e-12:raise RuntimeError('Gate queue/physics/norm differs')
            if row['crashed']!=(C is not None and t>=C) or row['time_to_crash']!=time_to_crash(t,C) or row['outcome']!=result['outcome'] or row['capability']!='gate':raise RuntimeError('Gate row crash clock/outcome differs')
            if row['step']!=t or row['scene_id']!=scene or row['seed']!=result['seed'] or row['disagreement']!=info['disagreement'] or row['churn']!=info['churn'] or row['min_dist']!=action['min_dist'] or row['infer_boundary']!=(info['step']==t):raise RuntimeError('Gate scalar/action row pairing differs')
            gated_actions+=int(active)
        rows+=r
    return dict(passed=True,runs=len(results),actions=len(rows),inferences=total,gated_actions=gated_actions,gated_inferences=gated_inferences,at_risk_inferences=risk_inferences,at_risk_gated_inferences=risk_gated,independent_all_sample_reduction=True,native_same_input_rng_exact=True,paired_settled_physical_arrays_exact=True,threshold=cfg['gate_threshold']),rows


def main(root,port):
    runtime=install();plan=json.loads((root/'plan.json').read_text());cfg=plan['configuration'];runner=runtime.Runner(root,port,cfg);results=[];drain=[False]
    signal.signal(signal.SIGUSR1,lambda *_:drain.__setitem__(0,True))
    deadline=int(os.environ['CB_ALLOCATION_START'])+cfg['resources']['minutes']*60
    try:
        from gate_inheritance import read_inherited
        results=read_inherited(root,schedule(cfg,plan['shard']))
        if results:
            proof,_=check(root,results);proof.update(states=[0,42],repeats=[0,1],new_vlm_calls=0,inherited_smoke=True)
            atomic_json(root/'GATE_SMOKE_PASS.json',proof)
            atomic_json(root/'progress.json',dict(completed=len(results),expected=150,results=results,inherited=4))
        for spec in schedule(cfg,plan['shard'])[len(results):]:
            if (root.parent.parent/'STOP.json').exists():raise RuntimeError('Gate campaign stopped')
            if drain[0] or (root/'DRAIN_SIGNAL').exists() or time.time()+cfg['case_checkpoint_seconds']>=deadline:
                atomic_json(root/'PARTIAL.json',dict(completed=len(results),expected=150,reason='Whole-case allocation checkpoint margin',results=results));return
            result=runner.run(spec);results.append(result);atomic_json(root/'progress.json',dict(completed=len(results),expected=150,results=results))
            if plan['shard']==0 and len(results)==4:
                proof,_=check(root,results);proof.update(states=[0,42],repeats=[0,1],new_vlm_calls=0)
                atomic_json(root/'GATE_SMOKE_PASS.json',proof)
        atomic_json(root/'COMPUTE_COMPLETE.json',dict(completed=len(results),expected=150,results=results,slurm_job=os.environ['SLURM_JOB_ID']))
    except BaseException as error:
        atomic_json(root/'STOP.json',dict(stage='gate_compute',error=type(error).__name__,reason=str(error),completed=len(results),slurm_job=os.environ['SLURM_JOB_ID']));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--port',type=int,required=True);a=p.parse_args();main(a.root,a.port)
