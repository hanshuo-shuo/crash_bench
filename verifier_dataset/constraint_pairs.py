"""Collect rule-counterfactual pairs from exact native source trajectories.

All simulator annotations are noncolliding. Every original action-state hash is
still required to match. Each source is executed once; a persisted suffix and
an independently checked separator support each counterfactual pair.
"""
import argparse
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from .inventory import read,sha,write
from .replay import ARRAYS,assert_trace
from .paired_scene import make_env
from .paired_geometry import material_sphere,body_bounds,destination,disjoint
from .paired_goal import definition,snapshot,evaluate
from .contact_contract import actor_geometry,prohibited
from .constraint_evidence import CONTRACT,verify_trace,verify_layout,rows as make_rows
from . import constraint_visuals


def layouts(env,goal,step,sample_index,horizon,obstacle,source_id):
    import numpy as np
    from .paired_inputs import receiver_phrase
    receiver_phrase(goal)  # Unsupported task wording cannot become a label.
    # Native object-On requires XY distance <3 cm. Declare a fixed 4 cm
    # half-width around the INITIAL receiver, allowing 1 cm more per axis;
    # this is an explicit extra goal and never uses the future witness endpoint.
    target=body_bounds(env,goal[1]);dest=destination(env,goal,object_xy_halfwidth=.04)['fixed_root_region']
    a=[(np.asarray(target[0])+[-.015,-.015,-.02]).tolist(),
       (np.asarray(target[1])+[.015,.015,.015]).tolist()]
    material=material_sphere(env,goal[1]);center=np.mean(a,axis=0);half=(np.asarray(a[1])-a[0])/2
    goal_center=np.mean(dest,axis=0);axis=goal_center-center;axis[2]=0
    if np.linalg.norm(axis)<1e-8:return []
    axis/=np.linalg.norm(axis);perpendicular=np.asarray([-axis[1],axis[0],0.])
    choices=[]
    widths={role:.0012+.0024*int.from_bytes(hashlib.sha256((source_id+'|'+str(step)+'|'+role+'|appearance-v1').encode()).digest()[:8],'big')/2**64
            for role in ('a','b')}
    layout_seed=hashlib.sha256((source_id+'|'+str(step)+'|layout-v1').encode()).digest()
    offset=.18+.16*int.from_bytes(layout_seed[:8],'big')/2**64
    signs=(1,-1) if layout_seed[8]%2==0 else (-1,1)
    for sign in signs:
        bcenter=(center+goal_center)/2+sign*offset*perpendicular;bcenter[2]=center[2]
        b=[(bcenter-half).tolist(),(bcenter+half).tolist()]
        if not disjoint(a,b) or not disjoint(dest,b):continue
        choices.append(dict(contract=CONTRACT,native_goal=goal,native_obstacle=obstacle,
            checkpoint=step,sample_index=sample_index,remaining_horizon=horizon-step,
            a=a,b=b,target=target,goal=dest,material=material,
            geometry_source='Current initial geometry only; candidate locations are fixed before inspecting future actions',
            object_receiver_region='Initial receiver XY +/-4 cm, Z -1..25 cm; native On predicate also required',
            render_widths=widths,appearance_note='Boundary stroke widths vary by a fixed source/checkpoint seed, independently of the protected-color query; physics and guard bounds are unchanged',
            alternate_region_offset_metres=offset,alternate_region_sign=sign,layout_seed_sha256=layout_seed.hex(),
            fixed_virtual_boundaries=True,physical_collision_walls=False))
    return choices


def execute(case,maximum):
    import numpy as np
    import mujoco
    env=None;original_step=None;stream=None
    try:
        folder=case/'execution';folder.mkdir()
        env,physical_hash,update_marker,source,context=make_env(case,folder,render=True,
            visual_transform=constraint_visuals.add)
        if context['step']!=0:raise RuntimeError('One complete native replay per original source is required')
        goals=[[str(g[0]).lower()]+list(g[1:]) for g in env.env.parsed_problem['goal_state']]
        goals=[g for g in goals if len(g)==3 and g[0] in ('on','in') and not env.env._eval_predicate(g)]
        if not goals:
            write(case/'COMPLETE.json',dict(passed=True,matched_units=0,rows=[],source=source,reason='No supported unsolved atomic native goal'));return
        goal=goals[0];m,d=env.sim.model,env.sim.data
        native_geo=read(Path(source['root'])/'geometry_details.json')
        protected={m.geom_name2id(g['name']) for g in native_geo['protected']}
        source_row=read(case/'INPUT.json')
        scope=actor_geometry(env,protected,source_row['student']['task']);actors=set(scope['actors'])
        obstacle=read(Path(source['root'])/'geometry.json')['obstacle']
        spec=definition(env,goal)
        try:material=material_sphere(env,goal[1])
        except RuntimeError as error:
            write(case/'COMPLETE.json',dict(passed=True,matched_units=0,rows=[],source=source,reason=str(error)));return
        geom=m.geom_name2id(material['geom_name']);body=int(env.env.obj_body_id[goal[1]])
        shadow=mujoco.MjData(m._model);step=0;substep=0;safe=True;samples=[];actions=[];candidates=[]
        stream=gzip.open(folder/'SAMPLES.jsonl.gz','wt')
        def sample(phase):
            nonlocal safe
            before=physical_hash(env)
            for name in ARRAYS:getattr(shadow,name)[:]=np.asarray(getattr(d,name))
            shadow.time=d.time;mujoco.mj_forward(m._model,shadow)
            record=dict(step=step,substep=substep,phase=phase,native_predicate=bool(env.env._eval_predicate(goal)))
            for mode,data in [('native',d),('synchronized',shadow)]:
                contacts=[[int(c.geom1),int(c.geom2),float(c.dist)] for c in data.contact[:data.ncon]]
                record[mode+'_protected_contacts']=[c for c in contacts if c[0] in protected or c[1] in protected]
                record[mode+'_point']=np.asarray(data.geom_xpos[geom]).tolist()
                record[mode+'_root']=np.asarray(data.body_xpos[body] if mode=='native' else data.xpos[body]).tolist()
                state=snapshot(data,spec,synchronized=mode=='synchronized')
                record[mode+'_goal_state']=state;record[mode+'_goal_contacts']=contacts
                record[mode+'_goal_success']=evaluate(spec,state,contacts)
                if mode=='native' or phase in ('initial','endpoint'):
                    safe &= not any(prohibited((a,b),protected,actors) and dist<=0 for a,b,dist in contacts)
            record['safe_history']=safe
            if record['native_goal_success']!=record['native_predicate']:raise RuntimeError('Independent native goal mismatch')
            if physical_hash(env)!=before:raise RuntimeError('Read-only source check mutated physics')
            stream.write(json.dumps(record)+'\n');samples.append(record);return record
        initial=sample('initial')
        geometry=dict(contract=CONTRACT,goal_definition=spec,protected=sorted(protected),actors=sorted(actors),
            contact_scope=scope,material=material,native_goal=goal,all_region_geoms_noncolliding=True,
            contact_sampling='Native initial/every integration and synchronized initial/action endpoints',
            virtual_guard_sampling='Native and synchronized material centers at every integration and endpoint')
        write(folder/'GEOMETRY.json',geometry)
        original_step=env.sim.step
        def monitored(*args,**kwargs):
            nonlocal substep
            result=original_step(*args,**kwargs);substep+=1;sample('integration');return result
        env.sim.step=monitored
        tape_path=Path(source['root'])/'steps.jsonl'
        if sha(tape_path)!=source['tape_sha256']:raise RuntimeError('Original source tape changed')
        tape=[json.loads(line) for line in tape_path.read_text().splitlines()];assert_trace(tape,source['actions'])
        from .replay import checkpoint_steps
        wanted=set(checkpoint_steps(len(tape),maximum=maximum,schedule='early'))
        low,high=map(np.asarray,env.env.action_spec);last=initial
        for item in tape:
            t=item['step']-1
            if t in wanted and not last['native_goal_success'] and not last['synchronized_goal_success'] and last['safe_history']:
                try:proposals=layouts(env,goal,t,len(samples)-1,context['remaining_horizon'],obstacle,source['source_id'])
                except RuntimeError as error:
                    write(case/('checkpoint_%03d_REJECTED.json'%t),dict(reason=str(error),label='unknown'));proposals=[]
                for index,layout in enumerate(proposals):
                    option=case/('checkpoint_%03d_option_%d'%(t,index));option.mkdir()
                    write(option/'LAYOUT.json',layout)
                    images=constraint_visuals.capture(env,option,layout,physical_hash)
                    candidates.append(dict(folder=option,layout=layout,images=images))
            step=item['step'];substep=0;action=np.clip(item['applied'],low,high)
            obs,_,done,_=env.step(action.tolist());actual=physical_hash(env)
            equal=actual==item['physics_after_sha256']
            if not equal or bool(done)!=item['success']:
                write(folder/'MISMATCH.json',dict(step=step,expected=item['physics_after_sha256'],actual=actual,
                    expected_success=item['success'],actual_success=bool(done)))
                raise RuntimeError('Annotated source trajectory differs at '+str(step))
            actions.append(dict(step=step,action=action.tolist(),source_physics_equal=equal,actual_physics_sha256=actual))
            last=sample('endpoint');update_marker(obs)
        stream.close();stream=None
        write(folder/'ACTIONS.json',actions);(folder/'model.xml').write_text(m.get_xml())
        check=verify_trace(samples,actions,geometry)
        write(folder/'TRACE_VERIFIED.json',check)
        all_rows=[];outcomes=[];selected_steps=set()
        for c in candidates:
            option,layout=c['folder'],c['layout'];verification=verify_layout(layout,samples,actions)
            write(option/'VERIFIED.json',verification)
            visibility=constraint_visuals.prescreen(read(option/'VISIBILITY.json'))
            chosen=verification['matched_pair'] and visibility['quantitative_prescreen'] and layout['checkpoint'] not in selected_steps
            if chosen:
                selected_steps.add(layout['checkpoint']);records=make_rows(layout,verification,source,c['images'])
                for row in records:
                    row['metadata'].update(evidence_root=str(option),trace_root=str(folder),
                        source_witness=source_row['evidence']['witness_root'],
                        ancestry_component=source_row['metadata']['ancestry_component'])
                    for image in row['student']['images']:image['path']=str(option/image['path'])
                write(option/'CANDIDATES.json',records);all_rows.extend(records)
            write(option/'HASHES.json',{str(p.relative_to(option)):sha(p) for p in option.rglob('*') if p.is_file()})
            outcomes.append(dict(option=str(option),checkpoint=layout['checkpoint'],matched_pair=verification['matched_pair'],
                selected_first_valid_option=chosen,visibility_prescreen=visibility,
                positive_witness=verification['positive_witness'],conditional_negative=verification['conditional_negative']))
        write(case/'CANDIDATES.json',all_rows)
        write(case/'COMPLETE.json',dict(passed=True,source=source,rows=outcomes,matched_units=len(selected_steps),
            candidate_rows=len(all_rows),native_trace=check,training_allowed=False,api_calls=0,
            code_commit=os.environ['CB_CODE_COMMIT'],slurm_job=os.environ['SLURM_JOB_ID']))
        write(folder/'HASHES.json',{p.name:sha(p) for p in folder.iterdir() if p.is_file()})
    finally:
        if stream is not None:stream.close()
        if env is not None:
            if original_step is not None:delattr(env.sim,'step')
            env.close()


def main(root,case_index=None):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Quest compute allocation required')
    maximum=read(root/'CAMPAIGN.json')['maximum_checkpoints']
    if case_index is not None:execute(root/('case_%03d'%case_index),maximum);gc.collect();return
    device=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip()
    write(root/'GPU.json',dict(devices=device,scope='OpenGL rendering only; no neural policy inference',slurm_job=os.environ['SLURM_JOB_ID']))
    outcomes=[];all_rows=[]
    for index,row in enumerate(read(root/'CASES.json')):
        case=root/('case_%03d'%index);case.mkdir();write(case/'INPUT.json',row)
        witness=Path(row['evidence']['witness_root']);checkpoint=witness/'checkpoint_000'
        if row['evidence']['checkpoint_step']!=0 or sha(checkpoint/'context.json')!=row['evidence']['checkpoint_context_sha256']:
            raise RuntimeError('Original canonical checkpoint changed')
        for source,dest in [('context.json','CANONICAL_CONTEXT.json'),('physics.npz','CANONICAL_PHYSICS.npz')]:
            shutil.copyfile(checkpoint/source,case/dest)
        with (case/'worker.log').open('w') as log:
            subprocess.run([sys.executable,'-u','-m','verifier_dataset.constraint_pairs',str(root),'--case-index',str(index)],
                stdout=log,stderr=subprocess.STDOUT,check=True)
        result=read(case/'COMPLETE.json');outcomes.append(result)
        if (case/'CANDIDATES.json').exists():all_rows.extend(read(case/'CANDIDATES.json'))
        write(root/'PROGRESS.json',dict(completed=len(outcomes),expected=len(read(root/'CASES.json')),candidate_rows=len(all_rows)))
    write(root/'CONSTRAINT_CANDIDATES.json',all_rows)
    write(root/'CONSTRAINT_COMPLETE.json',dict(passed=True,rows=outcomes,matched_units=sum(r['matched_units'] for r in outcomes),
        candidate_rows=len(all_rows),final_reasoning_targets=0,training_allowed=False,api_calls=0,slurm_job=os.environ['SLURM_JOB_ID']))
    write(root/'HASHES.json',{name:sha(root/name) for name in ('CONSTRAINT_CANDIDATES.json','CONSTRAINT_COMPLETE.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--case-index',type=int)
    a=p.parse_args();main(a.root,a.case_index)
