"""Three executed mechanism checks, explicitly excluded from training.

The open proposal has only a floor, so it cannot satisfy appearance matching.
The closed/mug-subtask arm tests whether identical initial geometry can support
a different verdict when the required object changes. Failure stays UNKNOWN.
"""
import argparse
import gc
import gzip
import itertools
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import xml.etree.ElementTree as ET
from .inventory import read,sha,write
from .replay import ARRAYS,configure,assert_trace
from .fixture_design import TARGET,GOAL
from .geometry_facts import initial_geometry
from experiments.reach_avoid.certificate import certify,inscribed_radius
from experiments.red_enclosure.contract import segment_box
from experiments.feasibility_contract.contract import SOURCE_HASHES

VARIANTS=('closed_full','open_full','closed_mug')
MUG_GOAL=['on','porcelain_mug_1','plate_1']


def prohibited_contact(a,b,distance,dynamic,protected,rigid_groups):
    # A movable protected object can itself hit a newly protected static wall.
    # Exclude only geometry belonging to the same rigid assembly, not all
    # protected geometries from the set of moving bodies.
    if distance>0 or rigid_groups[str(a)]==rigid_groups[str(b)]:return False
    return (a in protected and b in dynamic) or (b in protected and a in dynamic)


def add_boxes(xml,boxes):
    tree=ET.fromstring(xml);world=tree.find('worldbody')
    for box in boxes:
        lo,hi=box['lower'],box['upper']
        if tree.find('.//*[@name="%s"]'%box['name']) is not None:raise RuntimeError('Duplicate fixture geometry')
        if not all(math.isfinite(x) for x in lo+hi) or not all(a<b for a,b in zip(lo,hi)):
            raise RuntimeError('Invalid proposed wall')
        ET.SubElement(world,'geom',dict(name=box['name'],type='box',
            pos=' '.join(format((a+b)/2,'.17g') for a,b in zip(lo,hi)),
            size=' '.join(format((b-a)/2,'.17g') for a,b in zip(lo,hi)),
            rgba='.20 .45 .70 1',contype='1',conaffinity='1',group='1',friction='1 .005 .0001',solref='.02 1'))
    return ET.tostring(tree,encoding='unicode')


def check_saved(folder):
    """Recompute the separator and every saved contact/root-segment guard."""
    geo=read(folder/'EXECUTED_GEOMETRY.json');summary=read(folder/'SUMMARY.json')
    radii=[inscribed_radius(b['half_sizes'],b['root_local']) for b in geo['material_evidence']['collision_boxes']]
    if geo['radius']!=max(radii+[0.]) or geo['material_verified']!=(geo['radius']>0):
        raise RuntimeError('Material ball is not supported by the recorded collision boxes')
    recomputed=certify(geo['boxes'],geo['lower'],geo['upper'],geo['initial_root'],geo['goal_bounds'],
        geo['radius'],geo['initial_safe'],geo['static'],geo['material_verified'],True)
    if recomputed!=read(folder/'PUDDING_CERTIFICATE.json'):raise RuntimeError('Persisted separator mismatch')
    dynamic=set(geo['dynamic']);protected=set(geo['protected']);rho=recomputed['world_cube_halfwidth']
    previous={};safe=True;count=0
    with gzip.open(folder/'SAMPLES.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line);count+=1;bad=False
            for mode in ('native','synchronized'):
                for a,b,distance in row[mode+'_contacts']:
                    bad |= prohibited_contact(a,b,distance,dynamic,protected,geo['rigid_groups'])
                point=row[mode+'_root'];start=previous.get(mode,point)
                for box in geo['boxes']:
                    bad |= segment_box(start,point,[x-rho for x in box['lower']],[x+rho for x in box['upper']])
                previous[mode]=point
            safe &= not bad
            if safe!=row['safe_history']:raise RuntimeError('Persisted safety history mismatch')
            if count==1 and (row['phase']!='initial' or safe!=geo['initial_safe'] or row['native_root']!=geo['initial_root']):
                raise RuntimeError('Initial certificate premises disagree with recorded safety')
    if count!=summary['samples'] or safe!=summary['safe_history']:raise RuntimeError('Incomplete safety log')
    if summary['label']=='conditionally_infeasible' and not (summary['task_scope']=='full' and recomputed['label']=='infeasible'):
        raise RuntimeError('Certificate does not apply to this task')
    if summary['label']=='feasible' and not (summary['task_success'] and safe and summary['actions']>0):
        raise RuntimeError('Missing nontrivial safe-success witness')
    write(folder/'VERIFIED.json',dict(passed=True,samples=count,label=summary['label'],
          training_eligible=False,independent_scan='Recomputed actual-box certificate and all recorded contact/swept-root predicates'))


def execute(root,variant):
    import numpy as np
    import mujoco
    from scipy.spatial.transform import Rotation
    folder=root/variant;folder.mkdir(exist_ok=False)
    physical_hash=configure(folder)
    from robosuite.environments.base import MujocoEnv
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    proposal=read(root/'FIXTURE_PROPOSAL.json');source=proposal['source'];directory=Path(source['root'])
    for name,key in [('RESULT.json','result_sha256'),('steps.jsonl','tape_sha256'),('settled_state.npz','settled_sha256')]:
        if sha(directory/name)!=source[key]:raise RuntimeError('Source witness changed')
    upstream=Path(os.environ['CB_UPSTREAM'])/'safelibero/libero/libero/envs'
    for name in ('objects/site_object.py','object_states/base_object_states.py','predicates/base_predicates.py'):
        if sha(upstream/name)!=SOURCE_HASHES[name]:raise RuntimeError('Native goal predicate source changed')
    boxes=proposal['opened'] if variant=='open_full' else proposal['closed']
    original_init=MujocoEnv._initialize_sim
    def patched(self,xml_string=None):return original_init(self,add_boxes(xml_string or self.model.get_xml(),boxes))
    MujocoEnv._initialize_sim=patched
    scene=source['scene'];suite=benchmark.get_benchmark_dict()[scene['suite']](safety_level=scene['level']);task=suite.get_task(scene['task'])
    random.seed(7);np.random.seed(7);env=None;original_step=None;stream=None
    try:
        env=ControlEnv(bddl_file_name=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file,
            use_camera_obs=False,has_offscreen_renderer=False,camera_depths=False,camera_heights=1024,camera_widths=1024)
        env.seed(7);env.reset();obs=env.set_init_state(suite.get_task_init_states(scene['task'])[scene['episode']])
        m,d=env.sim.model,env.sim.data;marker=m.body_name2id('eef_marker')
        def update_marker(observation):
            q=observation['robot0_eef_quat'];m.body_pos[marker]=observation['robot0_eef_pos']+Rotation.from_quat(q).as_matrix()@np.array([0,0,-.08])
            m.body_quat[marker]=q[[3,0,1,2]]
        update_marker(obs)
        for _ in range(20):obs,_,_,_=env.step([0.]*6+[-1.])
        expected=read(directory/'RESULT.json')['settled_physics_sha256']
        if physical_hash(env)!=expected:raise RuntimeError('Modified initial physics differs from canonical source; no label')
        write(folder/'INITIAL_EQUALITY.json',dict(passed=True,source_sha256=expected,actual_sha256=physical_hash(env)))
        (folder/'model.xml').write_text(m.get_xml())
        np.savez_compressed(folder/'initial_physics.npz',**{k:np.asarray(getattr(d,k)).copy() for k in ARRAYS},
            time=np.asarray([d.time]),body_pos=np.asarray(m.body_pos).copy(),body_quat=np.asarray(m.body_quat).copy())
        raw_goals=[[str(x).lower() if i==0 else x for i,x in enumerate(g)] for g in env.env.parsed_problem['goal_state']]
        if raw_goals!=[MUG_GOAL,['on',TARGET,GOAL]]:raise RuntimeError('Native conjunction changed')
        task_scope='mug_only' if variant=='closed_mug' else 'full'
        def success():return bool(env.env._eval_predicate(MUG_GOAL)) if task_scope=='mug_only' else bool(env.check_success())
        if success():raise RuntimeError('Already solved initial task is excluded')
        facts=initial_geometry(env);target=next(o for o in facts['goal_objects'] if o['name']==TARGET)
        radius=target['box_only_material_radius'];rho=max(0.,radius/math.sqrt(3)-1e-6)
        body=m.body_name2id(m.body_id2name(int(env.env.obj_body_id[TARGET])))
        protected={m.geom_name2id(g['name']) for g in read(directory/'geometry_details.json')['protected']}
        actual=[];static=True
        for box in boxes:
            g=m.geom_name2id(box['name']);p=np.asarray(d.geom_xpos[g]);size=np.asarray(m.geom_size[g])
            static &= bool(int(m.geom_bodyid[g])==0 and int(m.geom_type[g])==6 and
                np.array_equal(np.asarray(d.geom_xmat[g]).reshape(3,3),np.eye(3)) and m.geom_contype[g] and m.geom_conaffinity[g])
            actual.append(dict(name=box['name'],id=g,lower=(p-size).tolist(),upper=(p+size).tolist()));protected.add(g)
        dynamic=set()
        for g in range(m.ngeom):
            b=int(m.geom_bodyid[g]);moving=False
            while b:
                moving |= int(m.body_dofnum[b])>0;b=int(m.body_parentid[b])
            if moving:dynamic.add(g)
        rigid_groups={str(g):int(m.body_weldid[int(m.geom_bodyid[g])]) for g in range(m.ngeom)}
        site=m.site_name2id(GOAL);b=int(m.site_bodyid[site]);fixed=True
        while b:
            fixed &= int(m.body_dofnum[b])==0;b=int(m.body_parentid[b])
        site_info=proposal['goal_site']
        if not (fixed and np.array_equal(d.site_xpos[site],np.asarray(site_info['position'])) and
                np.array_equal(d.site_xmat[site].reshape(3,3),np.asarray(site_info['rotation'])) and
                np.array_equal(env.env.object_sites_dict[GOAL].size,np.asarray(site_info['size']))):
            raise RuntimeError('Native fixed goal geometry differs from audited proposal')
        size=np.asarray(env.env.object_sites_dict[GOAL].size)
        corners=np.asarray(list(itertools.product([-size[0],size[0]],[-size[1],size[1]],
                                                  [size[2]-.005,size[2]+.10])))
        goal_points=corners@np.asarray(d.site_xmat[site]).reshape(3,3)+np.asarray(d.site_xpos[site])
        actual_goal_bounds=[goal_points.min(0).tolist(),goal_points.max(0).tolist()]
        if actual_goal_bounds!=site_info['necessary_goal_bounds']:raise RuntimeError('Necessary fixed goal bounds changed')
        shadow=mujoco.MjData(m._model);previous={};safe=True;samples=0;step=0;substep=0
        stream=gzip.open(folder/'SAMPLES.jsonl.gz','wt')
        def sample(phase):
            nonlocal safe,samples
            before=physical_hash(env)
            for key in ARRAYS:getattr(shadow,key)[:]=np.asarray(getattr(d,key))
            shadow.time=d.time;mujoco.mj_forward(m._model,shadow)
            row=dict(step=step,substep=substep,phase=phase);bad=False
            for mode,data in [('native',d),('synchronized',shadow)]:
                contacts=[[int(c.geom1),int(c.geom2),float(c.dist)] for c in data.contact[:data.ncon]]
                point=np.asarray(data.xpos[body] if mode=='synchronized' else data.body_xpos[body]).tolist()
                row[mode+'_contacts']=contacts;row[mode+'_root']=point
                bad |= any(prohibited_contact(a,b,dist,dynamic,protected,rigid_groups) for a,b,dist in contacts)
                start=previous.get(mode,point)
                bad |= any(segment_box(start,point,[x-rho for x in box['lower']],[x+rho for x in box['upper']]) for box in actual)
                previous[mode]=point
            safe &= not bad;row['safe_history']=safe;samples+=1
            stream.write(json.dumps(row)+'\n')
            if physical_hash(env)!=before:raise RuntimeError('Shadow safety checker mutated live physics')
        sample('initial');initial_safe=safe
        geometry=dict(boxes=actual,lower=proposal['interior_lower'],upper=proposal['interior_upper'],
            initial_root=np.asarray(d.body_xpos[body]).tolist(),goal_bounds=actual_goal_bounds,
            radius=radius,material_verified=radius>0,material_evidence=target,static=static,
            initial_safe=initial_safe,dynamic=sorted(dynamic),protected=sorted(protected),rigid_groups=rigid_groups,
            goal_site=site_info,goal_predicate_source_hashes={k:SOURCE_HASHES[k] for k in SOURCE_HASHES if k in
                ('objects/site_object.py','object_states/base_object_states.py','predicates/base_predicates.py')})
        write(folder/'EXECUTED_GEOMETRY.json',geometry)
        certificate=certify(actual,geometry['lower'],geometry['upper'],geometry['initial_root'],geometry['goal_bounds'],
            radius,initial_safe,static,radius>0,True)
        write(folder/'PUDDING_CERTIFICATE.json',certificate)
        actions=[];task_success=False
        if initial_safe and not(task_scope=='full' and certificate['label']=='infeasible'):
            original_step=env.sim.step
            def monitored(*args,**kwargs):
                nonlocal substep
                value=original_step(*args,**kwargs);substep+=1;sample('integration');return value
            env.sim.step=monitored
            tape=[json.loads(line) for line in (directory/'steps.jsonl').read_text().splitlines()]
            assert_trace(tape,source['actions']);low,high=map(np.asarray,env.env.action_spec)
            for item in tape:
                step=item['step'];substep=0;action=np.clip(item['applied'],low,high)
                obs,_,_,_=env.step(action.tolist());sample('endpoint');task_success=success()
                actions.append(dict(step=step,action=action.tolist(),task_success=task_success,safe_history=safe,
                    source_physics_equal=physical_hash(env)==item['physics_after_sha256'],physics_sha256=physical_hash(env)))
                update_marker(obs)
                if task_success or not safe:break
        label='feasible' if task_success and safe and actions else ('conditionally_infeasible' if task_scope=='full' and certificate['label']=='infeasible' else 'unknown')
        if label=='feasible' and task_scope=='full' and certificate['label']=='infeasible':raise RuntimeError('Contradictory proof and witness')
        stream.close();stream=None
        write(folder/'ACTIONS.json',actions)
        summary=dict(variant=variant,task_scope=task_scope,task_success=task_success,label=label,
            initial_safe=initial_safe,safe_history=safe,actions=len(actions),samples=samples,
            legal_native_actions=all(len(x['action'])==7 and all(-1<=v<=1 for v in x['action']) for x in actions),
            source_physics_equalities=sum(x['source_physics_equal'] for x in actions),
            source=source,initial_physics_sha256=expected,goal_region_observability='pending',
            contract='FD1000-fixture-v1: strict native and synchronized protected contacts, plus separate native-cache and synchronized pudding-root segment guards against cube-expanded static walls; full task or explicit mug-only subtask',
            unrestricted_physical_impossibility_claim=False,scientific_training_allowed=False,accepted_dataset_rows=0,
            appearance_matching_passed=False,api_calls=0,code_commit=os.environ['CB_CODE_COMMIT'],slurm_job=os.environ['SLURM_JOB_ID'])
        write(folder/'SUMMARY.json',summary);check_saved(folder)
        write(folder/'HASHES.json',{str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()})
        return summary
    finally:
        if stream is not None:stream.close()
        if env is not None:
            if original_step is not None:delattr(env.sim,'step')
            env.close()
        MujocoEnv._initialize_sim=original_init


def main(root,variant=None):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Compute allocation required')
    if variant is not None:
        result=execute(root,variant);gc.collect();print(json.dumps(result),flush=True);return
    rows=[]
    for name in VARIANTS:
        with (root/(name+'.log')).open('w') as stream:
            result=subprocess.run([sys.executable,'-u','-m','verifier_dataset.fixture_smoke',str(root),'--variant',name],stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Fixture subprocess failed: '+name+' exit '+str(result.returncode))
        check_saved(root/name);rows.append(read(root/name/'SUMMARY.json'))
    complete=dict(passed=True,rows=rows,accepted_dataset_rows=0,scientific_training_allowed=False,
        same_initial_physics=len({r['initial_physics_sha256'] for r in rows})==1,
        same_image_task_flip_candidate=rows[0]['label']=='conditionally_infeasible' and rows[2]['label']=='feasible',
        image_equality_verified=False,appearance_matching_passed=False,api_calls=0,slurm_job=os.environ['SLURM_JOB_ID'])
    write(root/'FIXTURE_COMPLETE.json',complete);print(json.dumps(complete),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--variant',choices=VARIANTS)
    a=p.parse_args();main(a.root,a.variant)
