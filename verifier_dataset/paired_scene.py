"""Bounded physical prototype for same-image, different-target feasibility pairs.

Existing action suffixes supply positive witnesses; an unrelated enclosed object
supplies a task-applicable conditional negative. Every construction remains a
development candidate until rendering, grounding and grouped audits pass.
"""
import argparse
import gc
import gzip
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from .inventory import read,sha,write
from .replay import ARRAYS,configure,clean,assert_trace
from .fixture_smoke import add_boxes,restore_canonical_initial
from .contact_contract import descendants,prohibited
from .paired_geometry import CONTRACT,proposals,material_sphere,point_certificate,inside,sphere_radius
from .paired_goal import definition as goal_definition,snapshot as goal_snapshot,evaluate as goal_evaluate
from experiments.red_enclosure.contract import segment_box
from experiments.feasibility_contract.contract import SOURCE_HASHES


def make_env(case,folder,proposal=None):
    import numpy as np
    from scipy.spatial.transform import Rotation
    physical_hash=configure(folder)
    from robosuite.environments.base import MujocoEnv
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    row=read(case/'INPUT.json');witness=Path(row['evidence']['witness_root'])
    summary=read(witness/'SUMMARY.json');source=summary['source'];scene=source['scene']
    if sha(witness/'SUMMARY.json')!=row['evidence']['witness_summary_sha256']:
        raise RuntimeError('Source witness changed')
    upstream=Path(os.environ['CB_UPSTREAM'])/'safelibero/libero/libero/envs'
    for name in ('objects/site_object.py','object_states/base_object_states.py','predicates/base_predicates.py'):
        if sha(upstream/name)!=SOURCE_HASHES[name]:raise RuntimeError('Native goal predicate source changed')
    suite=benchmark.get_benchmark_dict()[scene['suite']](safety_level=scene['level']);task=suite.get_task(scene['task'])
    original=MujocoEnv._initialize_sim
    if proposal is not None:
        def patched(self,xml_string=None):return original(self,add_boxes(xml_string or self.model.get_xml(),proposal['boxes']))
        MujocoEnv._initialize_sim=patched
    env=None
    try:
        random.seed(7);np.random.seed(7)
        env=ControlEnv(bddl_file_name=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file,
            use_camera_obs=False,has_offscreen_renderer=False,camera_depths=False,camera_heights=1024,camera_widths=1024)
        env.seed(7);env.reset();obs=env.set_init_state(suite.get_task_init_states(scene['task'])[scene['episode']])
        m=env.sim.model;marker=m.body_name2id('eef_marker')
        def update_marker(observation):
            q=observation['robot0_eef_quat'];m.body_pos[marker]=observation['robot0_eef_pos']+Rotation.from_quat(q).as_matrix()@np.array([0,0,-.08])
            m.body_quat[marker]=q[[3,0,1,2]]
        update_marker(obs)
        for _ in range(20):obs,_,_,_=env.step([0.]*6+[-1.])
        restore_canonical_initial(env,case,folder)
        context=read(case/'CANONICAL_CONTEXT.json')
        if physical_hash(env)!=context['physical_sha256']:raise RuntimeError('Constructed canonical physical state differs')
        return env,physical_hash,update_marker,source,context
    except BaseException:
        if env is not None:env.close()
        raise
    finally:MujocoEnv._initialize_sim=original


def prepare(case):
    folder=case/'probe';folder.mkdir();env=None
    try:
        env,physical_hash,_,source,context=make_env(case,folder)
        goals=[[str(g[0]).lower()]+list(g[1:]) for g in env.env.parsed_problem['goal_state']]
        eligible=[g for g in goals if len(g)==3 and g[0] in ('on','in') and not env.env._eval_predicate(g)]
        if not eligible:
            write(case/'PROPOSALS.json',dict(rows=[],rejected=[dict(reason='No unsolved supported native atomic goal')],labels_created=False));return
        goal=eligible[0];obstacle=read(Path(source['root'])/'geometry.json')['obstacle']
        try:material_sphere(env,goal[1])
        except RuntimeError as error:
            write(case/'PROPOSALS.json',dict(rows=[],rejected=[dict(reason=str(error),object=goal[1])],labels_created=False));return
        before=physical_hash(env);choices,rejected=proposals(env,goal[1],{obstacle},goal)
        if physical_hash(env)!=before:raise RuntimeError('Read-only paired design mutated physics')
        write(case/'PROPOSALS.json',dict(rows=choices,rejected=rejected,source=source,
            checkpoint=context['step'],contract=CONTRACT,labels_created=False,maximum_candidates=3))
    finally:
        if env is not None:env.close()


def scan_saved(folder):
    """Recompute task-applicable proof and both safety histories from raw samples."""
    geo=read(folder/'GEOMETRY.json');summary=read(folder/'SUMMARY.json')
    if geo['contract']!=CONTRACT:raise RuntimeError('Wrong paired task contract')
    expected_negative=[summary['primary_goal'][0],geo['proposal']['enclosed'],summary['primary_goal'][2]]
    if summary['enclosed_goal']!=expected_negative or summary['fixed_destination']!=geo['destination']['fixed_root_region']:
        raise RuntimeError('Certificate belongs to another target or destination')
    for role,material in geo['materials'].items():
        if material['object']!=geo['proposal'][role]:raise RuntimeError('Material-point task identity mismatch')
        if material['radius']!=sphere_radius(material['geom_type'],material['primitive_size']):
            raise RuntimeError('Unverified primitive material sphere')
    static=all(b['world_body']==0 and b['geom_type']==6 and b['contype'] and b['conaffinity'] and
               b['rotation']==[1.,0.,0.,0.,1.,0.,0.,0.,1.] for b in geo['boxes'])
    if static!=geo['static']:raise RuntimeError('Unverified static wall geometry')
    cert=point_certificate(geo['boxes'],geo['lower'],geo['upper'],geo['materials']['enclosed'],
        geo['destination']['fixed_root_region'],geo['initial_safe']['enclosed'] and geo['initial_fixture_valid'],geo['static'])
    if cert!=read(folder/'CERTIFICATE.json'):raise RuntimeError('Saved material-point separator disagrees')
    protected=set(geo['protected']);actors={r:set(v) for r,v in geo['actors'].items()}
    safe={r:True for r in actors};previous={};count=0;last=None;integrations=0;endpoints=[];last_pair=(0,0)
    with gzip.open(folder/'SAMPLES.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line);count+=1;last=row
            for mode in ('native','synchronized'):
                value=goal_evaluate(geo['goal_definition'],row[mode+'_goal_state'],row[mode+'_contacts'])
                if value!=row[mode+'_goal_success'] or row[mode+'_goal_state']['target_position']!=row[mode+'_roots']['primary']:
                    raise RuntimeError('Saved native goal formula or object identity mismatch')
            if row['primary_native_goal']!=row['native_goal_success']:
                raise RuntimeError('Native predicate and independent formula disagree')
            pair=(row['step'],row['substep'])
            if row['phase']=='initial':
                if count!=1 or pair!=(0,0):raise RuntimeError('Invalid initial sample order')
            elif row['phase']=='integration':
                if not ((pair[0]==last_pair[0] and pair[1]==last_pair[1]+1) or
                        (pair[0]==last_pair[0]+1 and pair[1]==1)):
                    raise RuntimeError('Noncontiguous paired integration samples')
                integrations+=1;last_pair=pair
            elif row['phase']=='endpoint':
                if pair!=last_pair:raise RuntimeError('Endpoint does not follow its integration samples')
                endpoints.append(row['step'])
            else:raise RuntimeError('Unexpected paired sample phase')
            for role in actors:
                rho=max(0.,geo['materials'][role]['radius']/math.sqrt(3)-1e-6)
                for mode in ('native','synchronized'):
                    bad=any(prohibited((a,b),protected,actors[role]) and dist<=0 for a,b,dist in row[mode+'_contacts'])
                    point=row[mode+'_points'][role];old=previous.get((role,mode),point)
                    bad |= any(segment_box(old,point,[x-rho for x in box['lower']],
                                           [x+rho for x in box['upper']]) for box in geo['boxes'])
                    previous[(role,mode)]=point;safe[role] &= not bad
                if safe[role]!=row['safe_history'][role]:raise RuntimeError('Saved task-specific safety mismatch')
            if count==1 and (row['phase']!='initial' or safe!=geo['initial_safe']):
                raise RuntimeError('Unverified initial safety')
            if count==1:
                walls=set(geo['wall_geoms'])
                overlaps=[c for mode in ('native','synchronized') for c in row[mode+'_contacts']
                          if c[2]<=0 and ((c[0] in walls)!=(c[1] in walls))]
                if (not overlaps)!=geo['initial_fixture_valid']:
                    raise RuntimeError('Unverified constructed-wall initial validity')
    actions=read(folder/'ACTIONS.json')
    if count!=summary['samples'] or len(actions)!=summary['actions'] or safe!=summary['safe_history']:
        raise RuntimeError('Incomplete pair witness evidence')
    if integrations!=summary['integration_steps'] or endpoints!=list(range(1,len(actions)+1)) or count!=1+integrations+len(actions):
        raise RuntimeError('Incomplete integration/endpoint coverage')
    if any(len(x['action'])!=7 or not all(math.isfinite(v) and -1<=v<=1 for v in x['action']) for x in actions):
        raise RuntimeError('Illegal action in paired witness')
    witness=bool(actions and geo['initial_fixture_valid'] and safe['primary'] and last['primary_native_goal'] and last['synchronized_goal_success'] and
        inside(last['native_roots']['primary'],geo['destination']['fixed_root_region']) and
        inside(last['synchronized_roots']['primary'],geo['destination']['fixed_root_region']))
    if witness!=summary['positive_witness'] or summary['matched_pair']!=(witness and cert['label']=='conditionally_infeasible'):
        raise RuntimeError('Saved pair verdict lacks matching evidence')
    write(folder/'VERIFIED.json',dict(passed=True,samples=count,positive_witness=witness,
        conditional_negative=cert['label']=='conditionally_infeasible',matched_pair=summary['matched_pair'],
        verifier='Independent scan of persisted raw contacts, material-center segments and geometry certificate',
        training_allowed=False))


def execute(case,index):
    import numpy as np
    import mujoco
    folder=case/('option_%02d'%index);folder.mkdir();env=None;original_step=None;stream=None
    proposal=read(case/'PROPOSALS.json')['rows'][index]
    try:
        env,physical_hash,update_marker,source,context=make_env(case,folder,proposal)
        m,d=env.sim.model,env.sim.data
        native_geometry=read(Path(source['root'])/'geometry_details.json')
        protected={m.geom_name2id(g['name']) for g in native_geometry['protected']}
        boxes=[];static=True;wall_geoms=set()
        for box in proposal['boxes']:
            g=m.geom_name2id(box['name']);p=np.asarray(d.geom_xpos[g]);half=np.asarray(m.geom_size[g])
            static &= bool(int(m.geom_bodyid[g])==0 and int(m.geom_type[g])==6 and m.geom_contype[g] and m.geom_conaffinity[g] and
                           np.array_equal(np.asarray(d.geom_xmat[g]).reshape(3,3),np.eye(3)))
            boxes.append(dict(name=box['name'],lower=(p-half).tolist(),upper=(p+half).tolist(),
                world_body=int(m.geom_bodyid[g]),geom_type=int(m.geom_type[g]),contype=int(m.geom_contype[g]),
                conaffinity=int(m.geom_conaffinity[g]),rotation=np.asarray(d.geom_xmat[g]).reshape(-1).tolist()))
            protected.add(g)
            wall_geoms.add(g)
        names={r:proposal[r] for r in ('primary','enclosed')}
        roots={r:int(env.env.obj_body_id[name]) for r,name in names.items()}
        robot=descendants(m,{m.body_name2id(env.robots[0].robot_model.root_body)})
        actors={r:robot|descendants(m,{body}) for r,body in roots.items()}
        if any(ids&protected for ids in actors.values()):raise RuntimeError('Task actor overlaps protected geometry')
        materials={r:material_sphere(env,name) for r,name in names.items()}
        centers={r:m.geom_name2id(material['geom_name']) for r,material in materials.items()}
        shadow=mujoco.MjData(m._model);step=0;substep=0;samples=0;integrations=0;previous={};safe={r:True for r in actors}
        goal=proposal['primary_goal'];region=proposal['destination']['fixed_root_region'];goal_spec=goal_definition(env,goal)
        stream=gzip.open(folder/'SAMPLES.jsonl.gz','wt')
        def sample(phase):
            nonlocal samples
            before=physical_hash(env)
            for key in ARRAYS:getattr(shadow,key)[:]=np.asarray(getattr(d,key))
            shadow.time=d.time;mujoco.mj_forward(m._model,shadow)
            row=dict(step=step,substep=substep,phase=phase,primary_native_goal=bool(env.env._eval_predicate(goal)))
            for mode,data in [('native',d),('synchronized',shadow)]:
                contacts=[[int(c.geom1),int(c.geom2),float(c.dist)] for c in data.contact[:data.ncon]]
                points={r:np.asarray(data.geom_xpos[g]).tolist() for r,g in centers.items()}
                positions={r:np.asarray(data.body_xpos[b] if mode=='native' else data.xpos[b]).tolist() for r,b in roots.items()}
                row[mode+'_contacts']=contacts;row[mode+'_points']=points;row[mode+'_roots']=positions
                state=goal_snapshot(data,goal_spec,synchronized=mode=='synchronized')
                row[mode+'_goal_state']=state;row[mode+'_goal_success']=goal_evaluate(goal_spec,state,contacts)
                for role in actors:
                    rho=max(0.,materials[role]['radius']/math.sqrt(3)-1e-6)
                    bad=any(prohibited((a,b),protected,actors[role]) and dist<=0 for a,b,dist in contacts)
                    point=points[role];old=previous.get((role,mode),point)
                    bad |= any(segment_box(old,point,[x-rho for x in box['lower']],[x+rho for x in box['upper']]) for box in boxes)
                    previous[(role,mode)]=point;safe[role] &= not bad
            row['safe_history']=dict(safe);samples+=1;stream.write(json.dumps(row)+'\n')
            if row['native_goal_success']!=row['primary_native_goal']:
                raise RuntimeError('Independent goal formula disagrees with pinned native predicate')
            if physical_hash(env)!=before:raise RuntimeError('Read-only paired checker mutated physics')
            return row
        initial=sample('initial');initial_safe=dict(safe)
        initial_wall_contacts=[c for mode in ('native','synchronized') for c in initial[mode+'_contacts']
            if c[2]<=0 and ((c[0] in wall_geoms)!=(c[1] in wall_geoms))]
        initial_fixture_valid=not initial_wall_contacts
        for role in materials:materials[role]['initial_world_center']=initial['native_points'][role]
        geo=dict(contract=CONTRACT,proposal=proposal,boxes=boxes,lower=proposal['lower'],upper=proposal['upper'],
            materials=materials,destination=proposal['destination'],protected=sorted(protected),
            actors={r:sorted(gs) for r,gs in actors.items()},initial_safe=initial_safe,static=static,
            initial_fixture_valid=initial_fixture_valid,initial_wall_contacts=initial_wall_contacts,wall_geoms=sorted(wall_geoms),
            native_support_contacts_allowed=True,goal_definition=goal_spec,source=source,checkpoint=context['step'])
        write(folder/'GEOMETRY.json',geo)
        cert=point_certificate(boxes,geo['lower'],geo['upper'],materials['enclosed'],region,
            initial_safe['enclosed'] and initial_fixture_valid,static)
        write(folder/'CERTIFICATE.json',cert)
        initially_done=bool(env.env._eval_predicate(goal))
        actions=[];last=initial
        if initial_fixture_valid and all(initial_safe.values()) and cert['label']=='conditionally_infeasible' and not initially_done:
            original_step=env.sim.step
            def monitored(*args,**kwargs):
                nonlocal substep,integrations
                value=original_step(*args,**kwargs);substep+=1;integrations+=1;sample('integration');return value
            env.sim.step=monitored
            source_path=Path(source['root'])/'steps.jsonl'
            if sha(source_path)!=source['tape_sha256']:raise RuntimeError('Source action tape changed')
            tape=[json.loads(line) for line in source_path.read_text().splitlines()];assert_trace(tape,source['actions'])
            low,high=map(np.asarray,env.env.action_spec)
            # One fixed native settling tail may supply a completed legal
            # witness after the inherited commands. It uses the benchmark's
            # existing zero-motion/open-gripper action, never an API or policy
            # search, and stays within the stated remaining action budget.
            commands=[dict(item,command_kind='source_replay') for item in tape[context['step']:]]
            commands += [dict(step=None,applied=[0.]*6+[-1.],physics_after_sha256=None,
                              command_kind='native_settling_tail') for _ in range(20)]
            for item in commands:
                if step>=context['remaining_horizon']:break
                step+=1;substep=0;action=np.clip(item['applied'],low,high)
                obs,_,_,_=env.step(action.tolist());last=sample('endpoint')
                goal_success=bool(last['primary_native_goal'] and last['synchronized_goal_success'] and inside(last['native_roots']['primary'],region)
                                  and inside(last['synchronized_roots']['primary'],region))
                actions.append(dict(step=step,source_step=item['step'],command_kind=item['command_kind'],action=action.tolist(),
                    source_physics_equal=physical_hash(env)==item['physics_after_sha256'],
                    actual_physics_sha256=physical_hash(env),goal_success=goal_success,safe_history=dict(safe)))
                update_marker(obs)
                if goal_success or not safe['primary']:break
        stream.close();stream=None
        positive=bool(actions and safe['primary'] and actions[-1]['goal_success'])
        summary=dict(contract=CONTRACT,actions=len(actions),samples=samples,integration_steps=integrations,initial_safe=initial_safe,
            safe_history=safe,positive_witness=positive,negative_certificate=cert['label'],
            matched_pair=positive and cert['label']=='conditionally_infeasible',
            primary_goal=goal,enclosed_goal=proposal['enclosed_goal'],fixed_destination=region,
            source=source,checkpoint=context['step'],remaining_horizon=context['remaining_horizon'],
            initial_physics_sha256=context['physical_sha256'],initially_solved=initially_done,
            source_equality_diagnostic=sum(x['source_physics_equal'] for x in actions),
            replayed_source_actions=sum(x['command_kind']=='source_replay' for x in actions),
            native_settling_tail_actions=sum(x['command_kind']=='native_settling_tail' for x in actions),
            maximum_settling_tail_actions=20,initial_fixture_valid=initial_fixture_valid,
            task_scope='Derived atomic native predicate AND explicit fixed green root region; marker-conditioned task contrast',
            synchronized_native_goal_required=True,independent_native_goal_formula_checked=True,
            labels_created_for_training=False,training_allowed=False,api_calls=0,
            code_commit=os.environ['CB_CODE_COMMIT'],slurm_job=os.environ['SLURM_JOB_ID'])
        write(folder/'ACTIONS.json',actions);write(folder/'SUMMARY.json',summary)
        (folder/'model.xml').write_text(m.get_xml())
        scan_saved(folder)
        write(folder/'HASHES.json',{p.name:sha(p) for p in folder.iterdir() if p.is_file()})
    finally:
        if stream is not None:stream.close()
        if env is not None:
            if original_step is not None:delattr(env.sim,'step')
            env.close()


def main(root,mode=None,option=None):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Compute allocation required')
    if mode=='prepare':prepare(root);gc.collect();return
    if mode=='execute':execute(root,option);gc.collect();return
    rows=[]
    for index,row in enumerate(read(root/'CASES.json')):
        case=root/('case_%03d'%index);case.mkdir();write(case/'INPUT.json',row)
        witness=Path(row['evidence']['witness_root']);checkpoint=witness/('checkpoint_%03d'%row['evidence']['checkpoint_step'])
        if sha(checkpoint/'context.json')!=row['evidence']['checkpoint_context_sha256']:
            raise RuntimeError('Canonical checkpoint context changed')
        for source,dest in [('context.json','CANONICAL_CONTEXT.json'),('physics.npz','CANONICAL_PHYSICS.npz')]:
            shutil.copyfile(checkpoint/source,case/dest)
        with (case/'prepare.log').open('w') as log:
            subprocess.run([sys.executable,'-u','-m','verifier_dataset.paired_scene',str(case),'--mode','prepare'],stdout=log,stderr=subprocess.STDOUT,check=True)
        design=read(case/'PROPOSALS.json');outcomes=[]
        for option_index in range(len(design['rows'])):
            with (case/('option_%02d.log'%option_index)).open('w') as log:
                subprocess.run([sys.executable,'-u','-m','verifier_dataset.paired_scene',str(case),
                    '--mode','execute','--option',str(option_index)],stdout=log,stderr=subprocess.STDOUT,check=True)
            folder=case/('option_%02d'%option_index);scan_saved(folder)
            result=read(folder/'SUMMARY.json');outcomes.append(result)
            if result['matched_pair']:break
        rows.append(dict(case=str(case),source_id=row['metadata']['source_id'],family=row['metadata']['family'],
            candidates=len(design['rows']),outcomes=outcomes,matched_pair=any(x['matched_pair'] for x in outcomes)))
        write(root/'PROGRESS.json',dict(completed=len(rows),expected=len(read(root/'CASES.json')),
            matched_pairs=sum(r['matched_pair'] for r in rows)))
    result=dict(passed=True,rows=rows,matched_pairs=sum(r['matched_pair'] for r in rows),
        source_states=len({r['source_id'] for r in rows}),accepted_dataset_rows=0,
        student_images_created=False,reasoning_targets_created=False,training_allowed=False,
        api_calls=0,slurm_job=os.environ['SLURM_JOB_ID'])
    write(root/'PAIRED_COMPLETE.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--mode',choices=('prepare','execute'));p.add_argument('--option',type=int)
    a=p.parse_args();main(a.root,a.mode,a.option)
