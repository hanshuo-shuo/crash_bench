"""CPU action-tape replay with source equality and integration-contact evidence."""
import argparse
import faulthandler
import gc
import gzip
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from .inventory import read, sha, write

ARRAYS = ['qpos', 'qvel', 'act', 'ctrl', 'qacc', 'qacc_warmstart', 'qfrc_applied',
          'xfrc_applied', 'mocap_pos', 'mocap_quat', 'userdata']


def checkpoint_steps(actions, maximum=10):
    # Retain meaningful remaining work; never supervise an already-solved endpoint.
    eligible = list(range(0, max(0, actions-10), 5))
    if len(eligible) <= maximum:
        return eligible
    return [eligible[i*(len(eligible)-1)//(maximum-1)] for i in range(maximum)]


def assert_trace(steps, count):
    if len(steps) != count or [r['step'] for r in steps] != list(range(1, count+1)):
        raise RuntimeError('Noncontiguous or incomplete original action tape')
    for row in steps:
        action = row['applied']
        if len(action) != 7 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in action):
            raise RuntimeError('Invalid action tape')


def configure(root,rendering=False):
    upstream = Path(os.environ['CB_UPSTREAM'])
    repo = Path(__file__).resolve().parents[1]
    cfg = root/'libero_config'; cfg.mkdir(exist_ok=False)
    benchmark = upstream/'safelibero/libero/libero'
    write(cfg/'config.yaml', dict(benchmark_root=str(benchmark), bddl_files=str(benchmark/'bddl_files'),
        init_states=str(benchmark/'init_files'), assets=str(benchmark/'assets'), datasets=str(upstream/'safelibero/libero/datasets')))
    os.environ['LIBERO_CONFIG_PATH'] = str(cfg)
    sys.path[:0] = [str(upstream/'safelibero'), str(repo/'experiments/feasibility')]
    from robosuite.environments.base import MujocoEnv
    import robosuite.utils.binding_utils as bindings
    write(root/'BACKEND.json',dict(MUJOCO_GL=os.environ.get('MUJOCO_GL'),
        PYOPENGL_PLATFORM=os.environ.get('PYOPENGL_PLATFORM'),binding_backend=bindings._MUJOCO_GL,
        rendering_enabled=rendering))
    if not rendering:
        from reset_forward import replay_renderer_reset_forward
        replay_renderer_reset_forward(MujocoEnv)
    # Match the exact historical hash implementation, not a new relaxed tolerance.
    import importlib.util
    spec = importlib.util.spec_from_file_location('fd_uncertainty_geometry', repo/'analysis/uncertainty/geometry.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.physical_hash


def clean(value):
    import numpy as np
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean(v) for v in value]
    if isinstance(value, (str,int,float,bool,type(None))): return value
    return {'unsupported_type':type(value).__name__}


def replay(row, folder, physical_hash, canonicalize=False, design_fixture=False):
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    from scipy.spatial.transform import Rotation
    folder.mkdir(exist_ok=False)
    source = Path(row['root']); result = read(source/'RESULT.json')
    for name, key in [('RESULT.json','result_sha256'),('steps.jsonl','tape_sha256'),('settled_state.npz','settled_sha256')]:
        if sha(source/name) != row[key]:
            raise RuntimeError('Historical source changed: '+str(source/name))
    scene = row['scene']; suite = benchmark.get_benchmark_dict()[scene['suite']](safety_level=scene['level'])
    task = suite.get_task(scene['task'])
    steps = [json.loads(x) for x in (source/'steps.jsonl').read_text().splitlines()]
    assert_trace(steps, row['actions'])
    random.seed(7); np.random.seed(7)
    env = None; original_step = None; had_step_override = False; previous_step_override = None
    started = time.monotonic(); contacts = []; saved = []
    try:
        env = ControlEnv(bddl_file_name=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file,
                         use_camera_obs=False, has_offscreen_renderer=False, camera_depths=False,
                         camera_heights=1024, camera_widths=1024)
        env.seed(7); env.reset()
        obs = env.set_init_state(suite.get_task_init_states(scene['task'])[scene['episode']])
        model, data = env.sim.model, env.sim.data
        marker = model.body_name2id('eef_marker')

        def update_marker(observation):
            quat = observation['robot0_eef_quat']
            model.body_pos[marker] = observation['robot0_eef_pos'] + Rotation.from_quat(quat).as_matrix() @ np.array([0,0,-.08])
            model.body_quat[marker] = quat[[3,0,1,2]]

        update_marker(obs)
        for _ in range(20):
            obs, _, _, _ = env.step([0.]*6+[-1.])
        with np.load(source/'settled_state.npz', allow_pickle=False) as expected:
            for key in ARRAYS:
                actual = np.asarray(getattr(data,key))
                if not np.array_equal(actual,expected[key]):
                    write(folder/'MISMATCH.json', dict(stage='settled',field=key,
                        max_abs=float(np.max(np.abs(actual-expected[key]))) if actual.size else 0.))
                    raise RuntimeError('Settled physics differs: '+key)
        if physical_hash(env) != result['settled_physics_sha256']:
            raise RuntimeError('Settled physical/model hash differs')
        geometry = read(source/'geometry.json')
        protected = set(geometry['protected_geoms'])
        details = read(source/'geometry_details.json')
        if {g['id'] for g in details['protected']} != protected:
            raise RuntimeError('Protected geometry manifest mismatch')
        for geom in details['protected']:
            if model.geom_id2name(geom['id']) != geom['name']:
                raise RuntimeError('Protected geometry identity changed')
        dynamic = set()
        for geom in range(model.ngeom):
            body = int(model.geom_bodyid[geom]); moving = False
            while body:
                moving |= int(model.body_dofnum[body]) > 0
                body = int(model.body_parentid[body])
            if moving and geom not in protected:
                dynamic.add(geom)
        write(folder/'GEOMETRY_IDENTITIES.json',dict(
            protected=sorted(protected), moving_nonprotected=sorted(dynamic),
            robot=geometry['robot_geoms'],objects_of_interest=clean(env.obj_of_interest),
            geoms=[dict(id=g,name=model.geom_id2name(g),body=model.body_id2name(int(model.geom_bodyid[g])),
                        type=int(model.geom_type[g])) for g in range(model.ngeom)]))
        from .geometry_facts import initial_geometry
        before_geometry=physical_hash(env)
        write(folder/'INITIAL_TASK_GEOMETRY.json',clean(initial_geometry(env)))
        if physical_hash(env)!=before_geometry:raise RuntimeError('Read-only geometry probe mutated physics')
        design=None
        if design_fixture:
            from .fixture_design import Design
            design=Design(env,folder,geometry)
        counter = dict(action=0, substep=0, total=0)
        def sample(phase):
            if design is not None:design.sample(counter['action'],counter['substep'])
            for contact in data.contact[:data.ncon]:
                pair = (int(contact.geom1),int(contact.geom2))
                bad = ((pair[0] in protected and pair[1] in dynamic) or
                       (pair[1] in protected and pair[0] in dynamic))
                if bad and contact.dist <= 0:
                    contacts.append(dict(action=counter['action'],substep=counter['substep'],phase=phase,
                                         geom_pair=list(pair),distance=float(contact.dist)))
        sample('initial')
        had_step_override = 'step' in vars(env.sim)
        previous_step_override = vars(env.sim).get('step')
        original_step = env.sim.step
        def step_and_sample(*args,**kwargs):
            value = original_step(*args,**kwargs)
            counter['substep'] += 1; counter['total'] += 1; sample('integration')
            return value
        env.sim.step = step_and_sample
        wanted = set(checkpoint_steps(len(steps)))
        legal = True; original_legal = True; equality = []; done = False; legal_tape=[]
        low, high = [np.asarray(v) for v in env.env.action_spec]
        for item in steps:
            t = item['step']-1
            if t in wanted:
                checkpoint = folder/('checkpoint_%03d'%t); checkpoint.mkdir()
                np.savez_compressed(checkpoint/'physics.npz', **{k:np.asarray(getattr(data,k)).copy() for k in ARRAYS},
                    sim_state=env.get_sim_state().copy(), time=np.asarray([data.time]),
                    model_body_pos=np.asarray(model.body_pos).copy(), model_body_quat=np.asarray(model.body_quat).copy())
                robot = env.robots[0]
                write(checkpoint/'context.json', clean(dict(step=t, remaining_horizon=(550 if scene['suite']=='safelibero_long' else 300)-t,
                    controller={k:v for k,v in vars(robot.controller).items() if isinstance(v,(np.ndarray,np.generic,str,float,int,bool,list,tuple,type(None)))},
                    gripper_action=robot.gripper.current_action, initial_success=bool(env.check_success()),
                    source_inference=str(source/('infer_%03d.npz'%(t//5+1))),
                    physical_sha256=physical_hash(env), safe_history=not contacts,
                    environment=dict(timestep=env.env.timestep,cur_time=env.env.cur_time,done=env.env.done),
                    restore_verified=False, witness_basis='State encountered on fully rechecked action-tape trajectory; not an arbitrary restored continuation')))
                saved.append(t)
            original_action = np.asarray(item['applied'])
            original_legal &= bool(np.all(original_action>=low) and np.all(original_action<=high))
            action = np.clip(original_action,low,high) if canonicalize else original_action
            legal &= bool(np.all(action>=low) and np.all(action<=high))
            legal_tape.append(dict(step=item['step'],action=action.tolist(),original=original_action.tolist()))
            counter.update(action=item['step'],substep=0)
            obs, _, done, _ = env.step(action.tolist())
            actual_hash = physical_hash(env)
            equal = actual_hash == item['physics_after_sha256']
            equality.append(dict(step=item['step'],equal=equal,sha256=actual_hash))
            if not equal or bool(done) != item['success']:
                write(folder/'MISMATCH.json',dict(stage='action',step=item['step'],expected=item['physics_after_sha256'],
                    actual=actual_hash,expected_success=item['success'],actual_success=bool(done)))
                raise RuntimeError('Per-action replay differs at '+str(item['step']))
            update_marker(obs)
        with gzip.open(folder/'CONTACTS.jsonl.gz','wt') as stream:
            for event in contacts: stream.write(json.dumps(event)+'\n')
        write(folder/'EQUALITY.json',equality)
        write(folder/'ACTION_TAPE.json',legal_tape)
        approved = bool(done and legal and not contacts and all(x['equal'] for x in equality))
        summary = dict(source=row,replay_verified=True,actions=len(steps),legal_actions=legal,success=bool(done),
            original_commands_legal=original_legal,canonicalized_to_native_action_spec=canonicalize,
            canonicalization_note='Clipped to native action_spec; accepted only if every original physical hash is still exactly reproduced' if canonicalize else None,
            protected_contact_events=len(contacts),integration_steps=counter['total'],
            eligible_feasible_witness=approved,checkpoints=saved,accepted_dataset_rows=0,
            safety_contract='No native MuJoCo contact with distance <=0 between the recorded protected obstacle and another moving body at initial state or any executed integration step',
            conditional_digital_evidence=True,continuous_physics_proof=False,arbitrary_restore_verified=False,
            seed=7,slurm_job=os.environ['SLURM_JOB_ID'],code_commit=os.environ['CB_CODE_COMMIT'],
            source_code_commit=row['source_code_commit'],elapsed_seconds=time.monotonic()-started)
        write(folder/'SUMMARY.json',summary)
        if design is not None:design.finish(summary)
        write(folder/'HASHES.json',{str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file()})
        return summary
    finally:
        if env is not None:
            # Remove the environment -> wrapper closure -> bound sim cycle before
            # native MuJoCo destruction. Never accept a shutdown crash as success.
            if original_step is not None:
                if had_step_override:
                    env.sim.step = previous_step_override
                else:
                    # Putting the bound method back in the instance dictionary
                    # creates a sim -> bound method -> sim cycle. Restore the
                    # original class descriptor instead.
                    delattr(env.sim,'step')
            env.close()


def main(root, case_index=None):
    faulthandler.enable()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Compute allocation required')
    rows=read(root/'CASES.json'); results=[]; errors=[]
    canonicalize=read(root/'CAMPAIGN.json').get('canonicalize',False)
    if case_index is not None:
        work=root/('worker_%03d'%case_index);work.mkdir(exist_ok=False)
        physical_hash=configure(work)
        value=replay(rows[case_index],root/('case_%03d'%case_index),physical_hash,canonicalize=canonicalize,
                     design_fixture=read(root/'CAMPAIGN.json')['stage']=='design-fixture')
        gc.collect()
        print(json.dumps({k:v for k,v in value.items() if k!='source'}),flush=True)
        return
    for i,row in enumerate(rows):
        folder=root/('case_%03d'%i)
        try:
            # MuJoCo/native allocations must not survive into the next scene.
            # A successful JSON followed by a native crash is not accepted.
            with (root/('worker_%03d.log'%i)).open('w') as log:
                process=subprocess.run([sys.executable,'-u','-m','verifier_dataset.replay',str(root),
                    '--case-index',str(i)],stdout=log,stderr=subprocess.STDOUT)
            if process.returncode != 0:
                raise RuntimeError('Isolated replay process exit '+str(process.returncode))
            value=read(folder/'SUMMARY.json'); results.append(value)
            print(json.dumps({k:v for k,v in value.items() if k!='source'}),flush=True)
        except Exception as error:
            folder.mkdir(exist_ok=True)
            failure=dict(case=i,source=row['root'],error=type(error).__name__,reason=str(error))
            errors.append(failure);write(folder/'FAILED.json',failure); print(json.dumps(failure),flush=True)
        write(root/'PROGRESS.json',dict(completed=len(results),failed=len(errors),expected=len(rows)))
    summary=dict(passed=not errors and len(results)==len(rows),completed=len(results),errors=errors,
                 eligible_witnesses=sum(r['eligible_feasible_witness'] for r in results),
                 eligible_checkpoints=sum(len(r['checkpoints']) for r in results if r['eligible_feasible_witness']),
                 accepted_dataset_rows=0,api_calls=0,rows=results,slurm_job=os.environ['SLURM_JOB_ID'])
    write(root/'REPLAY_COMPLETE.json',summary)
    if not summary['passed']: raise RuntimeError('Replay smoke has unresolved equality/infrastructure errors')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--case-index',type=int)
    a=p.parse_args();main(a.root,a.case_index)
