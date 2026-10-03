"""Bounded CPU pilot with native legal replay and independent evidence scan."""
import argparse
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
sys.path.insert(0, str(BASE / 'experiments/feasibility'))
from reference import Reference
from geometry import active_obstacle, object_points
from contract import (CAPS, SMOKE, PILOT, SOURCE_HASHES, UPSTREAM, bounds,
                      predicate, legal_action, exclusion, decide, witness_at)

ARRAYS = ['qpos', 'qvel', 'ctrl', 'qacc_warmstart', 'qfrc_applied',
          'xfrc_applied', 'mocap_pos', 'mocap_quat', 'act', 'qacc']

def clean(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, (str, float, int, bool, type(None))): return value
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    return {'type': type(value).__name__}

def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(value), indent=2, allow_nan=False)+'\n')
    tmp.replace(path)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def numeric_state(env, ref=None):
    d = env.sim.data
    controller = {k: clean(v) for k, v in vars(env.robots[0].controller).items()
                  if isinstance(v, (np.ndarray, np.generic, str, float, int, bool,
                                    tuple, list, type(None)))}
    return dict(time=float(d.time), arrays={k: np.asarray(getattr(d, k)).copy() for k in ARRAYS},
                sim_state=np.asarray(env.get_sim_state()).copy(), controller=controller,
                reference=clean(vars(ref)) if ref else None,
                python_rng=clean(random.getstate()), numpy_rng=clean(np.random.get_state()),
                action_queue=[], policy_rng=None)

def state_digest(snapshot):
    relevant = {k: snapshot[k] for k in ['time', 'arrays', 'sim_state', 'controller']}
    return hashlib.sha256(json.dumps(clean(relevant), sort_keys=True, allow_nan=False).encode()).hexdigest()

def configure(root):
    upstream = Path(os.environ['CB_UPSTREAM'])
    bench = upstream/'safelibero/libero/libero'
    config = root/'libero_config'; config.mkdir()
    write(config/'config.yaml', dict(benchmark_root=str(bench), bddl_files=str(bench/'bddl_files'),
          init_states=str(bench/'init_files'), assets=str(bench/'assets'),
          datasets=str(upstream/'safelibero/libero/datasets')))
    os.environ['LIBERO_CONFIG_PATH'] = str(config)
    sys.path[:0] = [str(upstream/'main'), str(upstream/'safelibero')]
    from libero.libero.envs.objects.site_object import SiteObject
    from libero.libero.envs.object_states.base_object_states import SiteObjectState
    from libero.libero.envs.predicates.base_predicates import In
    from libero.libero.envs.problems.libero_floor_manipulation import Libero_Floor_Manipulation
    classes = [SiteObject, SiteObjectState, In, Libero_Floor_Manipulation]
    checked = {}
    for cls in classes:
        path = Path(inspect.getfile(cls)).resolve()
        relative = str(path.relative_to((bench/'envs').resolve()))
        if relative not in SOURCE_HASHES or sha(path) != SOURCE_HASHES[relative]:
            raise RuntimeError('Unreviewed loaded predicate implementation: '+str(path))
        checked[relative] = dict(path=str(path), sha256=sha(path))
    if set(checked) != set(SOURCE_HASHES): raise RuntimeError('Missing proof implementation')
    from robosuite.environments.base import MujocoEnv
    from reset_forward import replay_renderer_reset_forward
    replay_renderer_reset_forward(MujocoEnv)
    write(root/'PROOF_SOURCE_AUDIT.json', checked)

def create_env(state, directory):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    suite = benchmark.get_benchmark_dict()[state['suite']](safety_level=state['level'])
    task = suite.get_task(state['task'])
    bddl = Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    initial = suite.get_task_init_states(state['task'])[state['episode']]
    random.seed(state['seed']); np.random.seed(state['seed'])
    env = ControlEnv(bddl_file_name=bddl, use_camera_obs=False,
                     has_offscreen_renderer=False, camera_depths=False)
    env.seed(state['seed']); env.reset()
    obs = env.set_init_state(initial)
    for _ in range(20): obs, _, _, _ = env.step([0.]*6+[-1.])
    expected_goal = [['in', state['target'], state['goal_site']]]
    actual = [[str(v).lower() if i == 0 else v for i, v in enumerate(x)]
              for x in env.env.parsed_problem['goal_state']]
    if actual != expected_goal: raise RuntimeError('Unsupported native goal '+str(actual))
    low, high = map(np.asarray, env.env.action_spec)
    ctl = env.robots[0].controller
    if (len(low) != 7 or not np.array_equal(low, -np.ones(7))
        or not np.array_equal(high, np.ones(7))
        or not np.allclose(ctl.output_max, [.05]*3+[.5]*3)):
        raise RuntimeError('Native capability mismatch')
    np.save(directory/'official_init.npy', initial)
    (directory/'task.bddl').write_bytes(bddl.read_bytes())
    (directory/'model.xml').write_text(env.sim.model.get_xml())
    write(directory/'task.json', dict(state=state, language=task.language, bddl_sha256=sha(bddl),
          native_goal=actual, action_spec=[low, high], controller_output_max=ctl.output_max,
          control_freq=env.env.control_freq, simulator_timestep=env.sim.model.opt.timestep))
    return env, obs

class Audit:
    def __init__(self, env, state, directory):
        self.env, self.state, self.directory = env, state, directory
        self.model = env.sim.model
        self.target_id = env.env.obj_body_id[state['target']]
        self.site_id = self.model.site_name2id(state['goal_site'])
        site = env.env.object_sites_dict[state['goal_site']]
        self.size = np.asarray(site.size).copy()
        self.obstacle = active_obstacle(env, env.env._get_observations())
        self.obstacle_id = env.env.obj_body_id[self.obstacle]
        self.initial_obstacle = env.sim.data.body_xpos[self.obstacle_id].copy()
        self.protected = {x['geom_id'] for x in object_points(env, self.obstacle)[1]}
        self.target = {x['geom_id'] for x in object_points(env, state['target'])[1]}
        robot_root = self.model.body_name2id(env.robots[0].robot_model.root_body)
        self.robot = set()
        for g in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[g])
            while body:
                if body == robot_root: self.robot.add(g); break
                body = int(self.model.body_parentid[body])
        write(directory/'geometry_ids.json', dict(protected=sorted(self.protected),
              target=sorted(self.target), robot=sorted(self.robot), active_obstacle=self.obstacle,
              geoms=[dict(id=g, name=self.model.geom_id2name(g),
                         body=self.model.body_id2name(int(self.model.geom_bodyid[g])))
                     for g in range(self.model.ngeom)], site_size=self.size))
        # A separate simulator forward computes synchronized endpoint poses without
        # changing the physics/controller/warmstart of the executing simulator.
        from robosuite.utils.binding_utils import MjSim
        self.sync = MjSim.from_xml_string(env.sim.model.get_xml())
        self.samples = gzip.open(directory/'samples.jsonl.gz', 'wt')
        self.steps = (directory/'steps.jsonl').open('w')
        self.controllers = gzip.open(directory/'controllers.jsonl.gz', 'wt')
        self.physics = {k: [] for k in ARRAYS+['sim_state', 'time']}
        self.stage = 'initial'; self.action_step = 0; self.substep = 0
        self.total_samples = 0; self.total_integrations = 0
        self.reset_meter()
        self.original_step = env.sim.step
        def observed_step(*args, **kwargs):
            value = self.original_step(*args, **kwargs)
            self.substep += 1; self.total_integrations += 1
            self.sample(env.sim.data, 'integration_cached')
            return value
        env.sim.step = observed_step

    def reset_meter(self):
        self.contact_count = 0; self.max_ingress = -float('inf')

    def sample(self, d, phase):
        p = np.asarray(d.body_xpos[self.target_id]).copy()
        s = np.asarray(d.site_xpos[self.site_id]).copy()
        R = np.asarray(d.site_xmat[self.site_id]).reshape(3, 3).copy()
        lower, upper = bounds(s.tolist(), R.tolist(), self.size.tolist())
        ingress = float(p[1]-lower[1])
        if not np.isfinite(np.r_[p, s, R.flatten(), ingress]).all():
            raise RuntimeError('Nonfinite pose')
        contacts = []; protected_count = 0
        for contact in d.contact[:d.ncon]:
            if contact.dist > 0: continue
            g1, g2 = int(contact.geom1), int(contact.geom2)
            relevant = ((g1 in self.protected and g2 in self.robot | self.target)
                        or (g2 in self.protected and g1 in self.robot | self.target))
            protected_count += int(relevant)
            contacts.append([g1, g2, float(contact.dist), bool(relevant)])
        self.contact_count += protected_count
        self.max_ingress = max(self.max_ingress, ingress)
        row = dict(stage=self.stage, step=self.action_step, substep=self.substep,
                   phase=phase, time=float(d.time), target_pos=p, site_pos=s,
                   site_rotation=R, site_size=self.size, lower=lower, upper=upper,
                   ingress_m=ingress, analytic_success=bool(np.all(p > lower) and np.all(p < upper)),
                   obstacle_pos=np.asarray(d.body_xpos[self.obstacle_id]).copy(),
                   contacts=contacts, protected_contact_count=protected_count)
        self.samples.write(json.dumps(clean(row), allow_nan=False)+'\n'); self.total_samples += 1
        return row

    def synchronized(self):
        src, dst = self.env.sim.data, self.sync.data
        self.sync.set_state(self.env.sim.get_state())
        for k in ARRAYS:
            np.asarray(getattr(dst, k))[:] = np.asarray(getattr(src, k))
        self.sync.forward()
        return self.sample(dst, 'forward_synchronized_copy')

    def endpoint(self, obs, ref, command=None, native_done=None):
        cached = self.sample(self.env.sim.data, 'native_action_endpoint')
        native = bool(self.env.check_success())
        if native != cached['analytic_success']:
            raise RuntimeError('Independent formula differs from native predicate')
        if native_done is not None and bool(native_done) != native:
            raise RuntimeError('env.step done differs from native success')
        synced = self.synchronized()
        snap = numeric_state(self.env, ref)
        for k in ARRAYS: self.physics[k].append(snap['arrays'][k])
        self.physics['sim_state'].append(snap['sim_state']); self.physics['time'].append(snap['time'])
        self.controllers.write(json.dumps(clean(dict(stage=self.stage, step=self.action_step,
              controller=snap['controller'], reference=snap['reference'],
              python_rng=snap['python_rng'], numpy_rng=snap['numpy_rng'], action_queue=[], policy_rng=None)))+'\n')
        displacement = float(np.abs(cached['obstacle_pos']-self.initial_obstacle).sum())
        row = dict(stage=self.stage, step=self.action_step, action=clean(command),
              native_success=native, synchronized_success=synced['analytic_success'],
              predicate_sync_agrees=native == synced['analytic_success'],
              max_ingress_m=self.max_ingress, protected_contact_count=self.contact_count,
              native_obstacle_l1_m=displacement,
              native_poses=clean(cached), synchronized_poses=clean(synced),
              all_object_poses={name: dict(position=self.env.sim.data.body_xpos[idx],
                                 quaternion=self.env.sim.data.body_xquat[idx])
                                for name, idx in self.env.env.obj_body_id.items()},
              all_site_poses=dict(positions=self.env.sim.data.site_xpos, rotations=self.env.sim.data.site_xmat),
              eef_pos=obs['robot0_eef_pos'], eef_quat=obs['robot0_eef_quat'],
              grasp_confirmed=bool(self.env.env._check_grasp(self.env.robots[0].gripper,
                                         self.env.env.objects_dict[self.state['target']])),
              physics_index=len(self.physics['time'])-1, physics_sha256=state_digest(snap),
              reference_phase=ref.phase if ref else None)
        self.steps.write(json.dumps(clean(row), allow_nan=False)+'\n'); self.steps.flush()
        return clean(row)

    def close(self):
        self.samples.close(); self.steps.close(); self.controllers.close()
        np.savez_compressed(self.directory/'physics.npz', **{k: np.asarray(v) for k, v in self.physics.items()})
        self.env.sim.step = self.original_step

def compare_historical(env, prefix_dir, t):
    file = prefix_dir/('checkpoint_%03d.npz'%t)
    if not file.exists(): return None
    with np.load(file) as saved:
        for k in ['qpos', 'qvel', 'ctrl']:
            if not np.array_equal(saved[k], np.asarray(getattr(env.sim.data, k))):
                raise RuntimeError('Historical restore mismatch %d/%s'%(t, k))
    return dict(step=t, path=str(file), sha256=sha(file), exact_fields=['qpos', 'qvel', 'ctrl'])

def execute(root, state, variant, replay=None):
    start = time.monotonic()
    name = state['id']+'_'+variant+('_replay' if replay else '')
    directory = root/'runs'/name; directory.mkdir(parents=True)
    env = None; audit = None
    manifest = dict(state=state, variant=variant, job=os.environ['SLURM_JOB_ID'],
          node=os.environ.get('SLURMD_NODENAME'), code_commit=os.environ['CB_CODE_COMMIT'],
          upstream_commit=UPSTREAM, API_calls=0, training=False, status='running')
    write(directory/'manifest.json', manifest)
    try:
        env, obs = create_env(state, directory)
        audit = Audit(env, state, directory)
        initial_snapshot = numeric_state(env)
        write(directory/'initial_restore.json', initial_snapshot)
        audit.stage = 'prefix' if 'prefix_steps' in state else 'suffix'
        first = audit.endpoint(obs, None)
        prefix_safe = first['protected_contact_count'] == 0
        restore = []; prefix_cost = 0
        if 'prefix_steps' in state:
            prefix_dir = Path(state['prefix_root'])/'runs'/('object_05_r%02d_identity_geometry'%state['prefix_repeat'])
            commands = [json.loads(line) for line in (prefix_dir/'steps.jsonl').read_text().splitlines()]
            restore.append(compare_historical(env, prefix_dir, 0))
            write(directory/'historical_provenance.json', dict(path=str(prefix_dir),
                  commands_sha256=sha(prefix_dir/'steps.jsonl'),
                  original_checkpoint=json.loads((prefix_dir/'checkpoint_252.json').read_text())))
            for t, item in enumerate(commands[:state['prefix_steps']], 1):
                if not legal_action(item['output']): raise RuntimeError('Illegal historical command')
                audit.reset_meter(); audit.action_step=t; audit.substep=0
                obs, _, done, _ = env.step(item['output'])
                row = audit.endpoint(obs, None, item['output'], done)
                prefix_safe = prefix_safe and row['protected_contact_count'] == 0
                match = compare_historical(env, prefix_dir, t)
                if match: restore.append(match)
            if len(commands) < state['prefix_steps']: raise RuntimeError('Incomplete historical prefix')
            prefix_cost = state['prefix_steps']
            write(directory/'historical_restore.json', dict(matches=restore, prefix_safe=prefix_safe,
                  scope='exact qpos/qvel/ctrl checkpoints; new deterministic reference replaces historical policy'))
        ref = Reference(state, variant)
        audit.stage='suffix'; audit.action_step=0; audit.substep=0; audit.reset_meter()
        suffix_initial = audit.endpoint(obs, ref)
        records = [suffix_initial]
        write(directory/'suffix_restore.json', numeric_state(env, ref))
        if replay:
            expected = [json.loads(l) for l in (replay/'steps.jsonl').read_text().splitlines()
                        if json.loads(l)['stage'] == 'suffix' and json.loads(l)['step'] > 0]
            original_initial = json.loads((replay/'initial_restore.json').read_text())
            if state_digest(initial_snapshot) != state_digest(original_initial):
                raise RuntimeError('Fresh full restore mismatch')
        else: expected = None
        for t in range(1, state['budget']+1):
            if expected is not None and t > len(expected): break
            action = expected[t-1]['action'] if expected is not None else ref.step(env, obs).tolist()
            if not legal_action(action): raise RuntimeError('Nonfinite or illegal reference action')
            audit.reset_meter(); audit.action_step=t; audit.substep=0
            obs, _, done, _ = env.step(action)
            row = audit.endpoint(obs, ref, action, done); records.append(row)
            if expected is not None and row['physics_sha256'] != expected[t-1]['physics_sha256']:
                raise RuntimeError('Exact action replay diverged at step '+str(t))
            if done: break
        write(directory/'final_restore.json', numeric_state(env, ref))
        write(directory/'final_poses.json', {k: records[-1][k] for k in [
              'step', 'native_success', 'synchronized_success', 'native_poses', 'synchronized_poses',
              'all_object_poses', 'all_site_poses', 'eef_pos', 'eef_quat', 'grasp_confirmed']})
        summary = dict(manifest, status='complete', run_id=name,
              prefix_contact_safe=prefix_safe, prefix_actions=prefix_cost,
              suffix_actions=records[-1]['step'], task_success=records[-1]['native_success'],
              synchronized_success=records[-1]['synchronized_success'],
              contact_safe=prefix_safe and all(x['protected_contact_count']==0 for x in records),
              native_proxy_safe=max(x['native_obstacle_l1_m'] for x in records)<=.001,
              max_ingress_m=max(x['max_ingress_m'] for x in records),
              samples=audit.total_samples, integration_steps=audit.total_integrations,
              replay_verified=expected is not None, wall_seconds=time.monotonic()-start)
        write(directory/'summary.json', summary)
    finally:
        if audit: audit.close()
        if env: env.close()
    # Verify from persisted samples, not the in-memory success boolean.
    verify_run(directory)
    write(root/'progress.json', dict(last_completed=name, time_unix=time.time()))
    print(json.dumps(dict(completed=name, task=summary['task_success'],
                    contact_safe=summary['contact_safe'], seconds=summary['wall_seconds'])), flush=True)
    return directory

def verify_run(directory):
    summary = json.loads((directory/'summary.json').read_text())
    ids = json.loads((directory/'geometry_ids.json').read_text())
    protected, actors = set(ids['protected']), set(ids['robot'])|set(ids['target'])
    grouped = {}; samples = 0
    with gzip.open(directory/'samples.jsonl.gz', 'rt') as stream:
        for line in stream:
            row=json.loads(line); samples += 1
            expected = predicate(row['target_pos'], row['site_pos'], row['site_rotation'], row['site_size'])
            if expected != row['analytic_success']: raise RuntimeError('Persisted predicate mismatch')
            lower, _ = bounds(row['site_pos'], row['site_rotation'], row['site_size'])
            ingress = row['target_pos'][1]-lower[1]
            if abs(ingress-row['ingress_m'])>1e-12: raise RuntimeError('Persisted ingress mismatch')
            count = sum((c[0] in protected and c[1] in actors) or
                        (c[1] in protected and c[0] in actors) for c in row['contacts'] if c[2] <= 0)
            if count != row['protected_contact_count']: raise RuntimeError('Persisted contact mismatch')
            key=(row['stage'], row['step'])
            item=grouped.setdefault(key, dict(max_ingress_m=-float('inf'), protected_contact_count=0))
            item['max_ingress_m']=max(item['max_ingress_m'], ingress)
            item['protected_contact_count']+=count
    steps=[json.loads(l) for l in (directory/'steps.jsonl').read_text().splitlines()]
    # The initial state is recorded before reference setup and once at suffix start.
    # Audit each positive action; duplicated step-zero records have identical poses.
    for row in steps:
        if row['step'] == 0: continue
        value=grouped[(row['stage'], row['step'])]
        if value['protected_contact_count'] != row['protected_contact_count'] or abs(value['max_ingress_m']-row['max_ingress_m'])>1e-12:
            raise RuntimeError('Persisted sample/step aggregation mismatch')
        if not legal_action(row['action']): raise RuntimeError('Persisted illegal action')
    with np.load(directory/'physics.npz') as values:
        if len(values['time']) != len(steps): raise RuntimeError('Missing full physics checkpoint')
    if samples != summary['samples']: raise RuntimeError('Incomplete sample stream')
    write(directory/'VERIFIED.json', dict(passed=True, samples=samples, actions=len([r for r in steps if r['step']>0]),
          hashes={p.name:sha(p) for p in directory.iterdir() if p.is_file() and p.name!='VERIFIED.json'}))

def summarize(root, states):
    decisions=[]; execution=[]
    for state in states:
        runs={}
        for variant in ['center', 'side']:
            directory=root/'runs'/(state['id']+'_'+variant)
            info=json.loads((directory/'summary.json').read_text()); execution.append(info)
            records=[json.loads(l) for l in (directory/'steps.jsonl').read_text().splitlines()
                     if json.loads(l)['stage']=='suffix']
            runs[variant]=(records, info)
        # Goal/safety contradictions are development sanity controls only.
        caps=CAPS if state['split']=='development_exposed' else {'natural_contact_only':None}
        for name, cap in caps.items():
            started=time.monotonic()
            cert=exclusion(cap, supported=state['split']=='development_exposed')
            witnesses={variant:witness_at(rows, cap, 300, info['prefix_contact_safe'])
                       for variant,(rows,info) in runs.items()}
            label=decide(any(v is not None for v in witnesses.values()),cert)
            center=witnesses['center'] is not None
            pool300=any(witness_at(rows,cap,150,info['prefix_contact_safe']) is not None
                        for rows,info in runs.values())
            initial_valid=all(info['prefix_contact_safe'] and
                    (cap is None or rows[0]['max_ingress_m']<=cap) and rows[0]['protected_contact_count']==0
                    for rows,info in runs.values())
            if not initial_valid and cert:
                # Keep out-of-domain records, but do not count already-unsafe starts
                # as interesting negative pairs.
                label='unknown'; cert=None
            decisions.append(dict(state=state['id'], layout_group=state['layout_group'], split=state['split'],
                condition=name, cap_m=cap, initially_valid=initial_valid,
                label=label, certificate=cert, witness_steps=witnesses,
                certificate_only=decide(False,cert), expert_300=decide(center,None),
                certificate_plus_expert_300=decide(center,cert),
                pool_total300=decide(pool300,None), pool_total600=decide(any(witnesses[v] is not None for v in witnesses),None),
                decision_seconds=time.monotonic()-started))
    labels={label:sum(d['label']==label for d in decisions) for label in ['feasible','infeasible','unknown']}
    methods={}
    for method in ['certificate_only','expert_300','certificate_plus_expert_300','pool_total300','pool_total600']:
        known=[d for d in decisions if d['label']!='unknown']
        determinate=[d for d in known if d[method]!='unknown']
        methods[method]=dict(total=len(decisions), determinate=sum(d[method]!='unknown' for d in decisions),
              covered_known=len(determinate), known_labels=len(known),
              errors_on_known=sum(d[method]!=d['label'] for d in determinate),
              unknown_on_all=sum(d[method]=='unknown' for d in decisions))
    metrics=dict(states=len(states), distinct_layout_groups=len({s['layout_group'] for s in states}),
          new_holdout_layouts=len({s['layout_group'] for s in states if s['split'].endswith('holdout')}),
          judgments=len(decisions), labels=labels, label_coverage=(len(decisions)-labels['unknown'])/len(decisions),
          unknown_fraction=labels['unknown']/len(decisions), methods=methods,
          suffix_actions=sum(r['suffix_actions'] for r in execution),
          prefix_replay_actions=sum(r['prefix_actions'] for r in execution),
          execution_seconds=sum(r['wall_seconds'] for r in execution),
          executions=len(execution), task_successes=sum(r['task_success'] for r in execution),
          contact_safe_executions=sum(r['contact_safe'] for r in execution),
          contact_safe_task_successes=sum(r['contact_safe'] and r['task_success'] and r['synchronized_success'] for r in execution),
          native_proxy_safe_executions=sum(r['native_proxy_safe'] for r in execution),
          API_calls=0, source=os.environ['CB_CODE_COMMIT'], job=os.environ['SLURM_JOB_ID'],
          interpretation='Privileged evidence conformance slice; no predictive generalization claim. Correlated pair variants are not independent states.')
    write(root/'decisions.json', decisions); write(root/'metrics.json',metrics)
    write(root/'execution.json', execution)
    lines=['# Feasibility contract pilot', '', metrics['interpretation'], '',
           'States: %d; distinct layout groups: %d; new held-out layouts: %d.'%(metrics['states'],metrics['distinct_layout_groups'],metrics['new_holdout_layouts']),
           'Labels: '+json.dumps(labels)+'. API calls: 0.', '',
           '| State | Split | Condition | Evidence label | Center witness step | Side witness step |',
           '|---|---|---|---|---:|---:|']
    for d in decisions:
        lines.append('|%s|%s|%s|%s|%s|%s|'%(d['state'],d['split'],d['condition'],d['label'],d['witness_steps']['center'],d['witness_steps']['side']))
    lines += ['', 'Task successes %d/%d; contact-safe task successes %d/%d. Native proxy is separate.'%(metrics['task_successes'],metrics['executions'],metrics['contact_safe_task_successes'],metrics['executions']),
              '', 'Certificate-only and certificate-assisted outputs have privileged contract access. These conformance errors are not an independent estimate of a new prediction method.',
              'All finite reference failures remain unknown unless a separate exclusion certificate applies.',
              'Near positive candidates without witnesses are incomplete pairs, not feasible ground truth.',
              'Natural checkpoints share an exposed layout and preserve their history; they cannot validate the constructed negative mechanism.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return metrics

def main(root, stage):
    configure(root)
    states=SMOKE if stage=='smoke' else PILOT
    write(root/'protocol.json',dict(stage=stage, states=states, caps=CAPS, reference_variants=['center','side'],
          action_budget_each=300, seed=7, upstream=UPSTREAM, code_commit=os.environ['CB_CODE_COMMIT'],
          slurm_job=os.environ['SLURM_JOB_ID'], contract='SC-INGRESS-v1 / SC-CONTACT-v1',API_calls=0))
    for state in states:
        center=execute(root,state,'center')
        execute(root,state,'side')
        if stage=='smoke': execute(root,state,'center',replay=center)
    metrics=summarize(root,states)
    if stage=='smoke':
        decisions=json.loads((root/'decisions.json').read_text())
        if not any(d['condition']=='obvious_positive_candidate' and d['label']=='feasible' for d in decisions):
            raise RuntimeError('No broad-cap safe witness; pilot gate remains closed')
    write(root/'COMPLETE.json',dict(stage=stage, passed=True,metrics=metrics,code_commit=os.environ['CB_CODE_COMMIT']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('stage',choices=['smoke','pilot']);a=p.parse_args()
    try: main(a.root,a.stage)
    except BaseException as e:
        write(a.root/'STOP.json',dict(type=type(e).__name__,reason=str(e),stage=a.stage,job=os.environ.get('SLURM_JOB_ID')))
        raise
