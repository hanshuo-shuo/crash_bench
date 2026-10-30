"""HIL suite runner on standard LIBERO: prefixes, policy-twin branches and summaries.

Protocol: docs/generative_verifier/HIL_SUITE_PROTOCOL.md; contract in hil_suite.py. Every run is a fresh process that
builds the task's environment as openpi's LIBERO evaluation does (256 px cameras, env.seed(7), reset, official initial
state, ten dummy steps). A prefix then runs the policy (experiments/feasibility/serve.py on 127.0.0.1:PORT, queried every
five steps) with task A's prompt until A succeeds or the suite's step limit, recording every applied action and physics
hash. A branch replays the prefix's applied actions to its checkpoint, checking the physics hash at every step, resets the
policy RNG to the prefix seed and continues under its condition's prompt. Each step records A's goal and the goals of the
other tasks the unit names (the other task's BDDL goal evaluated in A's environment; LIBERO-Goal and LIVING_ROOM_SCENE2
tasks share one scene).

ROOT --phase prefix|branch --shard K --shards N --port P runs a shard; ROOT --phase prefix|branch --index I --port P runs one
unit; ROOT --summarize-prefix, ROOT --plan-branches and ROOT --summarize aggregate (the last also checks recorded goal values
against native LIBERO checks on up to 400 branches, ROOT --state-check on all); ROOT --smoke --port P does all of it.
"""
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from .inventory import read, sha, write
from . import hil_suite as S

LIBERO_COMMIT = '8f1084e3132a39270c3a13ebe37270a43ece2a01'
UNITS = dict(prefix='prefixes', branch='branches', probe='probes')


def configure(root):
    """Standard LIBERO from the pinned checkout, a run-local LIBERO config, and the historical physics hash."""
    libero = Path(os.environ['CB_LIBERO']).resolve(); benchmark = libero/'libero/libero'
    cfg = root/'libero_config'; cfg.mkdir(exist_ok=True)
    if not (cfg/'config.yaml').exists():
        write(cfg/'config.yaml', dict(benchmark_root=str(benchmark), bddl_files=str(benchmark/'bddl_files'), init_states=str(benchmark/'init_files'),
                                      assets=str(benchmark/'assets'), datasets=str(libero/'libero/datasets')))
    os.environ['LIBERO_CONFIG_PATH'] = str(cfg)
    repo = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(libero), str(repo/'experiments/feasibility')]
    import libero.libero as module
    if not str(Path(module.__file__).resolve()).startswith(str(libero) + os.sep):
        raise RuntimeError('LIBERO resolved outside the pinned checkout: ' + module.__file__)
    import importlib.util
    spec = importlib.util.spec_from_file_location('fd_uncertainty_geometry', repo/'analysis/uncertainty/geometry.py')
    geometry = importlib.util.module_from_spec(spec); spec.loader.exec_module(geometry)
    return geometry.physical_hash


def seed_globals():
    """Construction and reset draw on the global RNGs before env.seed: seed them so every run renders the same scene."""
    import random
    import numpy as np
    random.seed(S.ENV_SEED); np.random.seed(S.ENV_SEED)


def make_env(suite, task_id, init):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    tasks = benchmark.get_benchmark_dict()[suite](); task = tasks.get_task(task_id)
    bddl = Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    seed_globals()
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=S.RESOLUTION, camera_widths=S.RESOLUTION)
    env.seed(S.ENV_SEED); env.reset()
    obs = env.set_init_state(tasks.get_task_init_states(task_id)[init])
    return env, task, obs


def goal_checker(env, suite, task_id):
    """Another task's BDDL goal (a conjunction) evaluated with A's environment's own predicates."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.bddl_utils import robosuite_parse_problem
    task = benchmark.get_benchmark_dict()[suite]().get_task(task_id)
    goal = robosuite_parse_problem(str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file))['goal_state']
    names = {x for state in goal for x in state[1:]}
    missing = sorted(n for n in names if n not in env.env.object_states_dict)
    if missing:
        raise RuntimeError('Other task names objects absent from this scene: %s' % missing)
    return lambda: all([env.env._eval_predicate(state) for state in goal])


def replan_steps():
    return S.REPLANS[os.environ['CB_POLICY_CONFIG']]


def observation(obs, prompt, suite):
    """The policy's input: openpi's LIBERO element, or for OFT the raw rotated 256 px images plus the suite (serve_oft.py)."""
    import numpy as np
    from .paired_policy import policy_input
    data = policy_input(obs, prompt)
    if os.environ['CB_POLICY_CONFIG'] == 'openvla_oft':
        data.update({'observation/image': np.ascontiguousarray(obs['agentview_image'][::-1, ::-1]),
                     'observation/wrist_image': np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1, ::-1]), 'suite': suite})
    return data


class Client:
    """The policy stream for one run, with the RNG reset and the per-request identity checks of the HIL runner."""
    def __init__(self, port, seed, run_id):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        self.client = WebsocketClientPolicy('127.0.0.1', port); self.run_id = run_id; self.requests = []
        self.ack = self.client.infer({'__paired_reset_rng__': seed, 'run_id': run_id})
        if self.ack.get('seed') != seed or self.ack.get('run_id') != run_id or self.ack.get('requests') != 0:
            raise RuntimeError('Policy RNG reset acknowledgement mismatch')

    def infer(self, data, step, log):
        import numpy as np
        from serve import digest
        result = self.client.infer(data); info = result['diagnostic']; actions = np.asarray(result['actions']); replan = replan_steps()
        if (info['run_id'] != self.run_id or info['request_index'] != len(self.requests) + 1 or info['input_sha256'] != digest(data)
                or actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < replan or not np.isfinite(actions).all()):
            raise RuntimeError('Policy stream, input or action shape mismatch')
        self.requests.append(dict(step=step, **{k: info[k] for k in ('request_index', 'input_sha256', 'rng_before', 'rng_after',
                                  'output_replayed', 'native_output_linf_difference', 'native_chunk_sha256')}))
        log.write(json.dumps(dict(self.requests[-1], actions=actions.tolist())) + '\n'); log.flush()
        return actions[:replan]

    def close(self):
        self.client._ws.close()


def record(env, step, obs, action, done, physical_hash, checkers):
    import numpy as np
    return dict(step=step, action=None if action is None else [float(x) for x in action], physics=physical_hash(env),
                eef=np.asarray(obs['robot0_eef_pos']).tolist(), gripper=np.asarray(obs['robot0_gripper_qpos']).tolist(),
                a=bool(env.env._check_success()), others={str(k): bool(f()) for k, f in checkers.items()}, done=bool(done))


def policy_loop(env, obs, client, prompt, suite, step, end, stop, physical_hash, checkers, folder):
    """Query every replan steps until stop(record) or step == end; returns the trace after the start record."""
    import collections
    import numpy as np
    from serve import digest
    queue = collections.deque(); trace = []; first = None; outside = 0
    low, high = env.env.action_spec
    with (folder/'POLICY.jsonl').open('w') as log:
        while step < end:
            if not queue:
                data = observation(obs, prompt, suite)
                if first is None:
                    first = dict(with_prompt=digest(data), state=digest({k: v for k, v in data.items() if k != 'prompt'}))
                    np.savez_compressed(folder/'first_policy_input.npz', **{k: v for k, v in data.items() if k not in ('prompt', 'suite')})
                queue.extend(client.infer(data, step, log))
            action = queue.popleft()
            outside += int(any(a < l or a > h for a, l, h in zip(action, low, high)))
            obs, _, done, _ = env.step(action.tolist()); step += 1
            trace.append(record(env, step, obs, action, done, physical_hash, checkers))
            if stop(trace[-1]):
                break
    return trace, first, outside


def save(folder, trace, row):
    with gzip.open(folder/'STEPS.jsonl.gz', 'wt') as stream:
        stream.writelines(json.dumps(x) + '\n' for x in trace)
    write(folder/'ROW.json', row)
    write(folder/'HASHES.json', {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file() and p.name != 'HASHES.json'})


def load(folder):
    for name, digest in read(folder/'HASHES.json').items():
        if sha(folder/name) != digest:
            raise RuntimeError('Run output changed: %s' % (folder/name))
    with gzip.open(folder/'STEPS.jsonl.gz', 'rt') as stream:
        return read(folder/'ROW.json'), [json.loads(x) for x in stream]


def provenance():
    return dict(api_calls=0, code_commit=os.environ.get('CB_CODE_COMMIT'), slurm_job=os.environ.get('SLURM_JOB_ID'),
                policy_config=os.environ.get('CB_POLICY_CONFIG'), libero_commit=LIBERO_COMMIT)


def run_prefix(root, index, port):
    unit = read(root/'PLAN.json')['prefixes'][index]; folder = root/'prefix'/unit['id']; folder.mkdir(parents=True)
    physical_hash = configure(root); started = time.monotonic(); env = client = None
    try:
        env, task, obs = make_env(unit['suite'], unit['task'], unit['init'])
        if task.language != unit['prompt'] or task.name != unit['name']:
            raise RuntimeError('Prompt differs from the LIBERO task language')
        checkers = {k: goal_checker(env, unit['suite'], k) for k in unit['others']}
        trace = [record(env, 0, obs, None, False, physical_hash, checkers)]
        if trace[0]['physics'] != physical_hash(env):
            raise RuntimeError('Goal readout mutated physics')
        for step in range(1, S.WAIT + 1):
            obs, _, done, _ = env.step(S.DUMMY); trace.append(record(env, step, obs, S.DUMMY, done, physical_hash, checkers))
        client = Client(port, unit['seed'], '%s_prefix_%s' % (os.environ['SLURM_JOB_ID'], unit['id']))
        tail, first, outside = policy_loop(env, obs, client, unit['prompt'], unit['suite'], S.WAIT, S.WAIT + unit['max_steps'], lambda x: x['done'],
                                           physical_hash, checkers, folder)
        trace += tail; success = trace[-1]['done']
        row = dict(unit, success=success, policy_steps=trace[-1]['step'] - S.WAIT if success else None, steps=trace[-1]['step'],
                   start_eef=trace[S.WAIT]['eef'], first_policy_input_sha256=first['with_prompt'], rng_reset=client.ack,
                   policy_requests=len(client.requests), replayed_outputs=sum(r['output_replayed'] for r in client.requests),
                   actions_outside_spec=outside, elapsed_seconds=time.monotonic() - started, **provenance())
        save(folder, trace, row)
    finally:
        if client is not None:
            client.close()
        if env is not None:
            env.close()


def perturb(env, unit, obs, rest, last_action, physical_hash):
    """Stage 3: apply the unit's physical intervention at the checkpoint (protocol appendix A).

    Returns (obs, info, scripted) where scripted lists the PUSH steps' records; info['applied'] is False with a reason when the
    intervention does not apply to this state (nothing resting or lifted, no clear destination, a fixed receptacle, penetration).
    """
    import numpy as np
    m, d = env.sim.model, env.sim.data; e = env.env
    moved, places = S.goal_parts(e.parsed_problem['goal_state'])
    eef = np.asarray(obs['robot0_eef_pos']); xyz = lambda n: np.asarray(d.body_xpos[e.obj_body_id[n]]).tolist()
    names = sorted(set(e.objects_dict) | set(e.fixtures_dict))
    others = lambda skip: [xyz(n)[:2] for n in names if n not in skip and n in e.obj_body_id]
    if unit['condition'] == 'PUSH':
        target = eef + np.asarray(S.PUSH_OFFSET); scripted = []; low, high = e.action_spec
        for k in range(S.PUSH_STEPS):
            delta = (target - np.asarray(obs['robot0_eef_pos'])) / S.PUSH_GAIN
            action = np.clip(np.concatenate([delta, np.zeros(3), [last_action[6]]]), low, high)
            obs, _, done, _ = env.step(action.tolist())
            scripted.append(record(env, unit['checkpoint'] + k + 1, obs, action, done, physical_hash, {}))
            scripted[-1]['scripted'] = True
        moved_by = float(np.linalg.norm(np.asarray(obs['robot0_eef_pos']) - eef))
        return obs, dict(applied=True, reason=None, eef_from=eef.tolist(), eef_to=np.asarray(obs['robot0_eef_pos']).tolist(), displacement=moved_by), scripted

    def move(name, xy, z, quat=None):
        j = m.joint_name2id(name + '_joint0'); adr, dof = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        before = d.qpos[adr:adr + 7].copy()
        d.qpos[adr:adr + 3] = [xy[0], xy[1], z]
        if quat is not None:
            d.qpos[adr + 3:adr + 7] = quat
        d.qvel[dof:dof + 6] = 0.; env.sim.forward()
        geoms = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == e.obj_body_id[name] or
                 (m.body_id2name(int(m.geom_bodyid[g])) or '').startswith(name)}
        deep = [c for c in d.contact[:d.ncon] if (int(c.geom1) in geoms or int(c.geom2) in geoms) and c.dist < S.PENETRATION]
        return before.tolist(), len(deep)

    def finish(name, before, penetrating, extra):
        info = dict(applied=penetrating == 0, reason=None if penetrating == 0 else 'penetration', object=name, qpos_before=before, **extra)
        return env.env._get_observations(force_update=True), info, []

    lifted = [n for n in moved if n in rest and xyz(n)[2] - rest[n]['z'] >= S.LIFTED]
    if unit['condition'] == 'OBJ':
        resting = sorted((n for n in moved if n in rest and abs(xyz(n)[2] - rest[n]['z']) < S.RESTING), key=lambda n: np.linalg.norm(np.asarray(xyz(n)) - eef))
        if not resting:
            return obs, dict(applied=False, reason='nothing resting'), []
        n = resting[0]; shift = S.best_shift(xyz(n)[:2], others({n}), S.SHIFT)
        if shift is None:
            return obs, dict(applied=False, reason='no clear destination'), []
        before, deep = move(n, shift['to'], xyz(n)[2])
        return finish(n, before, deep, dict(shift=shift))
    if unit['condition'] == 'DROP':
        if not lifted:
            return obs, dict(applied=False, reason='nothing lifted'), []
        n = min(lifted, key=lambda x: np.linalg.norm(np.asarray(xyz(x)) - eef)); shift = S.best_shift(eef[:2].tolist(), others({n}), S.DROP_OFFSET)
        if shift is None:
            return obs, dict(applied=False, reason='no clear destination'), []
        before, deep = move(n, shift['to'], rest[n]['z'], rest[n]['quat'])
        return finish(n, before, deep, dict(shift=shift))
    receptacle = places[0] if places else None
    if receptacle not in e.objects_dict:
        return obs, dict(applied=False, reason='fixed receptacle'), []
    if not lifted:
        return obs, dict(applied=False, reason='nothing lifted'), []
    if any(np.linalg.norm(np.asarray(xyz(n))[:2] - np.asarray(xyz(receptacle))[:2]) < S.OVER for n in lifted):
        return obs, dict(applied=False, reason='already over the receptacle'), []
    shift = S.best_shift(xyz(receptacle)[:2], others({receptacle} | set(lifted)), S.SHIFT)
    if shift is None:
        return obs, dict(applied=False, reason='no clear destination'), []
    before, deep = move(receptacle, shift['to'], xyz(receptacle)[2])
    return finish(receptacle, before, deep, dict(shift=shift))


def outcome3(trace, unit, start, info):
    if not info['applied']:
        return dict(applied=False, reason=info['reason'], a_done=None)
    return dict(applied=True, reason=None, **S.outcome(trace, unit, start))


def run_branch(root, index, port):
    plan = read(root/'PLAN.json'); unit = plan['branches'][index]
    folder = root/'branch'/unit['id']; folder.mkdir(parents=True)
    prefix_row, tape = load(Path(plan['prefix_root'])/'prefix'/unit['prefix'])
    if not prefix_row['success'] or not S.WAIT <= unit['checkpoint'] < tape[-1]['step']:
        raise RuntimeError('Checkpoint outside the successful prefix')
    physical_hash = configure(root); started = time.monotonic(); env = client = None
    try:
        env, task, obs = make_env(unit['suite'], unit['task'], unit['init'])
        if physical_hash(env) != tape[0]['physics']:
            raise RuntimeError('Initial state differs from the prefix')
        rest = {}
        for item in tape[1:unit['checkpoint'] + 1]:
            obs, _, done, _ = env.step(item['action'])
            if physical_hash(env) != item['physics'] or bool(done) != item['done'] or done:
                raise RuntimeError('Replay differs from the prefix at step %d' % item['step'])
            if item['step'] == S.WAIT and unit['condition'] in S.PERTURB:
                # Objects at rest after the wait steps: the heights and orientations a dropped object returns to.
                rest = {n: dict(z=float(env.sim.data.body_xpos[i][2]), quat=env.sim.data.qpos[env.sim.model.jnt_qposadr[env.sim.model.joint_name2id(n + '_joint0')] + 3:
                                                                                         env.sim.model.jnt_qposadr[env.sim.model.joint_name2id(n + '_joint0')] + 7].tolist())
                        for n, i in env.env.obj_body_id.items() if n in env.env.objects_dict}
        at = physical_hash(env)
        checkers = {k: goal_checker(env, unit['suite'], k) for k in unit['record']}
        start = record(env, unit['checkpoint'], obs, None, False, physical_hash, checkers)
        if start['physics'] != at:
            raise RuntimeError('Checkpoint readout mutated physics')
        info, scripted = None, []
        if unit['condition'] in S.PERTURB:
            obs, info, scripted = perturb(env, unit, obs, rest, tape[unit['checkpoint']]['action'], physical_hash)
            if not info['applied']:
                trace = [start]
                row = dict(unit, physics_at_checkpoint=at, perturbation=info, outcome=outcome3(trace, unit, prefix_row['start_eef'], info),
                           elapsed_seconds=time.monotonic() - started, **provenance())
                save(folder, trace, row); return
        client = Client(port, unit['seed'], '%s_%s' % (os.environ['SLURM_JOB_ID'], unit['id']))
        other = None if unit['other'] is None else str(unit['other'])
        stop = (lambda x: x['others'][other]) if unit['stop_on'] == 'B' else (lambda x: x['done'])
        begin = unit['checkpoint'] + len(scripted)
        tail, first, outside = policy_loop(env, obs, client, unit['prompt'], unit['suite'], begin, unit['checkpoint'] + unit['budget'], stop,
                                           physical_hash, checkers, folder)
        trace = [start] + scripted + tail
        result = S.outcome(trace, unit, prefix_row['start_eef']) if info is None else outcome3(trace, unit, prefix_row['start_eef'], info)
        row = dict(unit, physics_at_checkpoint=at, final_state=[float(x) for x in env.sim.get_state().flatten()], perturbation=info, prefix_steps=tape[-1]['step'], start_eef=prefix_row['start_eef'],
                   outcome=result, first_policy_input_sha256=first['with_prompt'],
                   policy_input_state_sha256=first['state'], rng_reset=client.ack, policy_requests=len(client.requests),
                   replayed_outputs=sum(r['output_replayed'] for r in client.requests),
                   maximum_native_difference=max(r['native_output_linf_difference'] for r in client.requests),
                   actions_outside_spec=outside, elapsed_seconds=time.monotonic() - started, **provenance())
        save(folder, trace, row)
    finally:
        if client is not None:
            client.close()
        if env is not None:
            env.close()


def run_probe(root, index, port):
    """Stage 5: replay a prefix and, every PROBE_EVERY policy steps, query the first action chunk under A, each B, an empty
    prompt (all with the prefix seed) and A with the next seed (the noise floor); no policy action is executed."""
    import numpy as np
    from serve import digest
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    plan = read(root/'PLAN.json'); unit = plan['probes'][index]; folder = root/'probe'/unit['id']; folder.mkdir(parents=True)
    prefix_row, tape = load(Path(plan['prefix_root'])/'prefix'/unit['id'])
    physical_hash = configure(root); started = time.monotonic(); env = client = None
    names = read(root/'NAMES.json')
    prompts = [('A', prefix_row['prompt'], prefix_row['seed']), ('noise', prefix_row['prompt'], (prefix_row['seed'] + 1) % 2 ** 32),
               ('empty', '', prefix_row['seed'])] + [('B%d' % b, S.language(names[unit['suite']][b]), prefix_row['seed']) for b in prefix_row['others']]
    try:
        env, task, obs = make_env(unit['suite'], unit['task'], unit['init'])
        if physical_hash(env) != tape[0]['physics']:
            raise RuntimeError('Initial state differs from the prefix')
        client = WebsocketClientPolicy('127.0.0.1', port); points = set(S.probe_points(prefix_row)); rows = []
        for item in tape[1:max(points) + 1]:
            obs, _, done, _ = env.step(item['action'])
            if physical_hash(env) != item['physics'] or done:
                raise RuntimeError('Replay differs from the prefix at step %d' % item['step'])
            if item['step'] not in points:
                continue
            chunks = {}
            for name, text, seed in prompts:
                run_id = '%s_probe_%s_%d_%s' % (os.environ['SLURM_JOB_ID'], unit['id'], item['step'], name)
                ack = client.infer({'__paired_reset_rng__': seed, 'run_id': run_id})
                if ack.get('seed') != seed or ack.get('requests') != 0:
                    raise RuntimeError('Policy RNG reset acknowledgement mismatch')
                data = observation(obs, text, unit['suite']); result = client.infer(data); info = result['diagnostic']
                if info['run_id'] != run_id or info['request_index'] != 1 or info['input_sha256'] != digest(data):
                    raise RuntimeError('Probe stream mismatch')
                chunks[name] = np.asarray(result['actions'])[:replan_steps()]
            gap = lambda k: float(np.linalg.norm(chunks[k] - chunks['A']))
            rows.append(dict(step=item['step'], progress=(item['step'] - S.WAIT) / prefix_row['policy_steps'], eef=item['eef'],
                             d_noise=gap('noise'), d_empty=gap('empty'), d_B={k[1:]: gap(k) for k in chunks if k.startswith('B')},
                             chunks={k: v.tolist() for k, v in chunks.items()}))
        write(folder/'PROBE.json', dict(unit=unit, points=rows, elapsed_seconds=time.monotonic() - started, **provenance()))
        write(folder/'HASHES.json', {'PROBE.json': sha(folder/'PROBE.json')})
    finally:
        if client is not None:
            client._ws.close()
        if env is not None:
            env.close()


def shard_units(plan, phase, shard, shards):
    """Prefixes round-robin; a prefix's branches all go to that prefix's shard, so twins share one node."""
    order = {p['id']: k for k, p in enumerate(plan['prefix_order'])} if phase == 'branch' else None
    units = plan[UNITS[phase]]
    return [i for i, u in enumerate(units) if (order[u['prefix']] if phase == 'branch' else i) % shards == shard]


def run(root, phase, shard, shards, port):
    plan = read(root/'PLAN.json'); units = plan[UNITS[phase]]; done = 0
    for i in shard_units(plan, phase, shard, shards):
        (root/'logs').mkdir(exist_ok=True)
        with (root/'logs'/('%s_%s.log' % (phase, units[i]['id']))).open('w') as log:
            subprocess.run([sys.executable, '-u', '-m', 'verifier_dataset.hil_suite_run', str(root), '--phase', phase, '--index', str(i),
                            '--port', str(port)], stdout=log, stderr=subprocess.STDOUT, check=True)
        done += 1
        write(root/('PROGRESS_%s_%d.json' % (phase, shard)), dict(shard=shard, done=done, last=units[i]['id'], slurm_job=os.environ.get('SLURM_JOB_ID')))
    write(root/('SHARD_%s_%d_COMPLETE.json' % (phase, shard)), dict(passed=True, shard=shard, units=done, slurm_job=os.environ.get('SLURM_JOB_ID')))


def summarize_prefix(root):
    plan = read(root/'PLAN.json'); rows = []
    for k in range(plan['shards']):
        if not read(root/('SHARD_prefix_%d_COMPLETE.json' % k))['passed']:
            raise RuntimeError('Prefix shard %d incomplete' % k)
    for unit in plan['prefixes']:
        row, trace = load(root/'prefix'/unit['id'])
        if [x['step'] for x in trace] != list(range(len(trace))) or row['success'] != trace[-1]['done']:
            raise RuntimeError('Prefix trace inconsistent: ' + unit['id'])
        rows.append({k: row[k] for k in ('id', 'suite', 'task', 'init', 'name', 'prompt', 'others', 'seed', 'max_steps', 'success',
                                         'policy_steps', 'steps', 'start_eef', 'elapsed_seconds', 'replayed_outputs', 'actions_outside_spec')})
    gate = S.stage0(rows)
    gate.update(seconds_per_run=S.H.distribution(r['elapsed_seconds'] for r in rows), replayed_outputs=sum(r['replayed_outputs'] for r in rows))
    write(root/'PREFIXES.json', rows); write(root/'STAGE0.json', gate)
    write(root/'PREFIX_COMPLETE.json', dict(passed=True, mode=plan['mode'], prefixes=len(rows), reading=gate['reading'], **provenance()))
    write(root/'PREFIX_HASHES.json', {n: sha(root/n) for n in ('PLAN.json', 'PREFIXES.json', 'STAGE0.json', 'PREFIX_COMPLETE.json')})
    return gate


def twin_check(units, rows):
    """Branches of one prefix and checkpoint started from one physics state, one policy observation and one RNG key."""
    groups = {}
    for u in units:
        groups.setdefault((u['prefix'], u['checkpoint']), []).append(rows[u['id']])
    for key, rs in groups.items():
        for field in ('physics_at_checkpoint', 'policy_input_state_sha256'):
            if len({r[field] for r in rs}) != 1:
                raise RuntimeError('Twin runs differ in %s at %s' % (field, key))
        if len({r['rng_reset']['key_hex'] for r in rs}) != 1:
            raise RuntimeError('Twin runs differ in the policy RNG key at %s' % (key,))
        prompts = [r['prompt'] for r in rs if r['condition'] != 'C0R']
        if len(set(prompts)) != len(prompts):
            raise RuntimeError('Two conditions share a prompt at %s' % (key,))
    return len(groups)


def reference_distances(units, traces):
    """For each mid-task SW run: distance from A's end effector at t to B's reference path (the t = 0 SW run)."""
    key = lambda u: (u['suite'], u['task'], u['other'], u['init'])
    reference = {key(u): [x['eef'] for x in traces[u['id']]] for u in units if u['condition'] == 'SW' and u['phase'] == 0}
    return {key(u) + (u['phase'],): min(math.dist(traces[u['id']][0]['eef'], p) for p in reference[key(u)])
            for u in units if u['condition'] == 'SW' and u['phase'] and key(u) in reference}


def summarize_physical(root):
    """Stage 3: recovery after each perturbation against C0 at the same prefix and checkpoint in the stage 1-2 branch run."""
    plan = read(root/'PLAN.json'); units = plan['branches']; base_root = Path(plan['baseline_root'])
    for k in range(plan['shards']):
        if not read(root/('SHARD_branch_%d_COMPLETE.json' % k))['passed']:
            raise RuntimeError('Branch shard %d incomplete' % k)
    for n, digest in plan['baseline_hashes'].items():
        if sha(base_root/n) != digest:
            raise RuntimeError('Baseline run output changed: ' + n)
    with gzip.open(base_root/'RUNS.jsonl.gz', 'rt') as stream:
        baseline = {}
        for r in map(json.loads, stream):
            if r['condition'] == 'C0':
                baseline[(S.prefix_id(r['suite'], r['task'], r['init']), r['checkpoint'])] = r['a_done']
    rows, outcomes = {}, {}
    for u in units:
        row, trace = load(root/'branch'/u['id'])
        if outcome3(trace, u, row['start_eef'] if 'start_eef' in row else None, row['perturbation']) != row['outcome']:
            raise RuntimeError('Recomputed outcome differs: ' + u['id'])
        rows[u['id']], outcomes[u['id']] = row, row['outcome']
    runs = [dict({k: u[k] for k in ('id', 'suite', 'task', 'init', 'phase', 'checkpoint', 'condition')}, perturbation=rows[u['id']]['perturbation'],
                 elapsed_seconds=rows[u['id']]['elapsed_seconds'], **outcomes[u['id']]) for u in units]
    with gzip.open(root/'RUNS.jsonl.gz', 'wt') as stream:
        stream.writelines(json.dumps(r) + '\n' for r in runs)
    reading = dict(mode=plan['mode'], stage3=S.stage3(units, outcomes, baseline), runs=len(runs), excluded=plan['excluded'],
                   seconds_per_run=S.H.distribution(r['elapsed_seconds'] for r in runs), total_run_hours=sum(r['elapsed_seconds'] for r in runs) / 3600.)
    write(root/'SUITE_READING.json', reading)
    write(root/'SUITE_COMPLETE.json', dict(passed=True, mode=plan['mode'], experiment='physical', runs=len(runs),
                                           readings={c: v['reading'] for c, v in reading['stage3'].items()}, **provenance()))
    write(root/'HASHES.json', {n: sha(root/n) for n in ('PLAN.json', 'RUNS.jsonl.gz', 'SUITE_READING.json', 'SUITE_COMPLETE.json')})


def summarize_probe(root):
    """Stage 5: ratios to the noise floor by progress bin, per group, with A's distance to B's reference path at each point."""
    plan = read(root/'PLAN.json')
    for k in range(plan['shards']):
        if not read(root/('SHARD_probe_%d_COMPLETE.json' % k))['passed']:
            raise RuntimeError('Probe shard %d incomplete' % k)
    reference = {}
    with gzip.open(Path(plan['branch_root'])/'RUNS.jsonl.gz', 'rt') as stream:
        ids = [r['id'] for r in map(json.loads, stream) if r['condition'] == 'SW' and r['phase'] == 0]
    for i in ids:
        row, trace = load(Path(plan['branch_root'])/'branch'/i)
        reference[(row['suite'], row['task'], row['other'], row['init'])] = [x['eef'] for x in trace]
    points = []
    for u in plan['probes']:
        folder = root/'probe'/u['id']
        if sha(folder/'PROBE.json') != read(folder/'HASHES.json')['PROBE.json']:
            raise RuntimeError('Probe output changed: ' + u['id'])
        for x in read(folder/'PROBE.json')['points']:
            for b, d in x['d_B'].items():
                ref = reference.get((u['suite'], u['task'], int(b), u['init']))
                points.append(dict(suite=u['suite'], task=u['task'], init=u['init'], other=int(b), group=u['group'], progress=x['progress'],
                                   d_B=d, d_empty=x['d_empty'], d_noise=x['d_noise'],
                                   to_B_path=None if ref is None else min(math.dist(x['eef'], q) for q in ref)))
    reading = dict(mode=plan['mode'], points=len(points), groups={})
    for g in sorted({p['group'] for p in points}):
        ps = [p for p in points if p['group'] == g]; near = [p for p in ps if p['to_B_path'] is not None and p['d_noise'] > 0]
        by = S.probe_summary(ps)
        reading['groups'][g] = dict(points=len(ps), by_progress=by, reading=S.probe_reading(by),
                                    path_correlation=S.estimate(near, lambda xs: S.spearman([x['to_B_path'] for x in xs], [x['d_B'] / x['d_noise'] for x in xs])))
    with gzip.open(root/'POINTS.jsonl.gz', 'wt') as stream:
        stream.writelines(json.dumps(p) + '\n' for p in points)
    write(root/'SUITE_READING.json', reading)
    write(root/'SUITE_COMPLETE.json', dict(passed=True, mode=plan['mode'], experiment='probe', points=len(points), **provenance()))
    write(root/'HASHES.json', {n: sha(root/n) for n in ('PLAN.json', 'POINTS.jsonl.gz', 'SUITE_READING.json', 'SUITE_COMPLETE.json')})


def summarize(root):
    plan = read(root/'PLAN.json')
    if plan.get('experiment') == 'physical':
        return summarize_physical(root)
    if plan.get('experiment') == 'probe':
        return summarize_probe(root)
    units = plan['branches']
    for k in range(plan['shards']):
        if not read(root/('SHARD_branch_%d_COMPLETE.json' % k))['passed']:
            raise RuntimeError('Branch shard %d incomplete' % k)
    rows, traces, outcomes = {}, {}, {}
    for u in units:
        row, trace = load(root/'branch'/u['id'])
        start = row['start_eef']
        if S.outcome(trace, u, start) != row['outcome']:
            raise RuntimeError('Recomputed outcome differs: ' + u['id'])
        rows[u['id']], traces[u['id']], outcomes[u['id']] = row, trace, row['outcome']
    twins = twin_check(units, rows)
    distances = reference_distances(units, traces)
    runs = [dict({k: u[k] for k in ('id', 'stages', 'group', 'suite', 'task', 'other', 'init', 'phase', 'checkpoint', 'condition')},
                 elapsed_seconds=rows[u['id']]['elapsed_seconds'], replayed_outputs=rows[u['id']]['replayed_outputs'],
                 actions_outside_spec=rows[u['id']]['actions_outside_spec'], **outcomes[u['id']]) for u in units]
    with gzip.open(root/'RUNS.jsonl.gz', 'wt') as stream:
        stream.writelines(json.dumps(r) + '\n' for r in runs)
    s1 = S.stage1(units, outcomes, distances); s2 = S.stage2(units, outcomes)
    reading = dict(mode=plan['mode'], stage1=s1, stage2=s2, claims=S.claims(s1, s2), twin_groups=twins, runs=len(runs),
                   excluded=plan['excluded'], seconds_per_run=S.H.distribution(r['elapsed_seconds'] for r in runs),
                   total_run_hours=sum(r['elapsed_seconds'] for r in runs) / 3600., replayed_outputs=sum(r['replayed_outputs'] for r in runs),
                   actions_outside_spec=sum(r['actions_outside_spec'] for r in runs))
    repeats = [u for u in units if u['condition'] == 'C0R']
    if repeats:
        pairs = []
        for u in repeats:
            twin = next(v for v in units if v['prefix'] == u['prefix'] and v['checkpoint'] == u['checkpoint'] and v['condition'] == 'C0')
            pairs.append(dict(unit=u['id'], same_trace=[x['physics'] for x in traces[u['id']]] == [x['physics'] for x in traces[twin['id']]],
                              same_outcome=outcomes[u['id']] == outcomes[twin['id']]))
        reading['repeat_check'] = pairs
    write(root/'SUITE_READING.json', reading)
    write(root/'SUITE_COMPLETE.json', dict(passed=True, mode=plan['mode'], runs=len(runs), claims=reading['claims'], **provenance()))
    write(root/'HASHES.json', {n: sha(root/n) for n in ('PLAN.json', 'RUNS.jsonl.gz', 'SUITE_READING.json', 'SUITE_COMPLETE.json')})
    return reading


def native_success(suite, task_id, state):
    """A task's own LIBERO success check, in a headless environment of that task set to a recorded physics state."""
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    task = benchmark.get_benchmark_dict()[suite]().get_task(task_id)
    seed_globals()
    env = ControlEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file), use_camera_obs=False,
                     has_offscreen_renderer=False, camera_depths=False)
    try:
        env.seed(S.ENV_SEED); env.reset(); env.set_init_state(np.asarray(state))
        return bool(env.env._check_success())
    finally:
        env.close()


def state_check(root, limit=None):
    """Recorded final goal values against native checks: B in B's own scene-identical environment (groups 1a, 1b), A in A's."""
    configure(root); plan = read(root/'PLAN.json'); units = plan['branches']
    if limit is not None and len(units) > limit:
        units = units[::-(-len(units) // limit)]
    rows = []
    for u in units:
        row, trace = load(root/'branch'/u['id']); last = trace[-1]
        checks = [('A', u['task'], last['a'])]
        if u['condition'] == 'SW' and u['group'] in ('1a', '1b'):
            checks.append(('B', u['other'], last['others'][str(u['other'])]))
        for which, task, recorded in checks:
            native = native_success(u['suite'], task, row['final_state'])
            rows.append(dict(unit=u['id'], which=which, recorded=recorded, native=native))
    mismatches = [r for r in rows if r['recorded'] != r['native']]
    write(root/'STATE_CHECK.json', dict(passed=not mismatches, checked=len(rows), b_checks=sum(r['which'] == 'B' for r in rows),
                                        recorded_true=sum(r['recorded'] for r in rows), mismatches=mismatches, **provenance()))
    if mismatches:
        raise RuntimeError('Recorded goal values differ from native checks: %d' % len(mismatches))


def plan_branches(root):
    """In-run branch plan (smoke): the branch units of this run's own prefixes."""
    plan = read(root/'PLAN.json'); prefixes = read(root/'PREFIXES.json')
    units, excluded = S.branch_plan(read(root/'NAMES.json'), prefixes, plan['mode'])
    plan.update(prefix_root=str(root), prefix_order=plan['prefixes'], branches=units, excluded=excluded)
    write(root/'PLAN.json', plan)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path)
    p.add_argument('--phase', choices=['prefix', 'branch', 'probe']); p.add_argument('--shard', type=int); p.add_argument('--shards', type=int)
    p.add_argument('--index', type=int); p.add_argument('--port', type=int)
    p.add_argument('--summarize-prefix', action='store_true'); p.add_argument('--plan-branches', action='store_true')
    p.add_argument('--summarize', action='store_true'); p.add_argument('--smoke', action='store_true')
    p.add_argument('--state-check', action='store_true')
    a = p.parse_args()
    try:
        if a.summarize_prefix:
            summarize_prefix(a.root)
        elif a.plan_branches:
            plan_branches(a.root)
        elif a.summarize:
            summarize(a.root)
            if read(a.root/'PLAN.json').get('experiment', 'switch') == 'switch':
                state_check(a.root, limit=400)
        elif a.state_check:
            state_check(a.root)
        elif a.smoke:
            run(a.root, 'prefix', 0, 1, a.port); summarize_prefix(a.root); plan_branches(a.root); run(a.root, 'branch', 0, 1, a.port)
            summarize(a.root); state_check(a.root)
        elif a.index is not None:
            dict(prefix=run_prefix, branch=run_branch, probe=run_probe)[a.phase](a.root, a.index, a.port)
        else:
            run(a.root, a.phase, a.shard, a.shards, a.port)
    except BaseException as error:
        name = 'STOP_%s.json' % ('index_%s_%s' % (a.phase, a.index) if a.index is not None else
                                 'smoke' if a.smoke else 'summary' if (a.summarize or a.summarize_prefix or a.plan_branches or a.state_check)
                                 else 'shard_%s_%s' % (a.phase, a.shard))
        write(a.root/name, dict(error=type(error).__name__, reason=str(error)))
        raise
