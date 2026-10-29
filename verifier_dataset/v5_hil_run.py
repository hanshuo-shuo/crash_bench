"""V5 HIL runner: rebuild each twin state, then let pi0.5 continue under the four prompts.

Protocol: docs/generative_verifier/V5_HIL_PROTOCOL.md; contract in v5_hil.py. Every run starts in a
fresh process from the canonical initial state: place the counterfactual layout, replay the source
tape to the checkpoint t, and match the physics hash recorded at t (the S1a layout run's, or the
source tape's for a natural layout). Both rule labels and clearances are recomputed at t. The
policy RNG is reset to the state's seed, then pi0.5 (experiments/feasibility/serve.py on
127.0.0.1:PORT) is queried every five steps with the condition's prompt until success or step 300.
The conditions and prompts come from the plan (the HIL four when it names none); each run also
records the end-effector position at step 0, after the layout is placed and before the replay.

ROOT --shard K --shards N --port P runs every N-th state; ROOT --state-index I --condition C --port P
runs one execution; ROOT --summarize aggregates.
"""
import argparse
import collections
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .inventory import read, sha, write
from . import v5_geometry as g
from . import v5_hil as H
from . import v5_layouts as L


def lines_at(env, spec, objects, goal, material):
    """Each rule's line through its anchors' current centres, with the label and clearance it gives now."""
    out = {}
    for k in H.LINES:
        rule = spec['rules'][k]; ends = []
        for cat in rule['key'].split(':', 1)[1].split('|'):
            match = [n for n in objects if g.category(n) == cat]
            if len(match) != 1:
                raise RuntimeError('Anchor category is not unique: ' + cat)
            ends.append(objects[match[0]]['center'][:2])
        label, clearance = g.label(dict(template='line', params=ends), [(g.point(material), (goal['lo'], goal['hi']))])
        out[k] = dict(ends=ends, label=label, clearance=clearance)
    return out


def run_condition(root, index, condition, port):
    import numpy as np
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from .paired_scene import make_env
    from .paired_policy import policy_input
    from .replay import assert_trace
    from .v5_scan import goal_region, scene_objects
    case = root/('state_%03d' % index); spec = read(case/'STATE.json')
    if condition not in spec['prompts']:
        raise RuntimeError('Condition not in the plan: ' + condition)
    folder = case/condition; folder.mkdir(); prompt = spec['prompts'][condition]; t = spec['checkpoint']
    if spec['prompt_tokens'][condition] >= H.MAX_TOKENS:
        raise RuntimeError('Prompt would be truncated')
    env = client = None; started = time.monotonic()
    try:
        env, physical_hash, update_marker, source, context = make_env(case, folder, render=True, policy=True)
        from geometry import object_points
        from serve import digest
        if context['step'] != 0:
            raise RuntimeError('Replay must start from the canonical initial state')
        m, d = env.sim.model, env.sim.data; geom = m.geom_name2id(spec['spec']['material_geom'])
        layout = None
        if spec['layout'] is not None:
            layout = read(Path(spec['layout_case'])/'LAYOUT.json')
            if not layout['valid']:
                raise RuntimeError('Only valid layouts are used')
            L.place(env, layout)
            if physical_hash(env) != layout['placed_physics_sha256']:
                raise RuntimeError('Placed layout differs from the S1a layout run')
        start = np.asarray(d.site_xpos[env.env.robots[0].eef_site_id]).tolist()
        tape_path = Path(source['root'])/'steps.jsonl'
        if sha(tape_path) != source['tape_sha256']:
            raise RuntimeError('Original source tape changed')
        tape = [json.loads(x) for x in tape_path.read_text().splitlines()]
        assert_trace(tape, source['actions'])
        low, high = map(np.asarray, env.env.action_spec); obs = None; checked = 0
        for item in tape[:t]:
            obs, _, done, _ = env.step(np.clip(item['applied'], low, high).tolist())
            # The source tape hashed each step before the marker update (as v5_render does); S1a after it.
            if layout is None:
                if physical_hash(env) != item['physics_after_sha256'] or bool(done) != item['success']:
                    raise RuntimeError('Replay differs from the source at step %d' % item['step'])
                checked += 1
            update_marker(obs)
            if layout is not None and str(item['step']) in layout['checkpoint_physics']:
                if physical_hash(env) != layout['checkpoint_physics'][str(item['step'])]:
                    raise RuntimeError('Replay differs from the S1a layout run at step %d' % item['step'])
                checked += 1
        at = physical_hash(env)
        if layout is not None and at != layout['checkpoint_physics'][str(t)]:
            raise RuntimeError('State at the checkpoint differs from the S1a layout run')
        if layout is None and checked != t:
            raise RuntimeError('Natural replay did not check every step')
        if obs is None:
            obs = env.env._get_observations(force_update=True)
        objects, surface = scene_objects(env, object_points)
        goal = goal_region(env, objects, spec['spec']['receiver'])
        material = np.asarray(d.geom_xpos[geom]).tolist(); lift = max(0., material[2] - surface)
        lines = lines_at(env, spec, objects, goal, material)
        for k in H.LINES:
            if lines[k]['label'] != spec['rules'][k]['label'] or abs(lines[k]['clearance'] - spec['rules'][k]['clearance']) > 1e-6:
                raise RuntimeError('Twin label differs at the checkpoint: ' + k)
        if abs(lift - spec['lift']) > 1e-6:
            raise RuntimeError('Lift differs at the checkpoint')
        if physical_hash(env) != at:
            raise RuntimeError('Checkpoint readout mutated physics')

        def record(step, observation, action, done):
            p = np.asarray(d.geom_xpos[geom])
            return dict(step=step, eef=np.asarray(observation['robot0_eef_pos']).tolist(), material=p.tolist(),
                        lift=max(0., float(p[2]) - surface), gripper=np.asarray(observation['robot0_gripper_qpos']).tolist(),
                        command=None if action is None else float(action[6]),
                        d={k: g.line_distance(p[:2], *lines[k]['ends']) for k in H.LINES}, done=bool(done))
        run_id = '%s_%s_%s' % (os.environ['SLURM_JOB_ID'], spec['state'], condition)
        client = WebsocketClientPolicy('127.0.0.1', port)
        ack = client.infer({'__paired_reset_rng__': spec['seed'], 'run_id': run_id})
        if ack.get('seed') != spec['seed'] or ack.get('run_id') != run_id or ack.get('requests') != 0:
            raise RuntimeError('Policy RNG reset acknowledgement mismatch')
        trace = [record(t, obs, None, False)]; queue = collections.deque(); requests = []; images = []
        if t == 0 and np.linalg.norm(np.subtract(start, trace[0]['eef'])) > 1e-9:
            raise RuntimeError('Starting position differs from the observation at step 0')
        step, done, outside, first = t, False, 0, None
        log = (folder/'POLICY.jsonl').open('w')
        while step < H.HORIZON and not done:
            if not queue:
                data = policy_input(obs, prompt)
                if first is None:
                    first = dict(with_prompt=digest(data), state=digest({k: v for k, v in data.items() if k != 'prompt'}))
                    np.savez_compressed(folder/'first_policy_input.npz', **{k: v for k, v in data.items() if k != 'prompt'})
                result = client.infer(data); info = result['diagnostic']; actions = np.asarray(result['actions'])
                if (info['run_id'] != run_id or info['request_index'] != len(requests) + 1 or info['input_sha256'] != digest(data)
                        or actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < H.REPLAN or not np.isfinite(actions).all()):
                    raise RuntimeError('Policy stream, input or action shape mismatch')
                requests.append(dict(step=step, **{k: info[k] for k in ('request_index', 'input_sha256', 'rng_before', 'rng_after',
                                     'output_replayed', 'native_output_linf_difference', 'native_chunk_sha256')}))
                log.write(json.dumps(dict(requests[-1], actions=actions.tolist())) + '\n'); log.flush()
                images.append(data['observation/image']); queue.extend(actions[:H.REPLAN])
            action = queue.popleft()
            outside += bool(np.any(action < low) or np.any(action > high))
            obs, _, done, _ = env.step(action.tolist())
            update_marker(obs); step += 1
            trace.append(record(step, obs, action, done))
        log.close()
        with gzip.open(folder/'STEPS.jsonl.gz', 'wt') as stream:
            stream.writelines(json.dumps(x) + '\n' for x in trace)
        np.savez_compressed(folder/'policy_images.npz', agentview=np.stack(images))
        row = dict(state=spec['state'], condition=condition, image=spec['image'], prompt=prompt, prompt_tokens=spec['prompt_tokens'][condition],
                   seed=spec['seed'], checkpoint=t, horizon=H.HORIZON - t, start_eef=start, physics_at_checkpoint=at, checked_hashes=checked,
                   lines=lines, lift_at_checkpoint=lift, rng_reset=ack, policy_requests=len(requests),
                   first_inference_step=requests[0]['step'], first_policy_input_sha256=first['with_prompt'],
                   policy_input_state_sha256=first['state'], replayed_outputs=sum(r['output_replayed'] for r in requests),
                   maximum_native_difference=max(r['native_output_linf_difference'] for r in requests),
                   actions_outside_spec=outside, outcome=H.outcome(trace), elapsed_seconds=time.monotonic() - started,
                   api_calls=0, code_commit=os.environ.get('CB_CODE_COMMIT'), slurm_job=os.environ.get('SLURM_JOB_ID'))
        write(folder/'ROW.json', row)
        write(folder/'HASHES.json', {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file() and p.name != 'HASHES.json'})
    finally:
        if client is not None:
            client._ws.close()
        if env is not None:
            env.close()


def twin_check(case, conditions):
    """The runs of a state started from one observation, one starting position and one policy RNG key."""
    rows = [read(case/c/'ROW.json') for c in conditions]
    for field in ('physics_at_checkpoint', 'policy_input_state_sha256', 'first_inference_step', 'start_eef'):
        if len({json.dumps(r[field]) for r in rows}) != 1:
            raise RuntimeError('Twin runs differ in ' + field)
    if len({r['rng_reset']['key_hex'] for r in rows}) != 1:
        raise RuntimeError('Twin runs differ in the policy RNG key')
    if len({r['first_policy_input_sha256'] for r in rows}) != len(rows):
        raise RuntimeError('Two conditions sent the same first policy input')
    write(case/'COMPLETE.json', dict(passed=True, state=rows[0]['state'], conditions=list(conditions),
                                     elapsed_seconds={r['condition']: r['elapsed_seconds'] for r in rows},
                                     slurm_job=os.environ.get('SLURM_JOB_ID')))


def run(root, shard, shards, port):
    plan = read(root/'PLAN.json'); states = plan['states']; conditions = plan.get('conditions', H.CONDITIONS); done = 0
    for index, spec in enumerate(states):
        if index % shards != shard:
            continue
        case = root/('state_%03d' % index); case.mkdir()
        row = read(Path(spec['spec']['v4_case'])/'INPUT.json'); write(case/'INPUT.json', row)
        checkpoint = Path(row['evidence']['witness_root'])/'checkpoint_000'
        if row['evidence']['checkpoint_step'] != 0 or sha(checkpoint/'context.json') != row['evidence']['checkpoint_context_sha256']:
            raise RuntimeError('Original canonical checkpoint changed')
        shutil.copyfile(checkpoint/'context.json', case/'CANONICAL_CONTEXT.json')
        shutil.copyfile(checkpoint/'physics.npz', case/'CANONICAL_PHYSICS.npz')
        write(case/'STATE.json', spec)
        for condition in conditions:
            with (case/(condition + '.log')).open('w') as log:
                subprocess.run([sys.executable, '-u', '-m', 'verifier_dataset.v5_hil_run', str(root), '--state-index', str(index),
                                '--condition', condition, '--port', str(port)], stdout=log, stderr=subprocess.STDOUT, check=True)
        twin_check(case, conditions); done += 1
        write(root/('PROGRESS_%d.json' % shard), dict(shard=shard, states_done=done, last=spec['state'], slurm_job=os.environ.get('SLURM_JOB_ID')))
    write(root/('SHARD_%d_COMPLETE.json' % shard), dict(passed=True, shard=shard, states=done, slurm_job=os.environ.get('SLURM_JOB_ID')))


def load(root, index, condition):
    folder = root/('state_%03d' % index)/condition
    for name, digest in read(folder/'HASHES.json').items():
        if sha(folder/name) != digest:
            raise RuntimeError('Run output changed: %s' % (folder/name))
    with gzip.open(folder/'STEPS.jsonl.gz', 'rt') as stream:
        trace = [json.loads(x) for x in stream]
    row = read(folder/'ROW.json')
    if H.outcome(trace) != row['outcome']:
        raise RuntimeError('Recomputed outcome differs: %s' % folder)
    return row, trace


def summarize(root):
    campaign = read(root/'CAMPAIGN.json'); states = read(root/'PLAN.json')['states']
    for k in range(campaign['shards']):
        if not read(root/('SHARD_%d_COMPLETE.json' % k))['passed']:
            raise RuntimeError('Shard %d incomplete' % k)
    outcomes, divergences, runs = {}, {}, []
    for index, spec in enumerate(states):
        if not read(root/('state_%03d' % index)/'COMPLETE.json')['passed']:
            raise RuntimeError('State %d incomplete' % index)
        loaded = {c: load(root, index, c) for c in H.CONDITIONS}
        outcomes[spec['state']] = {c: row['outcome'] for c, (row, _) in loaded.items()}
        divergences[spec['state']] = dict(C2_C3=H.divergence(loaded['C2'][1], loaded['C3'][1]),
                                          C0_C1=H.divergence(loaded['C0'][1], loaded['C1'][1]))
        runs += [dict(state=spec['state'], condition=c, base_source=spec['base_source'], time_bin=spec['time_bin'], phase=spec['phase'],
                      elapsed_seconds=row['elapsed_seconds'], policy_requests=row['policy_requests'], steps=row['outcome']['steps'],
                      replayed_outputs=row['replayed_outputs'], maximum_native_difference=row['maximum_native_difference'],
                      actions_outside_spec=row['actions_outside_spec'], **{k: row['outcome'][k] for k in ('completed', 'crossed_C2', 'crossed_C3', 'grasped')})
                 for c, (row, _) in loaded.items()]
    with gzip.open(root/'RUNS.jsonl.gz', 'wt') as stream:
        stream.writelines(json.dumps(r) + '\n' for r in runs)
    reading = H.readouts(states, outcomes, divergences)
    reading.update(mode=campaign['mode'], seconds_per_run=H.distribution(r['elapsed_seconds'] for r in runs),
                   total_run_hours=sum(r['elapsed_seconds'] for r in runs) / 3600., prompt_tokens=read(root/'PLAN.json')['prompt_tokens'],
                   replayed_outputs=sum(r['replayed_outputs'] for r in runs), actions_outside_spec=sum(r['actions_outside_spec'] for r in runs),
                   maximum_native_difference=max(r['maximum_native_difference'] for r in runs))
    write(root/'HIL_READING.json', reading)
    write(root/'HIL_COMPLETE.json', dict(passed=True, mode=campaign['mode'], reading=reading['reading'], timing_matters=reading['timing_matters'],
                                         states=len(states), runs=len(runs), api_calls=0, code_commit=os.environ.get('CB_CODE_COMMIT'),
                                         slurm_job=os.environ.get('SLURM_JOB_ID')))
    write(root/'HASHES.json', {n: sha(root/n) for n in ('PLAN.json', 'RUNS.jsonl.gz', 'HIL_READING.json', 'HIL_COMPLETE.json')})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path)
    p.add_argument('--shard', type=int); p.add_argument('--shards', type=int); p.add_argument('--port', type=int)
    p.add_argument('--state-index', type=int); p.add_argument('--condition')
    p.add_argument('--summarize', action='store_true')
    a = p.parse_args()
    try:
        if a.summarize:
            summarize(a.root)
        elif a.state_index is not None:
            run_condition(a.root, a.state_index, a.condition, a.port)
        else:
            run(a.root, a.shard, a.shards, a.port)
    except BaseException as error:
        name = ('STOP_summarize.json' if a.summarize else 'STOP_state_%s_%s.json' % (a.state_index, a.condition)
                if a.state_index is not None else 'STOP_shard_%s.json' % a.shard)
        write(a.root/name, dict(error=type(error).__name__, reason=str(error)))
        raise
