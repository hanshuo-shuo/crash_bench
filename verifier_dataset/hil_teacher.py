"""Teacher data for the post-training step (docs/generative_verifier/HIL_TRAIN_PROTOCOL.md): stored runs, rendered again.

A stored run keeps the action and the physics hash of every step but no images. This module replays one in the LIBERO
container: the census prefix up to the switch step, then the run's own steps (the scripted retreat, if the run had one, and
the policy's). The physics hash is checked after every step, so a replay that differs in any way fails. Before each step it
stores what the policy would have been shown (the two 256 px camera images in the orientation of openpi's LIBERO data and
the 8-number state) together with the action taken.

A third kind, script, needs no stored run: a scripted answer to STOP, RETURN or CANCEL at a switch step of a prefix (scripted
below; HIL_TRAIN_PROTOCOL.md section 8).

An episode of ROOT/PLAN.json: dict(id, kind, source (the stored run's folder), prefix, suite, task, init, checkpoint,
prompt). kind switch is a run that took over at the switch step with prompt B; kind keep is the prefix itself, task A under
its own prompt from the end of the wait steps: the same pictures as the switch episodes with the instruction unchanged.
Output per episode: ROOT/episode/ID.npz (image, wrist_image, state, actions, raw_actions) and ROOT/episode/ID.json.

Runs in the container's Python 3.8; the plan is written by experiments/mechanism/teacher.py.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

from . import hil_suite as S
from .inventory import read, sha, write


def label(action):
    """The action as a training label. The controller clips the six arm numbers to [-1, 1] and the gripper reads only the
    sign of the seventh, so this is the same command in the form LIBERO's own demonstrations have."""
    return [min(1., max(-1., float(x))) for x in action[:6]] + [float((action[6] > 0) - (action[6] < 0))]


def state(obs):
    """The 8-number state of paired_policy.policy_input (position, axis-angle, the two finger joints)."""
    import numpy as np
    quat = np.array(obs['robot0_eef_quat'], dtype=np.float64); quat[3] = np.clip(quat[3], -1, 1)
    den = np.sqrt(1 - quat[3] * quat[3])
    axis = np.zeros(3) if math.isclose(den, 0) else quat[:3] * 2 * math.acos(quat[3]) / den
    return np.concatenate((obs['robot0_eef_pos'], axis, obs['robot0_gripper_qpos']))


def frame(obs):
    import numpy as np
    return dict(image=np.ascontiguousarray(obs['agentview_image'][::-1, ::-1]),
                wrist_image=np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1, ::-1]), state=state(obs))


def input_check(obs, stored):
    """The re-rendered policy input at the hand-over against the one stored by the original run: the state exactly, the
    images up to rendering noise (hil_suite.images_match: GPU rendering differs by a few levels on a few values between
    processes; a wrong scene or camera would change thousands)."""
    import numpy as np
    from .paired_policy import policy_input
    now = policy_input(obs, '')
    out = dict(state_equal=bool(np.array_equal(now['observation/state'], stored['observation/state'])),
               state_matches_frame=bool(np.array_equal(now['observation/state'], state(obs))))
    for key in ('observation/image', 'observation/wrist_image'):
        match, levels, values = S.images_match(now[key], stored[key])
        out[key.split('/')[1]] = dict(match=bool(match), linf=levels, changed=values)
    return out


def mine(index, shard, shards, worker, workers):
    """Episodes are dealt round-robin over shards * workers processes."""
    return index % (shards * workers) == shard * workers + worker


def export(root, plan, episode, physical_hash):
    import numpy as np
    from . import hil_suite_run as H
    started = time.monotonic()
    source = Path(episode['source']); row, trace = H.load(source)
    _, tape = H.load(Path(plan['prefix_root'])/'prefix'/episode['prefix'])
    if episode['kind'] == 'keep':
        if (source.name != episode['prefix'] or row['prompt'] != episode['prompt'] or episode['checkpoint'] != S.WAIT or not row['success']
                or not trace[-1]['a']):
            raise RuntimeError('The stored prefix is not the planned episode')
        trace = trace[S.WAIT:]
    elif (row['prompt'] != episode['prompt'] or row['checkpoint'] != episode['checkpoint'] or trace[0]['step'] != episode['checkpoint']
            or not trace[-1]['others'][str(row['other'])]):
        raise RuntimeError('The stored run is not the planned episode')
    scripted = sum(bool(x.get('scripted')) for x in trace)
    env, _, obs = H.make_env(episode['suite'], episode['task'], episode['init'])
    try:
        if physical_hash(env) != tape[0]['physics']:
            raise RuntimeError('Initial state differs from the prefix')
        for item in tape[1:episode['checkpoint'] + 1]:
            obs, _, _, _ = env.step(item['action'])
            if physical_hash(env) != item['physics']:
                raise RuntimeError('Replay differs from the prefix at step %d' % item['step'])
        if physical_hash(env) != trace[0]['physics']:
            raise RuntimeError('Switch state differs from the stored run')
        frames = []; check = None
        for k, item in enumerate(trace[1:]):
            if k == scripted:
                check = input_check(obs, np.load(source/'first_policy_input.npz'))
            frames.append(frame(obs))
            obs, _, _, _ = env.step(item['action'])
            if physical_hash(env) != item['physics']:
                raise RuntimeError('Replay differs from the stored run at step %d' % item['step'])
    finally:
        env.close()
    if check is None or not check['state_equal'] or not check['state_matches_frame']:
        raise RuntimeError('Policy input state differs from the stored run')
    if not (check['image']['match'] and check['wrist_image']['match']):
        raise RuntimeError('Policy input images differ from the stored run: %s' % json.dumps(check))
    folder = root/'episode'; raw = [item['action'] for item in trace[1:]]
    np.savez_compressed(folder/(episode['id'] + '.npz'), image=np.stack([f['image'] for f in frames]),
                        wrist_image=np.stack([f['wrist_image'] for f in frames]),
                        state=np.stack([f['state'] for f in frames]).astype(np.float32),
                        actions=np.asarray([label(a) for a in raw], np.float32), raw_actions=np.asarray(raw, np.float64))
    return dict(id=episode['id'], passed=True, frames=len(frames), scripted=scripted, input_check=check,
                clipped=sum(any(abs(x) > 1 for x in a[:6]) for a in raw), npz_sha256=sha(folder/(episode['id'] + '.npz')),
                elapsed_seconds=time.monotonic() - started, slurm_job=os.environ.get('SLURM_JOB_ID'), code_commit=os.environ.get('CB_CODE_COMMIT'))


def headless_env(suite, task_id, init):
    """hil_suite_run.make_env without cameras: the same physics, for runs that store no pictures."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    from . import hil_suite_run as H
    tasks = benchmark.get_benchmark_dict()[suite](); task = tasks.get_task(task_id)
    H.seed_globals()
    env = ControlEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file), use_camera_obs=False,
                     has_offscreen_renderer=False, camera_depths=False)
    env.seed(S.ENV_SEED); env.reset()
    return env, task, env.set_init_state(tasks.get_task_init_states(task_id)[init])


def scripted(root, plan, episode, physical_hash, render=True):
    """A scripted answer (hil_suite.Instruct) to STOP, RETURN or CANCEL at a switch step of a stored prefix. The prefix is
    replayed to the switch step, the script runs to its end, and the run is read with the suite's own rule
    (hil_suite.instruct_valid). Only a valid run is stored as an episode. A RETURN's retreat must end in the physics state of
    the stored home run of the same state. render False runs without cameras and stores no frames: the script as a baseline
    on states that are not exported."""
    import numpy as np
    from . import hil_suite_run as H
    started = time.monotonic(); c = episode['condition']; cp = episode['checkpoint']
    row, tape = H.load(Path(plan['prefix_root'])/'prefix'/episode['prefix'])
    if (not row['success'] or not S.WAIT <= cp < tape[-1]['step']
            or episode['prompt'] != dict(STOP=S.STOP, RETURN=S.RETURN, CANCEL=S.CANCEL % row['prompt'])[c]):
        raise RuntimeError('The stored prefix is not the planned episode')
    env, _, obs = (H.make_env if render else headless_env)(episode['suite'], episode['task'], episode['init'])
    try:
        e = env.env; d = env.sim.data; low, high = e.action_spec
        xyz = lambda n: np.asarray(d.body_xpos[e.obj_body_id[n]]).copy()
        eef = lambda o: [float(x) for x in o['robot0_eef_pos']]; quat = lambda o: [float(x) for x in o['robot0_eef_quat']]
        if physical_hash(env) != tape[0]['physics']:
            raise RuntimeError('Initial state differs from the prefix')
        for item in tape[1:cp + 1]:
            obs, _, _, _ = env.step(item['action'])
            if physical_hash(env) != item['physics']:
                raise RuntimeError('Replay differs from the prefix at step %d' % item['step'])
            if item['step'] == S.WAIT:   # the start pose and the heights objects rest at, as the branch runner reads them
                home = dict(eef=eef(obs), quat=quat(obs)); rest = {n: float(xyz(n)[2]) for n in e.objects_dict}
        moved, _ = S.goal_parts(e.parsed_problem['goal_state'])
        lifted = [n for n in moved if n in rest and xyz(n)[2] - rest[n] >= S.LIFTED]
        held = min(lifted, key=lambda n: float(np.linalg.norm(xyz(n) - np.asarray(eef(obs))))) if lifted else None
        lift = lambda: None if held is None else float(xyz(held)[2] - rest[held])
        grip = 1. if tape[cp]['action'][6] > 0 else -1.
        script = S.Instruct(c, home['eef'], home['quat'], eef(obs), quat(obs), tape[max(S.WAIT, cp - S.RETREAT_PARAMS['back_steps'])]['eef'], grip,
                            held is not None)
        trace = [H.record(env, cp, obs, None, False, physical_hash, {})]; frames = []; stages = []; lift0 = lift(); where = [None if held is None else xyz(held)]
        while True:
            action, stage = script.step(eef(obs), quat(obs), lift())
            if action is None:
                break
            if any(a < l or a > h for a, l, h in zip(action, low, high)):
                raise RuntimeError('Scripted action outside the action range')
            if render:
                frames.append(frame(obs))
            obs, _, done, _ = env.step(action); stages.append(stage)
            trace.append(H.record(env, cp + len(trace), obs, action, done, physical_hash, {}))
            where.append(None if held is None else xyz(held))
        joints = np.asarray(obs['robot0_joint_pos'])
        limits = env.sim.model.jnt_range[[env.sim.model.joint_name2id('robot0_joint%d' % k) for k in range(1, 8)]]
        w = S.INSTRUCT_PARAMS['still_window']; lift_end = lift()
        info = dict(status=stage, held=held, grip=grip, steps=len(trace) - 1, stages={n: stages.count(n) for n in dict.fromkeys(stages)},
                    motion_steps=sum(n != 'still' for n in stages) if c != 'STOP' else 0, lift_at_switch=lift0, release_lift=script.release_lift,
                    lift_end=lift_end, drop=None if script.release_lift is None else script.release_lift - lift_end,
                    arm_path=S.path_length(trace, trace[-1]['step'] - w, trace[-1]['step']),
                    object_path=None if held is None else float(sum(np.linalg.norm(q - p) for p, q in zip(where[-w - 1:], where[-w:]))),
                    joint_margin=float(np.min(np.minimum(joints - limits[:, 0], limits[:, 1] - joints))))
    finally:
        env.close()
    if c == 'RETURN' and episode.get('home_source'):
        _, stored = H.load(Path(episode['home_source'])); n = info['motion_steps']
        if (n >= len(stored) or not stored[n].get('scripted') or (n + 1 < len(stored) and stored[n + 1].get('scripted'))
                or stored[n]['physics'] != trace[n]['physics']):
            raise RuntimeError('The scripted retreat differs from the stored home run')
        info['matches_home_run'] = True
    o = S.outcome(trace, dict(other=None), row['start_eef']); valid = S.instruct_valid(c, o, info)
    raw = [item['action'] for item in trace[1:]]; digest = None
    if render and valid:
        path = root/'episode'/(episode['id'] + '.npz')
        np.savez_compressed(path, image=np.stack([f['image'] for f in frames]), wrist_image=np.stack([f['wrist_image'] for f in frames]),
                            state=np.stack([f['state'] for f in frames]).astype(np.float32),
                            actions=np.asarray([label(a) for a in raw], np.float32), raw_actions=np.asarray(raw, np.float64))
        digest = sha(path)
    return dict(id=episode['id'], passed=True, valid=valid, condition=c, frames=len(raw), scripted=len(raw), script=info,
                outcome={k: o[k] for k in ('stopped', 'stop_path', 'home', 'home_after', 'a_done', 'start_gap', 'closest_home')},
                clipped=sum(any(abs(x) > 1 for x in a[:6]) for a in raw), npz_sha256=digest, rendered=bool(render),
                elapsed_seconds=time.monotonic() - started, slurm_job=os.environ.get('SLURM_JOB_ID'), code_commit=os.environ.get('CB_CODE_COMMIT'))


def run(root, shard, shards, worker, workers):
    from . import hil_suite_run as H
    plan = read(root/'PLAN.json'); physical_hash = H.configure(root); (root/'episode').mkdir(exist_ok=True); done = failed = 0
    for index, episode in enumerate(plan['episodes']):
        if not mine(index, shard, shards, worker, workers):
            continue
        meta = root/'episode'/(episode['id'] + '.json')
        if meta.exists() and read(meta)['passed']:
            done += 1; continue
        try:
            info = (scripted(root, plan, episode, physical_hash, plan.get('render', True)) if episode['kind'] == 'script' else
                    export(root, plan, episode, physical_hash))
        except RuntimeError as error:
            info = dict(id=episode['id'], passed=False, error=str(error), slurm_job=os.environ.get('SLURM_JOB_ID')); failed += 1
        write(meta, info); done += 1
        print(json.dumps({k: info.get(k) for k in ('id', 'passed', 'valid', 'frames', 'error', 'elapsed_seconds')}), flush=True)
    write(root/('EXPORT_%d_%d_COMPLETE.json' % (shard, worker)), dict(shard=shard, worker=worker, episodes=done, failed=failed,
                                                                      slurm_job=os.environ.get('SLURM_JOB_ID')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path)
    for name in ('shard', 'shards', 'worker', 'workers'):
        p.add_argument('--' + name, type=int, required=True)
    a = p.parse_args(); run(a.root, a.shard, a.shards, a.worker, a.workers)
