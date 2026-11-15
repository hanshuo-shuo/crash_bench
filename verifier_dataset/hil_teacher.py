"""Teacher data for the post-training step (docs/generative_verifier/HIL_TRAIN_PROTOCOL.md): stored runs, rendered again.

A stored run keeps the action and the physics hash of every step but no images. This module replays one in the LIBERO
container: the census prefix up to the switch step, then the run's own steps (the scripted retreat, if the run had one, and
the policy's). The physics hash is checked after every step, so a replay that differs in any way fails. Before each step it
stores what the policy would have been shown (the two 256 px camera images in the orientation of openpi's LIBERO data and
the 8-number state) together with the action taken.

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
    """The re-rendered policy input at the hand-over against the one stored by the original run."""
    import numpy as np
    from .paired_policy import policy_input
    now = policy_input(obs, '')
    out = dict(state_equal=bool(np.array_equal(now['observation/state'], stored['observation/state'])),
               state_matches_frame=bool(np.array_equal(now['observation/state'], state(obs))))
    for key in ('observation/image', 'observation/wrist_image'):
        gap = np.abs(now[key].astype(np.int16) - stored[key].astype(np.int16))
        out[key.split('/')[1]] = dict(linf=int(gap.max()), changed=float((gap > 0).mean()))
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
    folder = root/'episode'; raw = [item['action'] for item in trace[1:]]
    np.savez_compressed(folder/(episode['id'] + '.npz'), image=np.stack([f['image'] for f in frames]),
                        wrist_image=np.stack([f['wrist_image'] for f in frames]),
                        state=np.stack([f['state'] for f in frames]).astype(np.float32),
                        actions=np.asarray([label(a) for a in raw], np.float32), raw_actions=np.asarray(raw, np.float64))
    return dict(id=episode['id'], passed=True, frames=len(frames), scripted=scripted, input_check=check,
                clipped=sum(any(abs(x) > 1 for x in a[:6]) for a in raw), npz_sha256=sha(folder/(episode['id'] + '.npz')),
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
            info = export(root, plan, episode, physical_hash)
        except RuntimeError as error:
            info = dict(id=episode['id'], passed=False, error=str(error), slurm_job=os.environ.get('SLURM_JOB_ID')); failed += 1
        write(meta, info); done += 1
        print(json.dumps({k: info.get(k) for k in ('id', 'passed', 'frames', 'error', 'elapsed_seconds')}), flush=True)
    write(root/('EXPORT_%d_%d_COMPLETE.json' % (shard, worker)), dict(shard=shard, worker=worker, episodes=done, failed=failed,
                                                                      slurm_job=os.environ.get('SLURM_JOB_ID')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path)
    for name in ('shard', 'shards', 'worker', 'workers'):
        p.add_argument('--' + name, type=int, required=True)
    a = p.parse_args(); run(a.root, a.shard, a.shards, a.worker, a.workers)
