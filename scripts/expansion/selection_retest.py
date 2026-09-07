#!/usr/bin/env python3
"""Bounded train-only closed-loop repeat panel; all A decisions freeze before B."""
from __future__ import annotations
import argparse
import copy
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import socket
import sys
import time
from collections import deque

import numpy as np
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from crashbench.mechanisms.observation_staleness import ObservationDelayQueue
from crashbench.branching.rng import capture_rng_state
from scripts.expansion.hash_tree_manifest import file_sha256, resolve_git_head, tree_manifest
from scripts.expansion.run_fragile_screen import classify_terminal

STRATA = ((5, 1), (10, 3), (15, 5), (5, 3), (10, 5), (15, 1))
RULES = ('Base', 'AlwaysRefresh', 'age1', 'age3', 'age5', 'period10')
SPLIT = ROOT / 'results/expansion/governance/split_manifest_v1_1.json'
FORMAL = ROOT / 'results/expansion/d3_formal_nominal/05ddc6a3a7c3_acb3d2d28dc2_20260830T103120Z/formal_nominal_analysis.json'


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')


def select_panel(split):
    rows = [r for r in split['assignments'] if r['role'] == 'train']
    tasks = sorted({r['task_id'] for r in rows})
    if tasks != ['libero_spatial:0', 'libero_spatial:2']:
        raise ValueError('unexpected training tasks')
    result = []
    for task in tasks:
        sources = sorted((r for r in rows if r['task_id'] == task), key=lambda r: r['physical_source_id'])
        if len(sources) < 6 or len({r['physical_source_id'] for r in sources}) != len(sources):
            raise ValueError('training sources missing or duplicated')
        for source, (steps, delay) in zip(sources[:6], STRATA):
            for condition in ('stale', 'matched_buffer_control'):
                result.append(dict(source, anchor_steps=steps, delay_steps=delay,
                                   condition=condition, panel_id=f'b{len(result):02d}'))
    return result


class TimedQueue:
    """Sidecar frame timestamps; preserve the existing mechanism operations exactly."""
    def __init__(self, delay):
        self.queue = ObservationDelayQueue(delay)
        self.frames = deque(maxlen=delay + 1)

    def deliver(self, fresh, condition, step, sim_time, refresh=False):
        frame = {'step': int(step), 'sim_time': float(sim_time)}
        if refresh:
            obs = self.queue.refresh(fresh)
            self.frames.clear()
            self.frames.append(frame)
        else:
            obs = self.queue.push(fresh)
            self.frames.append(frame)
            if condition == 'matched_buffer_control':
                obs = self.queue.refresh(fresh)
                self.frames.clear()
                self.frames.append(frame)
            elif condition != 'stale':
                raise ValueError(condition)
        delivered = self.frames[0]
        return obs, dict(current=frame, delivered=delivered,
                         age_steps=step-delivered['step'], age_seconds=sim_time-delivered['sim_time'])

    def snapshot(self):
        return copy.deepcopy((self.queue.snapshot_state(), tuple(self.frames)))

    def restore(self, snapshot):
        self.queue.restore_state(snapshot[0])
        self.frames.clear()
        self.frames.extend(copy.deepcopy(snapshot[1]))


def choice(rule, anchor):
    if rule == 'Base':
        return 0
    if rule == 'AlwaysRefresh':
        return 1
    if rule.startswith('age'):
        return int(anchor['age_steps'] >= int(rule[3:]))
    if rule == 'period10':
        return int(anchor['anchor_steps'] % 10 == 0)
    raise ValueError(rule)


def order(panel_index, repeat):
    return (0, 1) if (panel_index + repeat) % 2 == 0 else (1, 0)


def validate_records(anchors, records, phase):
    repeats = range(4) if phase == 'A' else range(4, 8)
    expected = {(a['panel_id'], r, o) for a in anchors for r in repeats for o in (0, 1)}
    keys = [(r['panel_id'], r['repeat'], r['option']) for r in records]
    if len(keys) != len(set(keys)) or set(keys) != expected or any(r['phase'] != phase for r in records):
        raise ValueError('incomplete, duplicate or wrong-phase outcomes')


def freeze(anchors, records):
    validate_records(anchors, records, 'A')
    def score(mapping, selected):
        used = [r for r in records if r['panel_id'] in selected and r['option'] == mapping[r['panel_id']]]
        return (np.mean([r['task_success'] for r in used]),
                -np.mean([r['catastrophe'] for r in used]),
                -np.mean([r['option'] for r in used]))
    ids = {a['panel_id'] for a in anchors}
    maps = {name: {a['panel_id']: choice(name, a) for a in anchors} for name in RULES}
    reference = {}
    for anchor in anchors:
        key = anchor['panel_id']
        reference[key] = max((0, 1), key=lambda opt: score({key: opt}, {key}))
    best_fixed = max(RULES[:2], key=lambda name: score(maps[name], ids))
    best_simple = max(RULES, key=lambda name: score(maps[name], ids))
    return {'reference': reference, 'rules': maps, 'best_fixed': best_fixed, 'best_simple': best_simple,
            'selection_phase': 'A', 'tie_break': 'success, fewer accidents, fewer refreshes, rule order'}


def sim_time(env):
    return float(env.sim_view._live_mj()[1].time)


def step_record(env, policy, queue, obs, condition, step, refresh):
    """Observe real infer calls, never call act a second time for logging."""
    fresh = env.policy_observation(obs, policy.resize_size)
    before = policy.snapshot_continuation()
    rng_sources = env.rng_sources() if hasattr(env, 'rng_sources') else {}
    rng_before = capture_rng_state(numpy_generators=rng_sources.get('generators', {}),
                                   numpy_random_states=rng_sources.get('random_states', {}),
                                   capture_torch=False)
    queue_before = queue.snapshot()
    delivered, timing = queue.deliver(fresh, condition, step, sim_time(env), refresh)
    if refresh:
        policy.reset()
    inference = []
    original = policy._policy.infer
    def recorded_infer(element, *args, **kwargs):
        inputs = copy.deepcopy(element)
        result = original(element, *args, **kwargs)
        inference.append({'input': inputs, 'output': copy.deepcopy(result)})
        return result
    policy._policy.infer = recorded_infer
    try:
        action = np.asarray(policy.act(delivered, env.task_description)).copy()
    finally:
        policy._policy.infer = original
    record = {'step': step, 'timing': timing, 'fresh': fresh, 'delivered': delivered,
              'instruction': env.task_description, 'queue_before': queue_before,
              'rng_before': rng_before,
              'queue_after': queue.snapshot(), 'policy_before': before,
              'policy_after': policy.snapshot_continuation(), 'inference': inference,
              'chunk_remaining_before': len(before.payload['queue']),
              'nominal_proposal': before.payload['queue'][0] if before.payload['queue'] else None,
              'nominal_proposal_status': 'buffered' if before.payload['queue'] else 'unavailable',
              'action': action, 'controller_before': env.controller_state()}
    obs, _, done, _ = env.step(action.tolist())
    force = float(env.sim_view.max_contact_force(env.sim_view._robot_bodies()))
    record.update(state_after=env.flat_state().copy(), force=force, done=bool(done),
                  terminated=bool(env.episode_terminated()))
    return obs, record


def rollout(env, policy, queue, anchor, option, path):
    obs = dict(env.sim_view._obs)
    positions = [np.asarray(obs['robot0_eef_pos']).copy()]
    start = time.monotonic()
    with gzip.open(path, 'xb', compresslevel=1) as stream:
        for step in range(100):
            obs, record = step_record(env, policy, queue, obs, anchor['condition'],
                                      anchor['anchor_steps']+step, bool(option and step == 0))
            pickle.dump(record, stream, protocol=5)
            positions.append(np.asarray(obs['robot0_eef_pos']).copy())
            if record['force'] >= 75 or record['done'] or record['terminated']:
                break
    accident = record['force'] >= 75
    outcome = classify_terminal(task_success=record['done'], catastrophe=accident,
                                intervention=bool(option), steps=step+1,
                                path_length=float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()))
    outcome.update(raw_done=record['done'], elapsed_seconds=time.monotonic()-start,
                   termination_reason=('accident' if accident else 'success' if record['done'] else
                                       'environment_termination' if record['terminated'] else 'timeout'))
    return outcome


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    out = args.output_dir
    split = json.loads(SPLIT.read_text())
    panel = select_panel(split)
    write_json(out/'panel.json', panel)  # Outcome-independent selection before model/environment work.
    write_json(out/'checkpoint.json', tree_manifest(args.checkpoint_dir, 'pi0_libero'))
    from crashbench.branching.state import capture_exact_state, restore_exact_state
    from crashbench.envs import LiberoEnv
    from crashbench.policies import build_policy
    from crashbench.glass_recovery_data import array_sha256
    from scripts.expansion.collect_staleness_statewise import source_by_physical_id
    import jax
    versions = {}
    for name in ('jax', 'jaxlib', 'numpy', 'mujoco', 'robosuite', 'openpi'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not registered'
    write_json(out/'provenance.json', {'commit': resolve_git_head(ROOT), 'job_id': os.getenv('SLURM_JOB_ID'),
               'host': socket.gethostname(), 'pid': os.getpid(), 'versions': versions,
               'python': sys.version, 'jax_devices': [str(d) for d in jax.devices()],
               'timestamp_semantics': 'queue enqueue simulator time; not sensor exposure time',
               'input_sha256': {str(p.relative_to(ROOT)): file_sha256(p) for p in (SPLIT, FORMAL)},
               'historical_D5_replay': False, 'test_outcomes_read': 0, 'max_branches': 384,
               'bundle_reuse': 'new anchors: old probes lack synchronized per-frame timing'})
    formal = json.loads(FORMAL.read_text())
    policy = build_policy('pi0', pretrained_checkpoint=str(args.checkpoint_dir),
                          config_name='pi0_libero', num_open_loop_steps=5)
    initial = policy.snapshot_continuation()
    anchors, exclusions = [], []
    for row in panel:
        source = source_by_physical_id(formal, row['physical_source_id'])
        state = np.asarray(source['source_state'], dtype=np.dtype(source['source_state_dtype']))
        if source['attempt_id'] != row['formal_attempt_id'] or array_sha256(state) != row['source_state_sha256']:
            raise ValueError('source identity mismatch')
        env = LiberoEnv('libero_spatial', int(source['task_id']), model_family='pi0', seed=int(source['reset_seed']))
        folder = out/row['panel_id']
        folder.mkdir()
        try:
            policy.restore_continuation(initial)
            env.seed(int(source['reset_seed']))
            obs = env.reset_to_exact(state)
            queue = TimedQueue(row['delay_steps'])
            reason = None
            with gzip.open(folder/'generation.pkl.gz', 'xb', compresslevel=1) as stream:
                for warm in range(10):
                    obs, _, done, _ = env.step(env.dummy_action())
                    force = float(env.sim_view.max_contact_force(env.sim_view._robot_bodies()))
                    pickle.dump({'warmup': warm, 'state': env.flat_state(), 'force': force, 'done': bool(done)}, stream)
                    if done or env.episode_terminated() or force >= 75:
                        reason = 'warmup_termination_or_accident'
                        break
                if reason is None:
                    for step in range(row['anchor_steps']):
                        obs, record = step_record(env, policy, queue, obs, row['condition'], step, False)
                        pickle.dump(record, stream, protocol=5)
                        if record['done'] or record['terminated'] or record['force'] >= 75:
                            reason = 'preanchor_termination_or_accident'
                            break
            if reason:
                exclusions.append(dict(row, reason=reason))
                continue
            bundle = capture_exact_state(env, policy, identity={'source_id': row['physical_source_id'],
                    'task_id': row['task_id'], 'anchor_id': row['panel_id'], 'policy_id': 'pi0',
                    'mechanism_id': 'observation_staleness_v1', 'trajectory_id': 'selection_retest'},
                    provenance={'train_only': True}, declared_branch_seed=0, capture_torch_rng=False)
            if bundle.policy_continuation.payload['queue']:
                raise ValueError('anchor not on action-chunk boundary')
            snapshot = queue.snapshot()
            # Age query operates only on a copied queue and never calls the policy.
            probe = TimedQueue(row['delay_steps'])
            probe.restore(snapshot)
            delivered, timing = probe.deliver(env.policy_observation(obs, policy.resize_size), row['condition'],
                                              row['anchor_steps'], sim_time(env))
            anchor = dict(row, age_steps=timing['age_steps'], age_seconds=timing['age_seconds'], bundle_id=bundle.bundle_id)
            with (folder/'bundle.pkl').open('xb') as stream:
                pickle.dump((bundle, snapshot), stream, protocol=5)
            with (folder/'selector_context.pkl').open('xb') as stream:
                pickle.dump({'delivered': delivered, 'timing': timing, 'instruction': env.task_description,
                             'chunk_remaining': 0, 'nominal_proposal': None}, stream, protocol=5)
            anchor['bundle_sha256'] = file_sha256(folder/'bundle.pkl')
            anchors.append(anchor)
        finally:
            env.env.close()
    write_json(out/'anchors.json', anchors)
    write_json(out/'exclusions.json', exclusions)
    if not anchors:
        raise RuntimeError('no valid anchors; no substitution allowed')
    for phase, repeats in [('A', range(4)), ('B', range(4, 8))]:
        if phase == 'B':
            if file_sha256(out/'freeze.json') != (out/'freeze.sha256').read_text().strip():
                raise ValueError('A selection freeze changed')
        records = []
        for index, anchor in enumerate(anchors):
            bundle_path = out/anchor['panel_id']/'bundle.pkl'
            if file_sha256(bundle_path) != anchor['bundle_sha256']:
                raise ValueError('bundle file changed')
            with bundle_path.open('rb') as stream:
                bundle, snapshot = pickle.load(stream)
            env = LiberoEnv('libero_spatial', int(anchor['task_id'].split(':')[1]), model_family='pi0', seed=0)
            try:
                queue = TimedQueue(anchor['delay_steps'])
                for repeat in repeats:
                    for position, option in enumerate(order(index, repeat)):
                        restore_exact_state(bundle, env, policy, restore_torch_rng=False)
                        queue.restore(snapshot)
                        name = f'{phase}_r{repeat}_o{option}'
                        trace = out/anchor['panel_id']/(name+'.pkl.gz')
                        outcome = rollout(env, policy, queue, anchor, option, trace)
                        row = dict(outcome, panel_id=anchor['panel_id'], phase=phase, repeat=repeat,
                                   option=option, order_position=position, trace_sha256=file_sha256(trace))
                        write_json(out/anchor['panel_id']/(name+'.json'), row)
                        records.append(row)
                        print(json.dumps(row), flush=True)
            finally:
                env.env.close()
        validate_records(anchors, records, phase)
        write_json(out/(phase+'.json'), records)
        if phase == 'A':
            selection = freeze(anchors, records)
            selection['A_sha256'] = file_sha256(out/'A.json')
            selection['anchors_sha256'] = file_sha256(out/'anchors.json')
            write_json(out/'freeze.json', selection)
            (out/'freeze.sha256').write_text(file_sha256(out/'freeze.json')+'\n')
    write_json(out/'complete.json', {'status': 'COMPLETE', 'branches': len(anchors)*16})


if __name__ == '__main__':
    main()
