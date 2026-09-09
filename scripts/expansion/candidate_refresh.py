#!/usr/bin/env python3
"""Historical-success training mechanism panel: 18 anchors, <=288 branches."""
from __future__ import annotations
import argparse
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import socket
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.expansion.selection_retest import (TimedQueue, SPLIT, FORMAL, write_json,
    order, validate_records, freeze, sim_time, step_record)
from scripts.expansion.hash_tree_manifest import file_sha256, resolve_git_head, tree_manifest
from scripts.expansion.run_fragile_screen import classify_terminal
PANEL = ROOT / 'docs/audits/20260908/candidate_refresh/panel.json'


def select_panel(split):
    manifest = json.loads(PANEL.read_text())
    for rel, digest in manifest['historical_input_sha256'].items():
        if file_sha256(ROOT / rel) != digest:
            raise ValueError('historical input changed')
    rows = manifest['panel']
    train = {r['physical_source_id']: r for r in split['assignments'] if r['role'] == 'train'}
    if len(rows) != 18 or len({r['physical_source_id'] for r in rows}) != 8:
        raise ValueError('candidate budget mismatch')
    for r in rows:
        if any(r[k] != v for k, v in train[r['physical_source_id']].items()):
            raise ValueError('training identity mismatch')
    return rows


def at_horizon(records, horizon):
    return [dict(r, **r['horizons'][str(horizon)]) for r in records]


def freeze_horizons(anchors, records):
    return {str(h): freeze(anchors, at_horizon(records, h)) for h in (100, 200)}


def rollout(env, policy, queue, anchor, option, path):
    obs = dict(env.sim_view._obs)
    positions = [np.asarray(obs['robot0_eef_pos']).copy()]
    start = time.monotonic()
    horizons = {}
    calls = 0
    with gzip.open(path, 'xb', compresslevel=1) as stream:
        for step in range(200):
            obs, record = step_record(env, policy, queue, obs, anchor['condition'],
                                      anchor['anchor_steps']+step, bool(option and step == 0))
            pickle.dump(record, stream, protocol=5)
            calls += len(record['inference'])
            positions.append(np.asarray(obs['robot0_eef_pos']).copy())
            accident = record['force'] >= 75
            terminal = accident or record['done'] or record['terminated']
            if step+1 in (100, 200) or terminal:
                outcome = classify_terminal(task_success=record['done'], catastrophe=accident,
                    intervention=bool(option), steps=step+1,
                    path_length=float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()))
                outcome.update(raw_done=record['done'], inference_calls=calls,
                    elapsed_seconds=time.monotonic()-start,
                    first_success_step=step+1 if outcome['task_success'] else None,
                    termination_reason=('accident' if accident else 'success' if record['done'] else
                                        'environment_termination' if record['terminated'] else 'timeout'))
                for h in (100, 200):
                    if h not in horizons and (step+1 == h or terminal and step+1 <= h):
                        horizons[h] = dict(outcome)
            if terminal:
                break
    return dict(horizons[200], horizons={str(h): v for h,v in horizons.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    out = args.output_dir
    split = json.loads(SPLIT.read_text())
    panel = select_panel(split)
    write_json(out/'panel.json', panel)  # Historical-outcome-selected census frozen before any new execution.
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
               'input_sha256': {str(p.relative_to(ROOT)): file_sha256(p) for p in (SPLIT, FORMAL, PANEL)},
               'historical_D5_replay': False, 'test_outcomes_read': 0, 'max_branches': 288, 'horizons': [100, 200],
               'bundle_reuse': 'new anchors: historical D5 complete bundles unavailable'})
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
        raw_horizon = int(getattr(env._raw_env(), 'horizon', 0))
        if raw_horizon < 10 + row['anchor_steps'] + 200:
            env.env.close()
            raise ValueError(f'environment horizon too short: {raw_horizon}')
        write_json(folder/'environment.json', {'horizon': raw_horizon, 'reset_seed': int(source['reset_seed'])})
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
                    'mechanism_id': 'observation_staleness_v1', 'trajectory_id': 'candidate_refresh'},
                    provenance={'train_only': True, 'commit': resolve_git_head(ROOT), 'job_id': os.getenv('SLURM_JOB_ID')}, declared_branch_seed=0, capture_torch_rng=False)
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
                             'chunk_remaining': 0, 'nominal_proposal': None, 'fresh': env.policy_observation(obs, policy.resize_size), 'policy_continuation': policy.snapshot_continuation()}, stream, protocol=5)
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
            selection = freeze_horizons(anchors, records)
            selection['A_sha256'] = file_sha256(out/'A.json')
            selection['anchors_sha256'] = file_sha256(out/'anchors.json')
            write_json(out/'freeze.json', selection)
            (out/'freeze.sha256').write_text(file_sha256(out/'freeze.json')+'\n')
    write_json(out/'complete.json', {'status': 'COMPLETE', 'branches': len(anchors)*16})


if __name__ == '__main__':
    main()
