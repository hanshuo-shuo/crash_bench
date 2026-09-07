#!/usr/bin/env python3
"""One bounded recovery of missing B cells after job 5680180's storage failure."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import pickle
import shutil
import socket
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.expansion.selection_retest import (TimedQueue, file_sha256, order, rollout,
                                              validate_records, write_json, resolve_git_head)
EXPECTED_FREEZE = 'd4bfbeb761b870b56737566cc0d8bfc13ec1422a97b97281a4653fa12ff345bf'


def missing_cells(anchors, records):
    expected = [(a['panel_id'], repeat, opt) for i, a in enumerate(anchors)
                for repeat in range(4, 8) for opt in order(i, repeat)]
    keys = [(r['panel_id'], r['repeat'], r['option']) for r in records]
    if len(keys) != len(set(keys)) or not set(keys).issubset(expected) or any(r['phase'] != 'B' for r in records):
        raise ValueError('invalid prior B outcomes')
    # An interrupted sequential job must have persisted an exact execution prefix.
    if set(keys) != set(expected[:len(keys)]):
        raise ValueError('prior B outcomes are not the expected execution prefix')
    return expected[len(keys):]


def validate_parent(parent):
    if file_sha256(parent/'freeze.json') != EXPECTED_FREEZE:
        raise ValueError('original A selection freeze changed')
    selection = json.loads((parent/'freeze.json').read_text())
    if (parent/'freeze.sha256').read_text().strip() != EXPECTED_FREEZE:
        raise ValueError('original freeze sidecar changed')
    for name, key in [('A.json', 'A_sha256'), ('anchors.json', 'anchors_sha256')]:
        if file_sha256(parent/name) != selection[key]:
            raise ValueError('frozen A input changed')
    anchors = json.loads((parent/'anchors.json').read_text())
    a = json.loads((parent/'A.json').read_text())
    validate_records(anchors, a, 'A')
    b = [json.loads(p.read_text()) for p in sorted(parent.glob('b*/B_r*_o*.json'))]
    missing = missing_cells(anchors, b)
    if len(anchors) != 24 or len(a) != 192 or len(b) != 139 or len(missing) != 53:
        raise ValueError('recovery scope changed: expected 192 A, 139 B, 53 missing')
    if missing[0] != ('b17', 5, 1):
        raise ValueError('unexpected interruption boundary')
    for row in a+b:
        path = parent/row['panel_id']/f"{row['phase']}_r{row['repeat']}_o{row['option']}.pkl.gz"
        if file_sha256(path) != row['trace_sha256']:
            raise ValueError(f'completed trace corrupt: {path}')
    for anchor in anchors:
        if file_sha256(parent/anchor['panel_id']/'bundle.pkl') != anchor['bundle_sha256']:
            raise ValueError('saved anchor changed')
    return anchors, b, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    parent, out = args.parent_dir.resolve(), args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    anchors, prior, missing = validate_parent(parent)
    checkpoint = json.loads((parent/'checkpoint.json').read_text())
    ckpt_root = Path(checkpoint['root'])
    for row in checkpoint['files']:
        if file_sha256(ckpt_root/row['path']) != row['sha256']:
            raise ValueError('original checkpoint changed')
    for name in ('A.json', 'anchors.json', 'exclusions.json', 'panel.json', 'freeze.json', 'freeze.sha256', 'checkpoint.json'):
        shutil.copyfile(parent/name, out/name)
    partial = parent/'b17/B_r5_o1.pkl.gz'
    provenance = {'kind': 'storage_failure_recovery', 'parent_dir': str(parent), 'parent_job': '5680180',
                  'job_id': os.getenv('SLURM_JOB_ID'), 'commit': resolve_git_head(ROOT),
                  'host': socket.gethostname(), 'pid': os.getpid(), 'A_freeze_sha256': EXPECTED_FREEZE,
                  'parent_completed_A': 192, 'parent_completed_B': 139, 'new_B_limit': 53,
                  'missing_cells': missing, 'partial_original_sha256': file_sha256(partial),
                  'partial_original_retained': str(partial), 'automatic_retry': False,
                  'limits': ['B spans two runtime processes/nodes; one repeat pair crosses job boundary',
                             'one interrupted unscored attempt replaced; no completed branch rerun']}
    write_json(out/'provenance.json', provenance)
    print('VALIDATED_PARENT: 331 completed traces, 24 bundles, frozen A and checkpoint hashes', flush=True)
    from crashbench.branching.state import restore_exact_state
    from crashbench.envs import LiberoEnv
    from crashbench.policies import build_policy
    policy = build_policy('pi0', pretrained_checkpoint=str(ckpt_root), config_name='pi0_libero', num_open_loop_steps=5)
    new = []
    for index, anchor in enumerate(anchors):
        cells = [cell for cell in missing if cell[0] == anchor['panel_id']]
        if not cells:
            continue
        folder = out/anchor['panel_id']
        folder.mkdir()
        with (parent/anchor['panel_id']/'bundle.pkl').open('rb') as stream:
            bundle, snapshot = pickle.load(stream)
        env = LiberoEnv('libero_spatial', int(anchor['task_id'].split(':')[1]), model_family='pi0', seed=0)
        try:
            queue = TimedQueue(anchor['delay_steps'])
            for key, repeat, option in cells:
                restore_exact_state(bundle, env, policy, restore_torch_rng=False)
                queue.restore(snapshot)
                name = f'B_r{repeat}_o{option}'
                path = folder/(name+'.pkl.gz')
                outcome = rollout(env, policy, queue, anchor, option, path)
                row = dict(outcome, panel_id=key, phase='B', repeat=repeat, option=option,
                           order_position=order(index, repeat).index(option), trace_sha256=file_sha256(path))
                write_json(folder/(name+'.json'), row)
                new.append(row)
                print(json.dumps(row), flush=True)
        finally:
            env.env.close()
    combined = prior+new
    validate_records(anchors, combined, 'B')
    if len(new) != 53:
        raise ValueError('recovery branch count mismatch')
    write_json(out/'B.json', combined)
    write_json(out/'trace_locations.json', [{**{k: row[k] for k in ('panel_id', 'repeat', 'option', 'trace_sha256')},
               'run_dir': str(parent if i < len(prior) else out.resolve())} for i, row in enumerate(combined)])
    write_json(out/'complete.json', {'status': 'COMPLETE', 'branches': 384, 'prior_branches': 331,
                                    'new_branches': 53, 'extra_incomplete_attempts': 1})


if __name__ == '__main__':
    main()
