"""One prospective C execution block from frozen existing bundles, no new anchors."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.expansion.hash_tree_manifest import file_sha256, resolve_git_head
from scripts.expansion.selection_retest import write_json, order


def read(path):
    return json.loads(Path(path).read_text())


def verify_checkpoint(checkpoint, manifest_path):
    manifest = read(manifest_path)
    rows = manifest['files'] if isinstance(manifest, dict) else manifest
    for row in rows:
        p = checkpoint / row.get('path', row.get('file'))
        if not p.is_file() or p.stat().st_size != row.get('size_bytes', row.get('bytes')):
            raise ValueError('checkpoint file missing or size changed: '+str(p))
        if file_sha256(p) != row['sha256']:
            raise ValueError('checkpoint hash mismatch: '+str(p))
    return len(rows)


def check_records(anchors, records, repeats, panel):
    key = 'episode_id' if panel == 'detour' else 'panel_id'
    expected = {(a[key], r, o) for a in anchors if a.get('triggered', True)
                for r in repeats for o in (0, 1)}
    actual = [(r[key], r['repeat'], r['option']) for r in records]
    if len(actual) != len(set(actual)) or set(actual) != expected or any(r['phase'] != 'C' for r in records):
        raise ValueError('C block incomplete, duplicated, or wrong phase')


def detour(args, spec, anchors):
    from crashbench.envs import LiberoEnv
    from crashbench.policies import OpenVLAPolicy
    from scripts.expansion.run_detour_benefit import branch
    config = read(ROOT / 'configs/detour_benefit/development_v1.json')
    if config != read(Path(spec['source_root']) / 'config.json'):
        raise ValueError('historical Detour configuration differs')
    # Reuse B's override contract; new numeric indices identify C, not new policy randomness.
    config['policy_seed_override']['4'] = config['policy_seed_override']['2']
    config['policy_seed_override']['5'] = config['policy_seed_override']['3']
    write_json(args.output / 'execution_config.json', config)
    rows = {r['episode_id']: r for r in config['panel']}
    policy = OpenVLAPolicy(pretrained_checkpoint=config['checkpoint'],
                           checkpoint_revision=config['checkpoint_revision'], capture_hidden=True)
    write_json(args.output / 'policy_identity.json', policy.checkpoint_identity)
    records = []
    env = LiberoEnv('libero_spatial', 0, seed=config['seed'])
    try:
        for i, anchor in enumerate(anchors):
            if not anchor['triggered']:
                continue
            folder = args.output / anchor['episode_id']
            folder.mkdir()
            source = Path(spec['source_root']) / anchor['episode_id'] / 'bundle.pkl'
            if file_sha256(source) != anchor['bundle_sha256']:
                raise ValueError('Detour bundle mismatch')
            (folder / 'bundle.pkl').symlink_to(source)
            for repeat in spec['repeats']:
                for option in order(i, repeat):
                    r = branch(env, policy, rows[anchor['episode_id']], anchor, config,
                               folder, repeat, option, phase='C')
                    records.append(r)
                    print('completed', anchor['episode_id'], repeat, option, 'branches', len(records), flush=True)
    finally:
        env.env.close()
    return records


def refresh(args, spec, anchors):
    from crashbench.envs import LiberoEnv
    from crashbench.policies import build_policy
    from crashbench.branching.state import restore_exact_state
    from scripts.expansion.selection_retest import TimedQueue
    from scripts.expansion.candidate_refresh import rollout
    policy = build_policy('pi0', pretrained_checkpoint=str(args.checkpoint_dir),
                          config_name='pi0_libero', num_open_loop_steps=5)
    records = []
    for i, anchor in enumerate(anchors):
        source = Path(spec['source_root']) / anchor['panel_id'] / 'bundle.pkl'
        if file_sha256(source) != anchor['bundle_sha256']:
            raise ValueError('Refresh bundle mismatch')
        with source.open('rb') as stream:
            bundle, snapshot = pickle.load(stream)
        folder = args.output / anchor['panel_id']
        folder.mkdir()
        (folder / 'bundle.pkl').symlink_to(source)
        env = LiberoEnv('libero_spatial', int(anchor['task_id'].split(':')[1]), model_family='pi0', seed=0)
        try:
            if int(env._raw_env().horizon) < 10+anchor['anchor_steps']+200:
                raise ValueError('environment cannot support original suffix budget')
            queue = TimedQueue(anchor['delay_steps'])
            for repeat in spec['repeats']:
                for position, option in enumerate(order(i, repeat)):
                    restore_exact_state(bundle, env, policy, restore_torch_rng=False)
                    queue.restore(snapshot)
                    name = f'C_r{repeat}_o{option}'
                    trace = folder / (name+'.pkl.gz')
                    r = dict(rollout(env, policy, queue, anchor, option, trace), panel_id=anchor['panel_id'],
                        phase='C', repeat=repeat, option=option, order_position=position,
                        bundle_sha256=anchor['bundle_sha256'], trace_sha256=file_sha256(trace))
                    write_json(folder / (name+'.json'), r)
                    records.append(r)
                    print('completed', anchor['panel_id'], repeat, option, 'branches', len(records), flush=True)
        finally:
            env.env.close()
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--panel', choices=['detour', 'candidate_refresh'], required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run only in the authorized Quest Slurm allocation')
    contract_path = ROOT / 'docs/audits/20260913/repeat_value/c_contract.json'
    contract = read(contract_path)
    for rel, digest in contract['input_sha256'].items():
        if file_sha256(ROOT / rel) != digest:
            raise ValueError('frozen input changed: '+rel)
    spec = contract['panels'][args.panel]
    source = Path(spec['source_root'])
    if file_sha256(source/'anchors.json') != spec['historical_anchor_sha256']:
        raise ValueError('historical anchors changed')
    anchors = read(source / 'anchors.json')
    if len(anchors) != spec['expected_anchors']:
        raise ValueError('anchor coverage mismatch')
    # Verify every input state before producing any new branch; no outcome-based substitution.
    for anchor in anchors:
        if anchor.get('triggered', True):
            key = anchor.get('episode_id', anchor.get('panel_id'))
            if file_sha256(source/key/'bundle.pkl') != anchor['bundle_sha256']:
                raise ValueError('bundle mismatch: '+key)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'contract.json', contract)
    write_json(args.output / 'anchors.json', anchors)
    checked = verify_checkpoint(args.checkpoint_dir, ROOT/spec['checkpoint_manifest'])
    versions = {}
    for name in ['numpy', 'torch', 'jax', 'jaxlib', 'transformers', 'mujoco', 'robosuite', 'libero']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    write_json(args.output / 'provenance.json', dict(panel=args.panel, commit=resolve_git_head(ROOT),
        job=os.environ['SLURM_JOB_ID'], host=platform.node(), python=sys.version, versions=versions,
        contract_sha256=file_sha256(contract_path), checkpoint_files_verified=checked,
        new_sources=0, new_prefixes=0, max_branches=spec['max_branches']))
    print('all bundles and checkpoint verified; starting fixed C block', flush=True)
    records = detour(args, spec, anchors) if args.panel == 'detour' else refresh(args, spec, anchors)
    check_records(anchors, records, spec['repeats'], args.panel)
    if len(records) != spec['max_branches']:
        raise ValueError('unexpected branch count')
    write_json(args.output / 'C.json', records)
    write_json(args.output / 'complete.json', dict(status='COMPLETE', branches=len(records),
        contract_sha256=file_sha256(contract_path), C_sha256=file_sha256(args.output/'C.json')))
    print('COMPLETE', len(records), flush=True)


if __name__ == '__main__':
    main()
