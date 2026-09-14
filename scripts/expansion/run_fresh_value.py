"""Bounded fresh-reset authoring and three separately launched execution blocks."""
import argparse
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crashbench.fresh_value import (
    PHASE_REPEATS, freeze_a, score_choices, summarize_ab, validate_anchors, validate_records,
)
from scripts.expansion.hash_tree_manifest import file_sha256, resolve_git_head
from scripts.expansion.run_detour_benefit import write, generate, branch, measured_act, step
from scripts.expansion.run_repeat_value_c import verify_checkpoint

CONTRACT = ROOT/'docs/audits/20260913/fresh_value/contract.json'
OLD_FREEZE = ROOT/'docs/audits/20260909/detour_benefit_gate/evidence/freeze.json'
CHECKPOINT_MANIFEST = ROOT/'docs/audits/20260909/detour_benefit_gate/evidence/checkpoint_hashes.json'


def read(path):
    return json.loads(Path(path).read_text())


def verify_run(run):
    contract = read(CONTRACT)
    if read(run/'contract.json') != contract:
        raise ValueError('prospective contract changed')
    for rel, digest in contract['input_sha256'].items():
        if file_sha256(ROOT/rel) != digest:
            raise ValueError('frozen code or input changed: '+rel)
    return contract


def record_runtime(run, stage, gpu):
    versions = {}
    for name in ('numpy', 'torch', 'transformers', 'mujoco', 'robosuite', 'libero'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    info = dict(stage=stage, commit=resolve_git_head(ROOT), job=os.environ['SLURM_JOB_ID'],
        host=platform.node(), pid=os.getpid(), python=sys.version, versions=versions,
        started_unix=time.time(), contract_sha256=file_sha256(CONTRACT))
    if gpu:
        import torch
        info.update(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
            checkpoint_files_verified=verify_checkpoint(Path(os.environ['CB_CHECKPOINT_PATH']), CHECKPOINT_MANIFEST))
    write(run/(stage+'_provenance.json'), info)


def policy_for(config):
    from crashbench.policies import OpenVLAPolicy
    return OpenVLAPolicy(pretrained_checkpoint=config['checkpoint'],
        checkpoint_revision=config['checkpoint_revision'], capture_hidden=True)


def prepare(run, contract):
    from crashbench.envs import LiberoEnv
    from crashbench.glass_recovery_data import array_sha256
    from crashbench.counterfactual_router import FrozenOutcomeRouter
    from scripts.expansion.run_formal_nominal import scene_fingerprint
    from scripts.capture_glass_detector_placements import _seed_everything
    from scripts.prepare_glass_recovery_placements import _sample_trace_anchor, _glass
    from scripts.collect_glass_recovery_pairs import TARGET
    config = contract['config']
    env = LiberoEnv(contract['suite'], contract['task_id'], seed=config['seed'])
    sources = []
    states = []
    excluded = {k: set(v) for k, v in contract['excluded_identities'].items()}
    (run/'sources').mkdir()
    try:
        # Materialize and isolate the entire fixed source list before any VLA rollout.
        for attempt in contract['source_attempts']:
            folder = run/'sources'/attempt['source_id']; folder.mkdir()
            obs = env.reset_fresh(attempt['reset_seed'])
            state = env.flat_state().copy()
            source_hash, fingerprint = array_sha256(state), scene_fingerprint(obs)
            if not np.isfinite(state).all():
                raise ValueError('non-finite fresh state')
            for key, value in [('source_state_sha256', source_hash), ('scene_fingerprint', fingerprint)]:
                if value in excluded[key]:
                    raise ValueError('source identity already seen: '+key)
                excluded[key].add(value)
            np.save(folder/'state.npy', state, allow_pickle=False)
            (folder/'model.xml').write_text(env.model_xml())
            record = dict(attempt, source_state_sha256=source_hash, scene_fingerprint=fingerprint,
                state_file_sha256=file_sha256(folder/'state.npy'), model_sha256=file_sha256(folder/'model.xml'),
                object_positions={k: np.asarray(v).tolist() for k, v in obs.items()
                                  if k.endswith('_pos') and not k.startswith('robot0_')})
            write(folder/'identity.json', record)
            sources.append(record); states.append(state)
        write(run/'source_identities.json', sources)
        policy = policy_for(config)
        write(run/'policy_identity.json', policy.checkpoint_identity)
        panel, placements, authoring = [], [], []
        g = contract['geometry']
        # Trace collection supplies geometry only; successes and failures all remain.
        for source, state in zip(sources, states):
            folder = run/'sources'/source['source_id']
            _seed_everything(config['seed']); env.seed(config['seed']); policy.reset()
            obs = env.reset_to(state)
            for _ in range(config['settle_steps']):
                obs, _, _, _ = env.step(env.dummy_action())
            bowl = np.asarray(obs[TARGET+'_pos']).copy()
            xyz, events = [], []
            with gzip.open(folder/'nominal.pkl.gz', 'xb', compresslevel=1) as stream:
                for index in range(contract['authoring_horizon']):
                    xyz.append(np.asarray(obs['robot0_eef_pos']).copy())
                    action, po, latency = measured_act(env, policy, obs, env.task_description)
                    obs, event = step(env, obs, action, None, [], index, 1, False, stream, po, latency)
                    events.append(event)
                    if event['reason']: break
            np.save(folder/'nominal_eef.npy', np.asarray(xyz), allow_pickle=False)
            anchor_xy, direction, fraction = _sample_trace_anchor(
                np.asarray(xyz), source['requested_fraction'], bowl[:2], g['min_target_clearance'])
            table_top = float(bowl[2]-g['bowl_rest_offset'])
            on = _glass('glass_1', anchor_xy, table_top, g['size'], g['density'])
            off_xy = anchor_xy + source['offpath_side']*g['offpath_offset']*np.array([-direction[1], direction[0]])
            off = _glass('glass_1', off_xy, table_top, g['size'], g['density'])
            placement = dict(placement_id='fresh_'+source['source_id'], task_suite=contract['suite'],
                task_id=contract['task_id'], instruction=env.task_description,
                source_state_path=f"sources/{source['source_id']}/state.npy",
                source_state_sha256=source['source_state_sha256'], on_path_glass=on,
                off_path_glass=off, nominal_fraction=fraction)
            placements.append(placement)
            authoring.append(dict(source_id=source['source_id'], events=events,
                trace_sha256=file_sha256(folder/'nominal.pkl.gz'),
                nominal_success=bool(events[-1]['reason']=='success'), retained=True,
                requested_fraction=source['requested_fraction'], actual_fraction=fraction,
                target_clearance=float(np.linalg.norm(anchor_xy-bowl[:2]))))
            for condition in contract['conditions']:
                panel.append(dict(episode_id=f'n{len(panel):02d}', source=source['source_state_sha256'],
                    source_id=source['source_id'], reset_seed=source['reset_seed'],
                    role='fresh_development', condition=condition,
                    placement_manifest=str(run/'placements.json'), placement=placement))
            print('authored', source['source_id'], 'source retained', flush=True)
        write(run/'placements.json', dict(placements=placements))
        write(run/'authoring.json', authoring)
        write(run/'panel.json', panel)
        # All new source geometries now frozen, before paired recovery outcomes.
        router = FrozenOutcomeRouter.load(ROOT/config['risk_path'])
        (run/'anchors').mkdir()
        anchors = []
        for row in panel:
            folder = run/'anchors'/row['episode_id']; folder.mkdir()
            anchors.append(generate(env, policy, router, row, config, folder))
            print('prefix complete', row['episode_id'], flush=True)
        validate_anchors(anchors)
        write(run/'anchors.json', anchors)
        write(run/'prepare_complete.json', dict(status='COMPLETE', sources=len(sources),
            prefixes=len(anchors), candidates=sum(a['triggered'] for a in anchors),
            paired_branches=0, input_sha256={p: file_sha256(run/p) for p in (
                'source_identities.json', 'placements.json', 'authoring.json', 'panel.json', 'anchors.json')}))
    except Exception as exc:
        write(run/'prepare_failure.json', dict(error=repr(exc), sources_materialized=len(sources),
            paired_branches=0, replacement_authorized=False))
        raise
    finally:
        env.env.close()


def execute(run, contract, phase):
    from crashbench.envs import LiberoEnv
    done = read(run/'prepare_complete.json')
    for rel, digest in done['input_sha256'].items():
        if file_sha256(run/rel) != digest: raise ValueError('authored input changed: '+rel)
    if phase == 'A':
        forbidden = ['A.json', 'freeze_A.json', 'B.json', 'forecast_AB.json', 'C.json']
    elif phase == 'B':
        forbidden = ['B.json', 'forecast_AB.json', 'C.json']
        frozen = read(run/'freeze_A.json')
        if frozen['A_sha256'] != file_sha256(run/'A.json'):
            raise ValueError('A outcomes changed after choice lock')
    else:
        forbidden = ['C.json']
        forecast = read(run/'forecast_AB.json')
        for rel, digest in forecast['input_sha256'].items():
            if file_sha256(run/rel) != digest: raise ValueError('pre-C input changed: '+rel)
    if any((run/p).exists() for p in forbidden):
        raise ValueError('execution stage already exists or was opened out of order')
    anchors, panel = read(run/'anchors.json'), read(run/'panel.json')
    validate_anchors(anchors)
    for anchor in anchors:
        if anchor['triggered'] and file_sha256(run/'anchors'/anchor['episode_id']/'bundle.pkl') != anchor['bundle_sha256']:
            raise ValueError('candidate bundle changed')
    ids = list(range(len(anchors)))
    if phase == 'B': ids = ids[::-1]
    if phase == 'C': ids = ids[12:]+ids[:12]
    (run/phase).mkdir()
    config = contract['config']
    policy = policy_for(config)
    env = LiberoEnv(contract['suite'], contract['task_id'], seed=config['seed'])
    records = []
    try:
        for i in ids:
            anchor, row = anchors[i], panel[i]
            if not anchor['triggered']: continue
            folder = run/phase/row['episode_id']; folder.mkdir()
            (folder/'bundle.pkl').symlink_to(run/'anchors'/row['episode_id']/'bundle.pkl')
            for repeat in PHASE_REPEATS[phase]:
                for option in ((0,1) if (i+repeat)%2==0 else (1,0)):
                    records.append(branch(env, policy, row, anchor, config, folder,
                        repeat, option, audit_controller=True, phase=phase))
                    print(phase, 'completed branches', len(records), flush=True)
        validate_records(anchors, records, phase)
        write(run/(phase+'.json'), records)
        write(run/(phase+'_complete.json'), dict(status='COMPLETE', branches=len(records),
            records_sha256=file_sha256(run/(phase+'.json'))))
    finally:
        env.env.close()


def lock_a(run):
    if any((run/p).exists() for p in ('B_provenance.json', 'B.json', 'C_provenance.json', 'C.json')):
        raise ValueError('choice freeze must precede all B/C execution')
    anchors, a = read(run/'anchors.json'), read(run/'A.json')
    choices = freeze_a(anchors, a, read(OLD_FREEZE)['info'])
    write(run/'freeze_A.json', dict(choices=choices, frozen_unix=time.time(),
        A_sha256=file_sha256(run/'A.json'), anchors_sha256=file_sha256(run/'anchors.json'),
        selection_block='A', future_outcomes_opened=0))
    print('A choices locked; no B/C execution yet', flush=True)


def lock_ab(run):
    if any((run/p).exists() for p in ('C_provenance.json', 'C.json')):
        raise ValueError('forecast freeze must precede all C execution')
    frozen = read(run/'freeze_A.json')
    if frozen['A_sha256'] != file_sha256(run/'A.json'):
        raise ValueError('A changed')
    rows = score_choices(read(run/'anchors.json'), {p:read(run/(p+'.json')) for p in ('A','B')}, frozen['choices'])
    write(run/'forecast_AB.json', dict(forecasts=summarize_ab(rows), frozen_unix=time.time(),
        input_sha256={p:file_sha256(run/p) for p in ('anchors.json','A.json','B.json','freeze_A.json')},
        C_outcomes_opened=0))
    print('A/B source forecasts locked; no C execution yet', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare','A','freeze_A','B','freeze_AB','C'], required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('All scientific execution runs on Quest in Slurm')
    run = args.run_dir.resolve()
    if args.stage == 'prepare':
        run.mkdir(parents=True, exist_ok=False)
        write(run/'contract.json', read(CONTRACT))
    contract = verify_run(run)
    record_runtime(run, args.stage, args.stage in ('prepare','A','B','C'))
    if args.stage == 'prepare': prepare(run, contract)
    elif args.stage in PHASE_REPEATS: execute(run, contract, args.stage)
    elif args.stage == 'freeze_A': lock_a(run)
    else: lock_ab(run)


if __name__ == '__main__': main()
