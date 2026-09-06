#!/usr/bin/env python3
"""Training-source exact-branch repeatability and neutral Refresh diagnostic."""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from crashbench.branching.state import capture_exact_state, restore_exact_state
from crashbench.branching.artifacts import pack_numeric_mapping
from crashbench.mechanisms.observation_staleness import ObservationDelayQueue
from crashbench.options.common import ObservationRefreshOption
from scripts.expansion.collect_staleness_statewise import source_by_physical_id
from scripts.expansion.run_staleness_screen import mechanism_observation
from scripts.expansion.run_fragile_screen import classify_terminal
from scripts.expansion.hash_tree_manifest import resolve_git_head


def digest(payload):return hashlib.sha256(payload).hexdigest()

def trace_difference(a,b):
    n=min(len(a),len(b));first={};max_delta={}
    for field in ('observation_sha256','policy_continuation_sha256','action','state','force'):
        different=[];deltas=[]
        for i in range(n):
            if field.endswith('sha256'):bad=a[i][field]!=b[i][field]
            else:
                delta=float(np.max(np.abs(np.asarray(a[i][field])-np.asarray(b[i][field]))));deltas.append(delta);bad=delta>0
            if bad:different.append(i)
        first[field]=None if not different else different[0]
        if deltas:max_delta[field]=max(deltas)
    return {'length_a':len(a),'length_b':len(b),'first_differing_step_0based':first,
            'maximum_absolute_difference':max_delta,'identical':len(a)==len(b) and all(x is None for x in first.values())}


def rollout(env,policy,queue,condition,refresh,max_steps):
    obs=dict(env.sim_view._obs);trace=[];path=[np.asarray(obs['robot0_eef_pos']).copy()]
    invoked=False;success=catastrophe=False
    for step in range(max_steps):
        fresh=env.policy_observation(obs,policy.resize_size)
        if refresh and not invoked:
            option=ObservationRefreshOption();view=option.spec.view({'policy_observation':fresh,'observation_queue':queue})
            delivered=option.refresh(view);policy.reset();invoked=True
        else:delivered=mechanism_observation(queue,fresh,condition=condition)
        obs_hash=digest(pack_numeric_mapping(delivered));policy_hash=policy.snapshot_continuation().sha256()
        action=np.asarray(policy.act(delivered,env.task_description)).copy()
        obs,_,done,_=env.step(action.tolist())
        force=float(env.sim_view.max_contact_force(env.sim_view._robot_bodies()))
        path.append(np.asarray(obs['robot0_eef_pos']).copy())
        trace.append({'step':step,'observation_sha256':obs_hash,'policy_continuation_sha256':policy_hash,
                      'action':action.tolist(),'state':env.flat_state().tolist(),'force':force})
        catastrophe=force>=75.;success=bool(done)
        if catastrophe or done or env.episode_terminated():break
    length=float(np.linalg.norm(np.diff(np.asarray(path),axis=0),axis=1).sum())
    return trace,classify_terminal(task_success=success,catastrophe=catastrophe,intervention=refresh,steps=len(trace),path_length=length)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--max-steps',type=int,default=100);args=parser.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    from crashbench.envs import LiberoEnv
    from crashbench.policies import build_policy
    split_path=ROOT/'results/expansion/governance/split_manifest_v1_1.json'
    formal_path=ROOT/'results/expansion/d3_formal_nominal/05ddc6a3a7c3_acb3d2d28dc2_20260830T103120Z/formal_nominal_analysis.json'
    split=json.loads(split_path.read_text());formal=json.loads(formal_path.read_text())
    bytask=defaultdict(list)
    for row in split['assignments']:
        if row['role']=='train':bytask[row['task_id']].append(row)
    selected=[sorted(rows,key=lambda r:r['physical_source_id'])[0] for task,rows in sorted(bytask.items())]
    if len(selected)!=2:raise ValueError('expected two existing train tasks')
    args.output_dir.mkdir(parents=True)
    import jax
    report={'kind':'neutral_refresh_engineering_probe','git_commit':resolve_git_head(ROOT),
            'job_id':os.environ.get('SLURM_JOB_ID'),'jax_version':jax.__version__,
            'engineering_only':True,'historical_anchor_replay':False,'test_rows_read':0,
            'source_selection':'first lexicographic train source per task','selected':selected,
            'input_sha256':{str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in [split_path,formal_path]},'anchors':[]}
    policy=build_policy('pi0',pretrained_checkpoint='gs://openpi-assets/checkpoints/pi0_libero',config_name='pi0_libero',num_open_loop_steps=5)
    initial_policy=policy.snapshot_continuation()
    for assignment in selected:
        source=source_by_physical_id(formal,assignment['physical_source_id']);task=int(source['task_id']);seed=int(source['reset_seed'])
        state=np.asarray(source['source_state'],dtype=np.dtype(source['source_state_dtype']))
        env=LiberoEnv('libero_spatial',task,model_family='pi0',seed=seed)
        for condition in ('fresh_control','matched_buffer_control','stale'):
            policy.restore_continuation(initial_policy);env.seed(seed);obs=env.reset_to_exact(state)
            queue=ObservationDelayQueue(0 if condition=='fresh_control' else 3)
            for _ in range(10):obs,_,_,_=env.step(env.dummy_action())
            for _ in range(5):
                fresh=env.policy_observation(obs,policy.resize_size)
                action=policy.act(mechanism_observation(queue,fresh,condition=condition),env.task_description)
                obs,_,done,_=env.step(np.asarray(action).tolist())
                if done or env.episode_terminated():raise RuntimeError('training source terminated before the fixed anchor')
            bundle=capture_exact_state(env,policy,identity={'source_id':assignment['physical_source_id'],'policy_id':'pi0',
                'mechanism_id':'observation_staleness_v1','task_id':assignment['task_id'],'trajectory_id':'neutral_probe',
                'anchor_id':condition+':5'},provenance={'engineering_only':True},declared_branch_seed=0)
            snapshot=queue.snapshot_state();anchor_dir=args.output_dir/f'task{task}_{condition}';anchor_dir.mkdir()
            # This new diagnostic preserves the full state and queue, not just IDs.
            (anchor_dir/'bundle_and_queue.pkl').write_bytes(pickle.dumps((bundle,snapshot),protocol=5))
            records={};outcomes={}
            for name,refresh in [('base_a',False),('base_b',False),('refresh_a',True),('refresh_b',True)]:
                restore_exact_state(bundle,env,policy);queue.restore_state(snapshot)
                records[name],outcomes[name]=rollout(env,policy,queue,condition,refresh,args.max_steps)
                (anchor_dir/(name+'.json')).write_text(json.dumps({'outcome':outcomes[name],'trace':records[name]},sort_keys=True)+'\n')
            comparisons={f'{a}_vs_{b}':trace_difference(records[a],records[b]) for a,b in [('base_a','base_b'),('refresh_a','refresh_b'),('base_a','refresh_a')]}
            row={'task_id':task,'condition':condition,'source_id':assignment['physical_source_id'],'bundle_id':bundle.bundle_id,
                 'anchor_policy_queue_length':len(bundle.policy_continuation.payload['queue']),
                 'outcomes':outcomes,'comparisons':comparisons}
            report['anchors'].append(row);(args.output_dir/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
            print(json.dumps(row),flush=True)
        close = getattr(env.env, "close", None)
        if close is not None:
            close()
    report['status']='COMPLETE_ENGINEERING_DIAGNOSTIC'
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    ledger={str(p.relative_to(args.output_dir)):digest(p.read_bytes()) for p in sorted(args.output_dir.rglob('*')) if p.is_file()}
    (args.output_dir/'output_sha256.json').write_text(json.dumps(ledger,indent=2,sort_keys=True)+'\n')

if __name__=='__main__':main()
