#!/usr/bin/env python3
"""Disentangle observation drift from policy/physics drift on saved probe anchors."""
from __future__ import annotations
import argparse,hashlib,json,os,pickle,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from crashbench.branching.state import restore_exact_state
from crashbench.branching.artifacts import pack_numeric_mapping
from crashbench.mechanisms.observation_staleness import ObservationDelayQueue
from scripts.expansion.run_staleness_screen import mechanism_observation
from scripts.expansion.probe_neutral_refresh import trace_difference
from scripts.expansion.hash_tree_manifest import resolve_git_head


def observation_differences(a,b):
    if set(a)!=set(b):raise ValueError('observation field identity differs')
    out={}
    for k in sorted(a):
        x,y=np.asarray(a[k]),np.asarray(b[k])
        if x.shape!=y.shape:raise ValueError('observation field shape differs')
        delta=abs(x.astype(np.float64)-y.astype(np.float64))
        out[k]={'max_abs':float(delta.max()),'mean_abs':float(delta.mean()),'changed_elements':int(np.count_nonzero(delta)),
                'elements':delta.size}
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--steps',type=int,default=35);args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    from crashbench.envs import LiberoEnv
    from crashbench.policies import build_policy
    ledger=json.loads((args.bundle_root/'output_sha256.json').read_text());args.output_dir.mkdir(parents=True)
    policy=build_policy('pi0',pretrained_checkpoint='gs://openpi-assets/checkpoints/pi0_libero',config_name='pi0_libero',num_open_loop_steps=5)
    report={'kind':'observation_replay_engineering_probe','git_commit':resolve_git_head(ROOT),'job_id':os.environ.get('SLURM_JOB_ID'),
            'engineering_only':True,'test_rows_read':0,'anchors':[]}
    for task in (0,2):
        name=f'task{task}_fresh_control/bundle_and_queue.pkl';path=args.bundle_root/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==ledger[name]
        bundle,queue_state=pickle.loads(path.read_bytes())
        env=LiberoEnv('libero_spatial',task,model_family='pi0',seed=0);queue=ObservationDelayQueue(0)
        row={'task_id':task,'bundle_id':bundle.bundle_id,'bundle_sha256':ledger[name],'comparisons':{},'observation_differences':[]}
        records={};reference=None
        for mode in ('live_a','live_b','replay_a'):
            restore_exact_state(bundle,env,policy);queue.restore_state(queue_state);obs=dict(env.sim_view._obs)
            trace=[];delivered_list=[];numeric={}
            for step in range(args.steps):
                fresh=env.policy_observation(obs,policy.resize_size)
                actual=mechanism_observation(queue,fresh,condition='fresh_control')
                delivered=actual if mode!='replay_a' else {k:np.asarray(v).copy() for k,v in reference[step].items()}
                if mode=='live_b':row['observation_differences'].append({'step':step,'fields':observation_differences(reference[step],actual)})
                delivered_list.append({k:np.asarray(v).copy() for k,v in delivered.items()})
                for k,v in actual.items():numeric[f'{step}:actual:{k}']=np.asarray(v).copy()
                obs_hash=hashlib.sha256(pack_numeric_mapping(delivered)).hexdigest();phash=policy.snapshot_continuation().sha256()
                action=np.asarray(policy.act(delivered,env.task_description)).copy();obs,_,done,_=env.step(action.tolist())
                force=float(env.sim_view.max_contact_force(env.sim_view._robot_bodies()))
                trace.append({'step':step,'observation_sha256':obs_hash,'policy_continuation_sha256':phash,
                              'action':action.tolist(),'state':env.flat_state().tolist(),'force':force})
                if done or env.episode_terminated():break
            if mode=='live_a':reference=delivered_list
            records[mode]=trace
            (args.output_dir/f'task{task}_{mode}.json').write_text(json.dumps(trace,sort_keys=True)+'\n')
            np.savez_compressed(args.output_dir/f'task{task}_{mode}_observations.npz',**numeric)
        row['comparisons']={mode:trace_difference(records['live_a'],records[mode]) for mode in ('live_b','replay_a')}
        report['anchors'].append(row);print(json.dumps(row),flush=True)
        (args.output_dir/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
        close=getattr(env.env,'close',None)
        if close is not None:close()
    report['status']='COMPLETE_ENGINEERING_DIAGNOSTIC';(args.output_dir/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    hashes={str(p.relative_to(args.output_dir)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output_dir.rglob('*')) if p.is_file()}
    (args.output_dir/'output_sha256.json').write_text(json.dumps(hashes,indent=2,sort_keys=True)+'\n')

if __name__=='__main__':main()
