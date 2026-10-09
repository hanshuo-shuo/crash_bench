"""Independent complete-shard metric reduction and Arrow readback; no fitting."""
import argparse
import json
from pathlib import Path
from common import atomic_json, churn, full_schedule, norm, outcome, sha, time_to_crash


def export(root):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from full import check_tensors
    plan=json.loads((root/'plan.json').read_text()); cfg=plan['configuration']
    complete=root/'COMPUTE_COMPLETE.json'
    progress=json.loads((complete if complete.exists() else root/'PARTIAL.json').read_text())
    results=progress['results']; expected=list(full_schedule(cfg,plan['shard']))
    if [x['spec'] for x in results]!=expected[:len(results)]:
        raise RuntimeError('Completed cases do not match frozen schedule')
    inference_count=check_tensors(root,results); rows=[]; action_count=0
    for result in results:
        path=root/'runs'/result['spec']['name']
        actions=[json.loads(x) for x in (path/'steps.jsonl').read_text().splitlines()]
        inferences=[json.loads(x) for x in (path/'inferences.jsonl').read_text().splitlines()]
        r=[json.loads(x) for x in (path/'rows.jsonl').read_text().splitlines()]
        if len(actions)!=result['actions'] or len(r)!=len(actions):raise RuntimeError('Incomplete action evidence')
        collision=next((x['step'] for x in actions if x['obstacle_l1_m']>.001),None)
        if collision!=result['collision_step'] or result['outcome']!=outcome(result['success'],collision):
            raise RuntimeError('Official outcome reduction differs')
        scales=None; previous=None; chunks={}
        for info in inferences:
            with np.load(path/('infer_%03d.npz'%info['infer_index']),allow_pickle=False) as sample:
                chunk=sample['actions']; scales=sample['normalization_scales'].tolist()
                value=churn(chunk,previous,scales)
                if value is None and info['churn'] is not None or value is not None and abs(value-info['churn'])>1e-12:
                    raise RuntimeError('Shifted churn differs')
                previous=chunk.copy();chunks[info['infer_index']]=chunk
        for action,row in zip(actions,r):
            t=action['step']; chunk=chunks[action['infer_index']]
            if not np.array_equal(action['raw'],chunk[(t-1)%5]):raise RuntimeError('Applied queue trace differs')
            if abs(norm(action['applied'],scales)-row['act_norm'])>1e-12:raise RuntimeError('Applied norm differs')
            if row['crashed']!=(collision is not None and t>=collision) or row['time_to_crash']!=time_to_crash(t,collision):
                raise RuntimeError('Post-action crash/TTC clock differs')
            if row['outcome']!=result['outcome'] or row['min_dist']!=action['min_dist']:
                raise RuntimeError('Action fields differ')
        rows.extend(r);action_count+=len(actions)
    output=root/'data';output.mkdir()
    pq.write_table(pa.Table.from_pylist(rows),output/'rollouts.parquet',compression='zstd')
    if pq.read_table(output/'rollouts.parquet').to_pylist()!=rows:raise RuntimeError('Parquet cell readback differs')
    receipt=dict(passed=True,runs=len(results),actions=action_count,inferences=inference_count,
        complete_matrix_shard=len(results)==600,source_commit=plan['code_commit'],
        parquet_sha256=sha(output/'rollouts.parquet'),all_diagnostic_tensors_reduced=True,
        task2_fit=False,results=results)
    atomic_json(root/('COMPLETE.json' if len(results)==600 else 'PARTIAL_EXPORTED.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();export(a.root)
