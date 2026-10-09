"""Allocated CPU independent whole-matrix audit and delivery; no fitting/API."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from common import FIELDS, atomic_json, config, full_schedule, scene_id, seed_for, sha


def audit(collection, output):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    cfg=config('full');plan=json.loads((collection/'SUBMITTED.json').read_text())
    if not (collection/'COLLECTION_COMPLETE.json').exists():raise RuntimeError('Collection not complete')
    expected={x['name']:x for x in full_schedule(cfg)}
    rows=[];results=[];seen=set();inferences=0;maximum_disagreement_error=0.;maximum_churn_error=0.
    settled={};filter_counts={'aegis_enabled':0,'aegis_native_empty_pointcloud_off':0,'nominal_off':0}
    source_manifest=json.loads((collection/'SOURCE_FILES.json').read_text())
    for item in source_manifest:
        if sha(collection/'source'/item['path'])!=item['sha256']:raise RuntimeError('Source changed after collection')
    for shard in [0,1]:
        root=collection/'shards'/str(shard);receipt=json.loads((root/'COMPLETE.json').read_text())
        if not receipt['passed'] or receipt['runs']!=600:raise RuntimeError('Incomplete/failed shard')
        table=pq.read_table(root/'data/rollouts.parquet');serialized=[]
        for result in receipt['results']:
            spec=result['spec'];name=spec['name']
            if name in seen or spec!=expected.get(name) or result['status']!='complete':raise RuntimeError('Frozen matrix identity mismatch')
            seen.add(name);directory=root/'runs'/name;scene=scene_id(spec['scene'])
            if result['seed']!=seed_for(spec['scene'],spec['repeat']):raise RuntimeError('Seed pairing mismatch')
            with np.load(directory/'settled_state.npz',allow_pickle=False) as arrays:
                digest=hashlib.sha256(np.asarray(arrays['qpos'],dtype='<f8').tobytes()).hexdigest()
                if digest!=spec['scene']['expected_qpos_sha256']:raise RuntimeError('Historical initial-state identity differs')
                frozen={k:arrays[k].copy() for k in arrays.files}
                if scene in settled:
                    if any(not np.array_equal(v,settled[scene][k]) for k,v in frozen.items()):raise RuntimeError('Paired settled physical arrays differ')
                else:settled[scene]=frozen
            geometry=json.loads((directory/'geometry.json').read_text())
            if geometry['obstacle']!=spec['scene']['expected_obstacle']:
                raise RuntimeError('Active protected obstacle differs')
            if spec['method']=='nominal':
                if geometry['filter_enabled']:raise RuntimeError('Nominal unexpectedly enabled QP')
                filter_counts['nominal_off']+=1
            else:
                # Pinned original lines219–221 disable QP when filtered points
                # are empty. Preserve/report that native perception failure;
                # requiringTrue would discard a legitimate full-AEGIS outcome.
                filter_counts['aegis_enabled' if geometry['filter_enabled'] else 'aegis_native_empty_pointcloud_off']+=1
            infos=[json.loads(x) for x in (directory/'inferences.jsonl').read_text().splitlines()]
            steps=[json.loads(x) for x in (directory/'steps.jsonl').read_text().splitlines()]
            r=[json.loads(x) for x in (directory/'rows.jsonl').read_text().splitlines()]
            if len(r)!=len(steps) or len(steps)!=result['actions'] or len(infos)!=result['inferences']:
                raise RuntimeError('Trace lengths differ')
            previous=None;chunks={};clock={};scales=None;last_rng=None
            for info in infos:
                if not info['native_action_array_equal'] or not info['rng_unchanged']:
                    raise RuntimeError('Per-infer native invariance missing')
                if last_rng is not None and info['rng_before']!=last_rng:raise RuntimeError('Native RNG sequence broken')
                last_rng=info['rng_after']
                with np.load(directory/('infer_%03d.npz'%info['infer_index']),allow_pickle=False) as sample:
                    values=sample['diagnostic_actions'];scales=sample['normalization_scales']
                    if values.shape!=(8,10,7) or sample['sample_seeds'].tolist()!=info['sample_seeds']:
                        raise RuntimeError('Diagnostic tensor/seeds differ')
                    reduced=float(np.std(values/scales[None,None,:],axis=0,ddof=1).mean())
                    error=abs(reduced-info['disagreement']);maximum_disagreement_error=max(error,maximum_disagreement_error)
                    if error>1e-12:raise RuntimeError('Independent NumPy sample-std reduction differs')
                    chunk=sample['actions'].copy()
                    value=None if previous is None else float(np.mean(np.abs(chunk[:5]-previous[5:])/scales))
                    if value is None:
                        if info['churn'] is not None:raise RuntimeError('First churn not missing')
                    else:
                        error=abs(value-info['churn']);maximum_churn_error=max(error,maximum_churn_error)
                        if error>1e-12:raise RuntimeError('Independent shifted churn reduction differs')
                    previous=chunk;chunks[info['infer_index']]=chunk;clock[info['infer_index']]=info
                inferences+=1
            collision=next((x['step'] for x in steps if x['obstacle_l1_m']>cfg['collision_l1_m']),None)
            label='crash' if collision is not None else ('safe_success' if result['success'] else 'safe_incomplete')
            if collision!=result['collision_step'] or label!=result['outcome']:raise RuntimeError('Official label reduction differs')
            for action,row in zip(steps,r):
                t=action['step'];index=action['infer_index'];info=clock[index]
                if not np.array_equal(action['raw'],chunks[index][(t-1)%5]):raise RuntimeError('Production queue differs')
                if not action['queue_array_equal'] or not action['physical_before_action_unchanged']:raise RuntimeError('Action/physics invariance missing')
                if abs(float(np.linalg.norm(np.asarray(action['applied'])/scales))-row['act_norm'])>1e-12:
                    raise RuntimeError('Independent applied norm differs')
                ttc=-1 if collision is None else (collision-t if t<=collision else None)
                if row['step']!=t or row['time_to_crash']!=ttc or row['crashed']!=(collision is not None and t>=collision):
                    raise RuntimeError('Action/crash/TTC clock differs')
                if row['min_dist']!=action['min_dist'] or row['disagreement']!=info['disagreement'] or row['churn']!=info['churn']:
                    raise RuntimeError('Infer/action field pairing differs')
                if row['infer_boundary']!=(info['step']==t) or row['outcome']!=label or row['scene_id']!=scene:
                    raise RuntimeError('Final label/inference boundary/scene differs')
            serialized.extend(r);results.append(result)
        if table.to_pylist()!=serialized:raise RuntimeError('Every-cell Parquet/JSONL equality failed')
        rows.extend(serialized)
    if seen!=set(expected) or len(settled)!=60:raise RuntimeError('Incomplete1200×60 coverage')
    deadline=time.monotonic()+120
    while not (collection/'API_FINAL.json').exists():
        if time.monotonic()>deadline:raise RuntimeError('API final ledger receipt absent')
        time.sleep(2)
    api=json.loads((collection/'API_FINAL.json').read_text())
    calls=api['root_calls']
    if len(calls)>1200 or any(x['status']!='settled' for x in calls.values()):raise RuntimeError('Unknown/excess API attempts retained')
    if not all(k in rows[0] for k in FIELDS):raise RuntimeError('Required fields missing')
    output.mkdir();pq.write_table(pa.Table.from_pylist(rows),output/'rollouts.parquet',compression='zstd')
    if pq.read_table(output/'rollouts.parquet').to_pylist()!=rows:raise RuntimeError('Aggregate Parquet readback differs')
    atomic_json(output/'rollouts.json',results)
    report=dict(passed=True,rollouts=len(results),states=len(settled),actions=len(rows),diagnostic_inferences=inferences,
        all_diagnostic_sample_tensors_independently_reduced=True,maximum_disagreement_error=maximum_disagreement_error,
        maximum_churn_error=maximum_churn_error,filter_counts=filter_counts,collection=str(collection),code_commit=plan['code_commit'],
        parquet_sha256=sha(output/'rollouts.parquet'),api=api,task2_fitted=False,
        official_labels='obstacle L1 displacement>1mm; crash precedence outcome; success retained separately',
        distance='exact OBB box pairs; native other convex collision geoms; old smoke distance excluded',
        interpretation='within3 familiar tasks; image+prompt pi05; sequentialB1 shared-prefix suffix; save-only snapshots; renderer caveat retained')
    atomic_json(output/'INDEPENDENT_AUDIT.json',report)
    (output/'README.md').write_text('''# Full Task1 collection\n\n1200 rollouts,60 frozen initializations,10 paired seeds,nominal/full AEGIS,M8,H10,replan5.\nEvery diagnostic tensor is retained and independently reduced. Infer proxies are repeated over5 actions; infer_boundary identifies observations.\nOfficial crash is obstacle L1 displacement>1mm; crashed is POST-action; TTC−1 means never crash,0 is collision action,null is post-crash. Future collision/outcome is not a sampler input.\nGeometry uses exact OBB separation/signed SAT depth for box pairs after a verified MuJoCo3.2.3 defect; native other convex collision representations. Old smoke min_dist excluded.\nKeep raw success separately from crash-precedence outcome for TSR. Command norm is not mechanical speed. Snapshots are save-only, not a validated continuation restore.\nThe pinned pi05 model is image+prompt conditioned; saved continuous state is not model conditioning. Shared prefix uses nativeB1 sequential suffix kernels. Strict same-input/RNG action noninterference is verified; fresh-renderer trajectory equality is not claimed.\nSTATS_PLAN.md/split.json are frozen before Task2 fitting. This audit performs no predictive fit or gate.\n''')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('collection',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();audit(a.collection,a.output)
