"""Read terminal evidence, independently reduce NumPy metrics, package without fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time
from adapter import adapt
from common import atomic_json, full_schedule, noise_seed, scene_id, seed_for
from preflight import stream_sha


def terminal_accounting(job):
    result=subprocess.run(['sacct','-j',str(job),'--format=JobID,State,ExitCode,ElapsedRaw','-n','-P'],capture_output=True,text=True,check=True)
    rows=[x.split('|') for x in result.stdout.splitlines() if x.split('|')[0]==str(job)]
    if len(rows)!=1 or rows[0][1:3]!=['COMPLETED','0:0']:
        raise RuntimeError('Shard is not terminal COMPLETED/0:0: '+result.stdout)
    return dict(job=str(job),state=rows[0][1],exit_code=rows[0][2],elapsed_seconds=int(rows[0][3]))


def validate_receipt(receipt, expected, commit):
    if not receipt['passed'] or receipt['runs']!=600 or not receipt['complete_matrix_shard'] or receipt['source_commit']!=commit:
        raise RuntimeError('Missing full successful600-case receipt')
    if [x['spec'] for x in receipt['results']]!=expected or any(x['status']!='complete' for x in receipt['results']):
        raise RuntimeError('Completed identities/order differ from frozen schedule')


def audit(collection, shard, output):
    import numpy as np
    import pyarrow.parquet as pq
    plan=json.loads((collection/'SUBMITTED.json').read_text());cfg=plan['configuration']
    root=collection/'shards'/str(shard)
    accounting=terminal_accounting(plan['jobs'][shard])
    if (collection/'STOP.json').exists() or (root/'STOP.json').exists() or (root/'PARTIAL.json').exists():
        raise RuntimeError('Failure/partial record present')
    receipt=json.loads((root/'COMPLETE.json').read_text())
    validate_receipt(receipt,list(full_schedule(cfg,shard)),plan['code_commit'])
    if stream_sha(root/'data/rollouts.parquet')!=receipt['parquet_sha256']:raise RuntimeError('Parquet byte hash differs')
    for item in json.loads((collection/'SOURCE_FILES.json').read_text()):
        if stream_sha(collection/'source'/item['path'])!=item['sha256']:raise RuntimeError('Frozen collection source changed')
    fingerprints=json.loads((collection/'FINGERPRINTS.json').read_text())
    for item in fingerprints['upstream_files']:
        if stream_sha(Path(plan['upstream_root'])/item['path'])!=item['sha256']:raise RuntimeError('Frozen upstream bytes changed')
    for item in fingerprints['assets']+[fingerprints['container']]:
        if stream_sha(Path(item['path']))!=item['sha256']:raise RuntimeError('Consumed asset/container bytes changed')
    original=(Path(plan['upstream_root'])/'main/main_aegis.py').read_text()
    table=pq.read_table(root/'data/rollouts.parquet').to_pylist()
    offset=0;settled={};inferences=0;max_u=0.;max_churn=0.;snapshots=0
    filters=dict(aegis_enabled=0,aegis_native_empty_pointcloud_off=0,nominal_off=0)
    for result in receipt['results']:
        spec=result['spec'];directory=root/'runs'/spec['name'];scene=scene_id(spec['scene'])
        if json.loads((directory/'RESULT.json').read_text())!=result:raise RuntimeError('Per-case receipt differs')
        if result['seed']!=seed_for(spec['scene'],spec['repeat']):raise RuntimeError('Seed differs')
        if (directory/'adapted_main_aegis.py').read_text()!=adapt(original,spec['method']):raise RuntimeError('Adapted original evaluator differs')
        proof=json.loads((directory/'CONDITIONAL_ACTION_EQUALITY.json').read_text())
        if not proof['passed'] or proof['actions']!=result['actions'] or proof['inferences']!=result['inferences']:
            raise RuntimeError('Conditional invariance proof incomplete')
        with np.load(directory/'settled_state.npz',allow_pickle=False) as saved:
            arrays={k:saved[k].copy() for k in saved.files}
        digest=hashlib.sha256(np.asarray(arrays['qpos'],dtype='<f8').tobytes()).hexdigest()
        if digest!=spec['scene']['expected_qpos_sha256'] or digest!=result['settled_qpos_sha256']:raise RuntimeError('Historical qpos identity differs')
        if scene in settled:
            if set(arrays)!=set(settled[scene]) or any(not np.array_equal(v,settled[scene][k]) for k,v in arrays.items()):
                raise RuntimeError('Paired settled physical arrays differ')
        else:settled[scene]=arrays
        geometry=json.loads((directory/'geometry.json').read_text())
        if geometry['obstacle']!=spec['scene']['expected_obstacle']:raise RuntimeError('Protected obstacle differs')
        if spec['method']=='nominal':
            if geometry['filter_enabled']:raise RuntimeError('Nominal enabled QP')
            filters['nominal_off']+=1
        else:filters['aegis_enabled' if geometry['filter_enabled'] else 'aegis_native_empty_pointcloud_off']+=1
        infos=[json.loads(x) for x in (directory/'inferences.jsonl').read_text().splitlines()]
        steps=[json.loads(x) for x in (directory/'steps.jsonl').read_text().splitlines()]
        rows=[json.loads(x) for x in (directory/'rows.jsonl').read_text().splitlines()]
        if len(steps)!=result['actions'] or len(rows)!=len(steps) or len(infos)!=result['inferences']:raise RuntimeError('Trace length differs')
        if table[offset:offset+len(rows)]!=rows:raise RuntimeError('Every-cell Parquet/JSONL differs')
        offset+=len(rows);previous=None;last_rng=None;chunks={};clock={};scales=None
        for index,info in enumerate(infos,1):
            if info['infer_index']!=index or info['step']!=1+5*(index-1) or not info['native_action_array_equal'] or not info['rng_unchanged']:
                raise RuntimeError('Inference clock/native invariance differs')
            if last_rng is not None and last_rng!=info['rng_before']:raise RuntimeError('Native RNG chain differs')
            last_rng=info['rng_after']
            with np.load(directory/('infer_%03d.npz'%index),allow_pickle=False) as sample:
                values=sample['diagnostic_actions'];chunk=sample['actions'].copy();scale=sample['normalization_scales']
                expected_seeds=[noise_seed(scene,result['seed'],index,i) for i in range(8)]
                if values.shape!=(8,10,7) or chunk.shape!=(10,7) or scale.shape!=(7,) or (scale<=0).any():raise RuntimeError('Sample/chunk/scale shape differs')
                if not np.isfinite(values).all() or not np.isfinite(chunk).all() or not np.isfinite(scale).all():raise RuntimeError('Nonfinite arrays')
                if sample['sample_seeds'].tolist()!=expected_seeds or info['sample_seeds']!=expected_seeds:raise RuntimeError('Independent diagnostic seeds differ')
                if info['normalization_scales']!=scale.tolist() or scales is not None and not np.array_equal(scales,scale):raise RuntimeError('Normalization scales differ')
                scales=scale.copy();u=float(np.std(values/scale[None,None,:],axis=0,ddof=1).mean())
                error=abs(u-info['disagreement']);max_u=max(max_u,error)
                if error>1e-12:raise RuntimeError('Independent sample STD differs')
                churn=None if previous is None else float(np.mean(np.abs(chunk[:5]-previous[5:])/scale))
                if churn is None:
                    if info['churn'] is not None:raise RuntimeError('First churn is not missing')
                else:
                    error=abs(churn-info['churn']);max_churn=max(max_churn,error)
                    if error>1e-12:raise RuntimeError('Independent shifted churn differs')
                previous=chunk;chunks[index]=chunk;clock[index]=info;inferences+=1
        collision=next((x['step'] for x in steps if x['obstacle_l1_m']>cfg['collision_l1_m']),None)
        outcome='crash' if collision is not None else ('safe_success' if result['success'] else 'safe_incomplete')
        if collision!=result['collision_step'] or outcome!=result['outcome'] or steps[-1]['success']!=result['success']:
            raise RuntimeError('Official collision/success label differs')
        for t,(action,row) in enumerate(zip(steps,rows),1):
            index=action['infer_index'];info=clock[index]
            if action['step']!=t or row['step']!=t or index!=(t-1)//5+1:raise RuntimeError('Action/inference clock differs')
            if not action['queue_array_equal'] or not action['physical_before_action_unchanged'] or not np.array_equal(action['raw'],chunks[index][(t-1)%5]):
                raise RuntimeError('Raw queue/physical nonmutation differs')
            for field,value in [('act_norm',float(np.linalg.norm(np.asarray(action['applied'])/scales))),('act_norm6',float(np.linalg.norm(np.asarray(action['applied'])[:6]/scales[:6])))]:
                if abs(row[field]-value)>1e-12:raise RuntimeError('Applied command norm differs')
            ttc=-1 if collision is None else (collision-t if t<=collision else None)
            if row['time_to_crash']!=ttc or row['crashed']!=(collision is not None and t>=collision):raise RuntimeError('TTC/crash clock differs')
            if row['min_dist']!=action['min_dist'] or row['disagreement']!=info['disagreement'] or row['churn']!=info['churn'] or row['infer_boundary']!=(t==info['step']):
                raise RuntimeError('Action/proxy pairing differs')
            if row['scene_id']!=scene or row['seed']!=result['seed'] or row['capability']!=spec['method'] or row['outcome']!=outcome:raise RuntimeError('Row identity/outcome differs')
        paths=sorted((directory/'snapshots').glob('*.json'))
        if len(paths)!=result['snapshots'] or len(paths)!=(result['actions']//20 if spec['method']=='nominal' else 0):raise RuntimeError('Save-only snapshot count differs')
        for file in paths:
            snap=json.loads(file.read_text());prefix='%03d'%snap['step']
            if snap['restore_verified'] or stream_sha(file.parent/(prefix+'_physics.npz'))!=snap['physics_sha256'] or stream_sha(file.parent/(prefix+'_observation.npz'))!=snap['observation_sha256']:
                raise RuntimeError('Saved snapshot bytes/scope differ')
            snapshots+=1
    if offset!=len(table) or len(settled)!=30 or inferences!=receipt['inferences']:raise RuntimeError('Incomplete shard coverage')
    output.mkdir(exist_ok=False);manifest=[]
    for file in sorted(root.rglob('*')):
        if file.is_symlink():manifest.append(dict(path=str(file.relative_to(root)),kind='symlink',target=str(file.readlink())))
        elif file.is_file():manifest.append(dict(path=str(file.relative_to(root)),kind='file',bytes=file.stat().st_size,sha256=stream_sha(file)))
    atomic_json(output/'FILES.json',manifest)
    archive=output/'evidence.tar'
    with tarfile.open(archive,'w',dereference=False) as tar:
        for item in manifest:
            file=root/item['path'];info=tar.gettarinfo(str(file),arcname='shard_%d/'%shard+item['path'])
            if item['kind']=='symlink':tar.addfile(info)
            else:
                with file.open('rb') as stream:tar.addfile(info,stream)
    report=dict(passed=True,shard=shard,runs=600,states=30,actions=len(table),diagnostic_inferences=inferences,
        diagnostic_samples_per_infer=[8,10,7],all_samples_independently_reduced=True,maximum_disagreement_error=max_u,
        maximum_churn_error=max_churn,snapshots=snapshots,filter_counts=filters,accounting=accounting,
        collection=str(collection),collection_commit=plan['code_commit'],audit_commit=(output.parent/'SOURCE_COMMIT').read_text().strip(),
        parquet_sha256=receipt['parquet_sha256'],files=len(manifest),regular_file_bytes=sum(x.get('bytes',0) for x in manifest),
        archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=stream_sha(archive),manifest_sha256=stream_sha(output/'FILES.json'),
        upstream_files_rehashed=len(fingerprints['upstream_files']),assets_rehashed=len(fingerprints['assets']),container_rehashed=True,
        experimental_data_fitted=False,new_rollouts=0,api_calls=0,other_shard_modified=False,
        scope='Independent NumPy sample STD/shifted churn/norm/labels; byte-equal original adapters; every-cell Parquet; snapshots save-only; no restore claim')
    atomic_json(output/'SHARD_AUDIT.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('collection',type=Path);p.add_argument('shard',type=int,choices=[0,1]);p.add_argument('output',type=Path)
    a=p.parse_args();audit(a.collection.resolve(),a.shard,a.output.resolve())
