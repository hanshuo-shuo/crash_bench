"""Fixed Task3 paired outcome summary; no fitting, threshold search, or new rollouts."""
import argparse
import json
import os
import time
from pathlib import Path
from collections import Counter
from common import atomic_json,sha,scene_id,seed_for,outcome
from analyze import load,write_csv
from gate import schedule,adapt_gate
from audit_shard import terminal_accounting
import cluster_stats


def features(run):
    C=run['C'];L=run['L'];success=int(run['success'])
    return dict(CAR=int(C is None),crash_rate=int(C is not None),TSR=success,
        safe_success=int(C is None and success),safe_incomplete=int(C is None and not success),
        budget_goal_incomplete=int(not success and L==300))


def paired_guard(runs,split):
    expected={(s,a,r) for s in split['train']+split['test'] for a in ['nominal','aegis','gate'] for r in range(5)}
    actual=[(r['scene'],r['arm'],r['repeat']) for r in runs]
    if len(actual)!=900 or len(set(actual))!=900 or set(actual)!=expected:raise RuntimeError('Complete paired60×5×3 population required')
    reference={}
    for run in runs:
        key=(run['scene'],run['repeat'])
        if not 1<=run['L']<=300 or run['C'] is not None and not 1<=run['C']<=run['L']:raise RuntimeError('Invalid outcome clock')
        if run['outcome']!=outcome(bool(run['success']),run['C']):raise RuntimeError('Outcome semantics differ')
        if key in reference and reference[key]!=run['seed']:raise RuntimeError('Paired native model seed differs')
        reference[key]=run['seed']


def tables(runs,split,cfg,output):
    paired_guard(runs,split);outcomes=[];differences=[];gate_rows=[];case_rows=[]
    by_key={(r['scene'],r['repeat'],r['arm']):r for r in runs}
    for population,states in [('test_primary',sorted(split['test'])),('train_descriptive',sorted(split['train'])),('all_states_descriptive',sorted(split['train']+split['test']))]:
        mult=cluster_stats.draws(states,[s.rsplit('/',1)[0] for s in states],cfg['bootstrap_replicates'],cfg['bootstrap_seed'])
        if population=='test_primary':
            import numpy as np
            np.save(output/'test_bootstrap_multiplicities.npy',mult,allow_pickle=False)
        chosen=[r for r in runs if r['scene'] in states]
        def estimate(records,values):
            point,ci=cluster_stats.mean(values,[r['scene'] for r in records],[1.]*len(records),states,mult)
            return dict(value=point,**ci)
        for arm in ['nominal','aegis','gate']:
            selected=[r for r in chosen if r['arm']==arm]
            for metric in features(selected[0]):
                values=[features(r)[metric] for r in selected]
                outcomes.append(dict(population=population,arm=arm,metric=metric,**estimate(selected,values),events=sum(values),rollouts=len(selected),states=len(states)))
        pairs=[by_key[(s,r,'gate')] for s in states for r in range(5)]
        for target,reference in [('gate','nominal'),('gate','aegis'),('aegis','nominal')]:
            target_pairs=[by_key[(s,r,target)] for s in states for r in range(5)]
            for metric in features(target_pairs[0]):
                values=[features(g)[metric]-features(by_key[(g['scene'],g['repeat'],reference)])[metric] for g in target_pairs]
                differences.append(dict(population=population,comparison=target+'-minus-'+reference,metric=metric,**estimate(target_pairs,values),paired_seeds=len(target_pairs),states=len(states),units='absolute_probability_difference'))
        for metric,num,den in [('gated_action_fraction','gated_actions','L'),('gated_infer_fraction','gated_inferences','inferences'),('at_risk_gated_infer_fraction','risk_gated','risk_inferences')]:
            gate_rows.append(dict(population=population,metric=metric,**estimate(pairs,[g[num]/g[den] for g in pairs]),numerator=sum(g[num] for g in pairs),denominator=sum(g[den] for g in pairs),rollouts=len(pairs),states=len(states),weighting='equal_state_then_rollout; total_counts_are_descriptive'))
    for run in sorted(runs,key=lambda r:(r['scene'],r['repeat'],r['arm'])):
        case_rows.append({k:run[k] for k in ['id','scene','task','arm','repeat','seed','C','L','success','outcome']})
        case_rows[-1].update(features(run))
        if run['arm']=='gate':case_rows[-1].update({k:run[k] for k in ['gated_actions','gated_inferences','risk_inferences','risk_gated']})
    for name,rows in [('outcomes.csv',outcomes),('paired_differences.csv',differences),('gate_frequencies.csv',gate_rows),('paired_cases.csv',case_rows)]:write_csv(output/name,rows)


def main(gate_root,output):
    plan=json.loads((gate_root/'SUBMITTED.json').read_text());cfg=plan['configuration']
    fingerprints=json.loads((gate_root/'FINGERPRINTS.json').read_text())
    for item in json.loads((gate_root/'SOURCE_FILES.json').read_text()):
        if sha(gate_root/'source'/item['path'])!=item['sha256']:raise RuntimeError('Frozen gate source changed')
    for item in fingerprints['upstream_files']:
        if sha(Path(plan['upstream_root'])/item['path'])!=item['sha256']:raise RuntimeError('Frozen gate upstream changed')
    for item in fingerprints['assets']+[fingerprints['container']]+cfg['controller_files']:
        if sha(Path(item['path']))!=item['sha256']:raise RuntimeError('Consumed gate asset/controller bytes changed')
    analysis=Path(plan['analysis_root'])/'analysis'
    if sha(analysis/'TRAIN_THRESHOLDS.json')!=plan['threshold_file_sha256']:raise RuntimeError('Frozen cutoff provenance changed')
    cutoff=json.loads((analysis/'TRAIN_THRESHOLDS.json').read_text())
    if cutoff['test_used'] or cutoff['gate_q90']!=cfg['gate_threshold']:raise RuntimeError('Train-only fixed cutoff changed')
    baseline,split,audit=load(Path(cfg['audited_input']))
    runs=[r for r in baseline if r['repeat']<5];receipts=[];accounting=[];gate_results=[]
    upstream=Path(plan['upstream_root']);original=(upstream/'main/main_aegis.py').read_text()
    expected_source=adapt_gate(original,'gate')
    for shard in [0,1]:
        p=gate_root/'shards'/str(shard)
        if (p/'STOP.json').exists() or (p/'PARTIAL.json').exists() or (gate_root/'STOP.json').exists():raise RuntimeError('Gate failure/partial record present')
        job=json.loads((p/'SUBMITTED.json').read_text())['job'];accounting.append(terminal_accounting(job))
        receipt=json.loads((p/'COMPLETE.json').read_text());receipts.append(receipt)
        if not receipt['passed'] or receipt['runs']!=150 or not receipt['complete_matrix_shard'] or receipt['api_calls']!=0 or receipt['source_commit']!=plan['code_commit'] or not receipt['independent_all_sample_reduction'] or not receipt['native_same_input_rng_exact'] or not receipt['paired_settled_physical_arrays_exact']:raise RuntimeError('Complete independently verified gate shard required')
        if sha(p/'data/rollouts.parquet')!=receipt['parquet_sha256']:raise RuntimeError('Gate Parquet changed')
        import pyarrow.parquet as pq
        table=pq.read_table(p/'data/rollouts.parquet').to_pylist();offset=0;gated=0;risk_gated=0;ninfer=0
        results=receipt['results']
        if [r['spec'] for r in results]!=schedule(cfg,shard):raise RuntimeError('Gate schedule changed')
        for result in results:
            spec=result['spec'];s=scene_id(spec['scene']);d=p/'runs'/spec['name']
            if json.loads((d/'RESULT.json').read_text())!=result or result['status']!='complete' or result['seed']!=seed_for(spec['scene'],spec['repeat']):raise RuntimeError('Gate per-case identity changed')
            if (d/'adapted_main_aegis.py').read_text()!=expected_source:raise RuntimeError('Gate adapter differs from frozen nominal execution hook')
            rows=[json.loads(x) for x in (d/'rows.jsonl').read_text().splitlines()]
            if table[offset:offset+len(rows)]!=rows or len(rows)!=result['actions']:raise RuntimeError('Gate every-cell trace differs')
            offset+=len(rows)
            steps=[json.loads(x) for x in (d/'steps.jsonl').read_text().splitlines()]
            infos=[json.loads(x) for x in (d/'inferences.jsonl').read_text().splitlines()]
            C=result['collision_step'];risk=[i for i in infos if C is None or i['step']<=C]
            a=sum(x['gate_active'] for x in steps);b=sum(x['disagreement']>cfg['gate_threshold'] for x in infos);c=sum(x['disagreement']>cfg['gate_threshold'] for x in risk)
            gated+=a;risk_gated+=c;ninfer+=len(infos);gate_results.append(result)
            runs.append(dict(id=spec['name'],scene=s,task=s.rsplit('/',1)[0],arm='gate',repeat=spec['repeat'],seed=result['seed'],C=C,L=result['actions'],success=int(result['success']),outcome=result['outcome'],inferences=len(infos),gated_actions=a,gated_inferences=b,risk_inferences=len(risk),risk_gated=c))
        if offset!=len(table) or gated!=receipt['gated_actions'] or risk_gated!=receipt['at_risk_gated_inferences'] or ninfer!=receipt['inferences']:raise RuntimeError('Gate counts differ')
    paired_guard(runs,split)
    output.mkdir(exist_ok=False)
    tables(runs,split,json.loads((Path(__file__).parent/'config.yaml').read_text())['task2'],output)
    billing_path=Path(cfg['collection_root'])/'API_FINAL.json';billing=json.loads(billing_path.read_text())
    atomic_json(output/'PROVENANCE.json',dict(gate_root=str(gate_root),gate_commit=plan['code_commit'],gate_gpu_accounting=accounting,prior_gpu_accounting=plan['prior_gpu_accounting'],actual_series_gpu_seconds=plan['prior_gpu_seconds']+sum(a['elapsed_seconds'] for a in accounting),authorized_series_gpu_seconds=plan['authorized_ceiling_gpu_seconds'],series_committed_usd=billing['campaign_committed_usd'],api_final_path=str(billing_path),api_final_sha256=sha(billing_path),threshold=cfg['gate_threshold'],threshold_file_sha256=plan['threshold_file_sha256'],
        task1_collection_commit=audit['code_commit'],task1_audit_sha256=sha(Path(cfg['audited_input'])/'INDEPENDENT_AUDIT.json'),task1_parquet_sha256=audit['parquet_sha256'],task2_analysis_root=str(analysis),task2_complete_sha256=sha(analysis/'ANALYSIS_COMPLETE.json'),
        gate_shard_parquet_sha256=[r['parquet_sha256'] for r in receipts],gate_complete_sha256=[sha(gate_root/'shards'/str(s)/'COMPLETE.json') for s in [0,1]],analysis_code_commit=os.environ.get('CB_CODE_COMMIT'),slurm_job=os.environ.get('SLURM_JOB_ID'),
        stats_plan_sha256=sha(Path(__file__).parent/'STATS_PLAN.md'),split_sha256=sha(Path(__file__).parent/'split.json'),started_unix=time.time(),final_source_upstream_assets_controller_bytes_verified=True,bootstrap='frozen task-stratified state multiplicities; paired5 seeds/three arms retained; threshold treated as fixed',
        stall_proxy='goal not reached at original300-action cap; no mechanical stillness inference',new_vlm_calls=0,raw_remote=str(gate_root)))
    atomic_json(output/'GATE_ANALYSIS_COMPLETE.json',dict(passed=True,gate_rollouts=300,matched_baseline_rollouts=600,test_rollouts_per_arm=90,train_states=42,test_states=18,threshold=cfg['gate_threshold'],new_rollouts=0,api_calls=0,files={p.name:sha(p) for p in output.iterdir() if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('gate_root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();main(a.gate_root.resolve(),a.output.resolve())
