"""Read-only analysis of an interrupted schedule; never label missing runs.

Only states with all 5 repeats from both methods enter paired rates and tables.
This deliberately does not bypass protocol.summarize's 600-row completeness gate.
"""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import csv
import hashlib
import json
from pathlib import Path
from protocol import SCENARIOS, CATEGORIES, schedule, scenario_name, tables, category, stable

def read_rows(path):
    rows = list(csv.DictReader(Path(path).open()))
    for r in rows:
        for key in ['task','episode','repeat','seed','end_step','modified_steps','policy_requests','video_frames']:
            r[key] = int(r[key])
        for key in ['collision_step','first_modified_step']:
            r[key] = int(r[key]) if r[key] else None
        for key in ['success','collided','exited']:
            if r[key] not in ['True','False']:
                raise ValueError('Invalid boolean '+key)
            r[key] = r[key] == 'True'
        r['vlm_correct'] = {'True':True,'False':False,'':None}[r['vlm_correct']]
        for key in ['max_obstacle_l1_m','modification_threshold','elapsed_seconds']:
            r[key] = float(r[key])
    return rows

def audit(rows):
    expected = list(schedule())
    if not 0 < len(rows) < len(expected):
        raise ValueError('This analyzer requires a nonempty partial run')
    states = defaultdict(list);pairs = defaultdict(list)
    for r,(s,e,repeat,method,seed) in zip(rows,expected):
        if (r['scenario'],r['episode'],r['repeat'],r['method'],r['seed']) != (scenario_name(s),e,repeat,method,seed):
            raise ValueError('Raw CSV is not the exact completed schedule prefix')
        if r['phase'] != 'full' or r['collided'] != (r['max_obstacle_l1_m'] > .001):
            raise ValueError('Phase or collision label inconsistent')
        if r['modification_threshold'] != 1e-6:
            raise ValueError('Unexpected action threshold')
        if any(len(r[k]) != 64 for k in ['qpos_sha256','policy_reset_key','first_action_chunk_sha256']):
            raise ValueError('Missing pairing evidence')
        states[(r['scenario'],r['episode'])].append(r)
        pairs[(r['scenario'],r['episode'],r['repeat'])].append(r)
    for key,rs in states.items():
        if len({r['qpos_sha256'] for r in rs})!=1 or len({r['active_obstacle'] for r in rs})!=1:
            raise ValueError('Settled state or active obstacle differs: '+str(key))
    action_mismatches=[]
    for key,rs in pairs.items():
        if len(rs)==2:
            for field in ['seed','policy_reset_key']:
                if rs[0][field]!=rs[1][field]:
                    raise ValueError('Paired seed/RNG mismatch: '+str(key))
            if rs[0]['first_action_chunk_sha256']!=rs[1]['first_action_chunk_sha256']:
                action_mismatches.append({'scenario':key[0],'episode':key[1],'repeat':key[2],
                                          'run_ids':[r['run_id'] for r in rs]})
    for field in ['code_commit','upstream_commit','slurm_job']:
        if len({r[field] for r in rows})!=1:
            raise ValueError('Mixed provenance: '+field)
    complete = []
    for key,rs in states.items():
        if len(rs)==10:
            if Counter(r['method'] for r in rs)!={'nominal':5,'aegis':5}:
                raise ValueError('Invalid method coverage')
            complete.extend(rs)
    return complete, states, {'completed_pair_checks':sum(len(rs)==2 for rs in pairs.values()),
        'qpos_checked_initial_states':len(states),'seed_and_qpos_mismatches':0,
        'exact_first_chunk_pairs_passed':not action_mismatches,
        'first_chunk_mismatch_pairs':action_mismatches,
        'interpretation':'Descriptive outcomes only; mismatching first policy outputs require investigation before an exact paired causal claim.' if action_mismatches else 'Pairing checks passed.'}

def rates(rows, threshold=.001):
    output = {}
    for name in ['合并',*map(scenario_name,SCENARIOS)]:
        g = {}
        for method in ['nominal','aegis']:
            rs=[r for r in rows if r['method']==method and (name=='合并' or r['scenario']==name)]
            n=len(rs);collisions=sum(r['max_obstacle_l1_m']>threshold for r in rs)
            success=sum(r['success'] for r in rs);ss=sum(category(r,threshold)=='安全完成' for r in rs)
            cs=sum(r['success'] and r['max_obstacle_l1_m']>threshold for r in rs)
            g[method]={'n':n,'success':success,'collisions':collisions,'safe_success':ss,
                'success_rate':success/n if n else None,'avoidance_rate':1-collisions/n if n else None,
                'safe_success_rate':ss/n if n else None,'collided_and_success':cs,
                'collided_and_success_fraction_all':cs/n if n else None,
                'success_given_collision':cs/collisions if collisions else None}
        output[name]=g
    return output

def write_rows(path, rows, fields=None):
    if fields is None:fields=list(rows[0])
    with Path(path).open('w',newline='') as file:
        w=csv.DictWriter(file,fieldnames=fields);w.writeheader();w.writerows(rows)

def analyze(raw, output, budget_path):
    raw=Path(raw);output=Path(output)
    rows=read_rows(raw);complete,states,audit_result=audit(rows)
    output.mkdir(parents=True,exist_ok=False)
    missing=[]
    for s in SCENARIOS:
        for e in range(20):
            rs=states.get((scenario_name(s),e),[])
            if len(rs)!=10:
                missing.append({'scenario':scenario_name(s),'episode':e,'completed_runs':len(rs),'required_runs':10})
    report={'status':'PARTIAL_INTERRUPTED_PAIRING_WARNING' if not audit_result['exact_first_chunk_pairs_passed'] else 'PARTIAL_INTERRUPTED','completed_runs':len(rows),'expected_runs':600,
        'complete_initial_states':len(complete)//10,'expected_initial_states':60,
        'paired_runs_used':len(complete),'completed_runs_excluded_from_paired_analysis':len(rows)-len(complete),
        'missing_or_incomplete_states':missing,'raw_csv_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),
        'audit':audit_result,'paired_rates_1mm':rates(complete),'all_completed_rates_1mm':rates(rows),
        'paired_rates_1cm':rates(complete,.01),'exits':sum(r['exited'] for r in rows),
        'exit_reasons':dict(Counter(r['exit_reason'] for r in rows if r['exited'])),
        'filter_disabled':sum(r['filter_status']=='未启用' for r in rows),
        'vlm_calls_with_completed_run':sum(r['method']=='aegis' for r in rows),
        'vlm_wrong':sum(r['vlm_correct'] is False for r in rows),
        'vlm_unreviewed':sum(r['method']=='aegis' and r['vlm_correct'] is None for r in rows),
        'thresholds':[tables(complete,t) for t in [.001,.01]]}
    flagged={(r['scenario'],r['episode']) for r in audit_result['first_chunk_mismatch_pairs']}
    strict=[r for r in complete if (r['scenario'],r['episode']) not in flagged]
    report['diagnostic_excluding_first_chunk_mismatch_states']={
        'purpose':'Diagnostic sensitivity only, not a replacement primary population.',
        'excluded_states':len(flagged),'remaining_states':len(strict)//10,'rates_1mm':rates(strict),
        'thresholds':[tables(strict,t) for t in [.001,.01]]}
    ledger=json.loads(Path(budget_path).read_text())
    formal=[c for key,c in ledger['calls'].items() if key.startswith('full/')]
    report['api']={'formal_attempts':len(formal),
        'formal_known_usd':str(sum((Decimal(c['cost_usd']) for c in formal if 'cost_usd' in c),Decimal(0))),
        'formal_unknown_reservations_usd':str(sum((Decimal(c['reserved_usd']) for c in formal if 'cost_usd' not in c),Decimal(0))),
        'total_committed_including_history_usd':str(Decimal(ledger['initial_spent_usd'])+sum((Decimal(c.get('cost_usd',c['reserved_usd'])) for c in ledger['calls'].values()),Decimal(0))),
        'limit_usd':ledger['limit_usd']}
    state_detail=[]
    for key,rs in sorted(states.items()):
        if len(rs)!=10:continue
        arms={m:[r for r in rs if r['method']==m] for m in ['nominal','aegis']}
        for threshold in [.001,.01]:
            labels={m:stable(rr,threshold) for m,rr in arms.items()}
            white=labels['nominal']==labels['aegis']=='安全完成' and sum(r['modified_steps']>0 for r in arms['aegis'])>=3
            state_detail.append({'scenario':key[0],'episode':key[1],'active_obstacle':rs[0]['active_obstacle'],
                'collision_threshold_m':threshold,'nominal_stable_class':labels['nominal'] or '不稳定',
                'aegis_stable_class':labels['aegis'] or '不稳定','unstable_either':None in labels.values(),
                'white_intervention':white,'aegis_runs_with_modification':sum(r['modified_steps']>0 for r in arms['aegis'])})
    write_rows(output/'state_classifications.csv',state_detail)
    write_rows(output/'missing_states.csv',missing)
    wrong=[{key:r[key] for key in ['run_id','scenario','episode','repeat','active_obstacle','vlm_object','vlm_correct']} for r in rows if r['vlm_correct'] is False]
    write_rows(output/'vlm_wrong.csv',wrong,fields=['run_id','scenario','episode','repeat','active_obstacle','vlm_object','vlm_correct'])
    text=['# 配对重复实验：部分结果','',
          '完成%d/600次；主分析只用%d个完整初始状态（%d次，每方法%d次）。'%(len(rows),len(complete)//10,len(complete),len(complete)//2),
          '%d次已完成记录因同状态的重复未齐而不进入配对比较；其余%d次未完成或未执行，不能计为失败。'%(len(rows)-len(complete),600-len(rows)),'',
          '行是π0.5，列是AEGIS；每格单位为初始状态。每方法5次至少4次同类才稳定。','']
    if flagged:
        text+=['质量警示：种子、RNG回执、qpos均一致，但%d对首个原始策略动作块不完全相同，涉及%d个状态。以下仅为观察到的结局统计，严格配对归因需要先排查该差异。'%(len(audit_result['first_chunk_mismatch_pairs']),len(flagged)),'']
    for result in report['thresholds']:
        threshold=result['collision_threshold_m'];suffix='1mm' if threshold==.001 else '1cm'
        text+=['## 碰撞阈值 '+suffix,'']
        table_rows=[]
        names=[*map(scenario_name,SCENARIOS),'合并']
        names += [key for key in result['groups'] if key.startswith('障碍/')]
        for name in names:
            g=result['groups'][name]
            assert sum(map(sum,g['table']))+g['碰巧']==g['states']
            text += ['### '+name+'（%d个完整状态）'%g['states'],'','| π0.5 / AEGIS | '+' | '.join(CATEGORIES)+' |','|---|---:|---:|---:|']
            for i,a in enumerate(CATEGORIES):
                text.append('| '+a+' | '+' | '.join(map(str,g['table'][i]))+' |')
                for j,b in enumerate(CATEGORIES):table_rows.append({'group':name,'nominal':a,'aegis':b,'states':g['table'][i][j]})
            text += ['','按用户定义的“碰巧”：%d；“白干预”：%d。'%(g['碰巧'],g['白干预']),'']
        write_rows(output/('cross_tables_'+suffix+'.csv'),table_rows)
        write_rows(output/('state_differences_'+suffix+'.csv'),result['safe_success_differences'])
    (output/'tables.md').write_text('\n'.join(text)+'\n')
    (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('output',type=Path);p.add_argument('--budget',type=Path,required=True)
    a=p.parse_args();report=analyze(a.raw,a.output,a.budget)
    print(json.dumps({key:report[key] for key in ['completed_runs','complete_initial_states','audit','api','exits','vlm_wrong','vlm_unreviewed']},ensure_ascii=False,indent=2))
