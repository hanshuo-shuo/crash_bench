"""Retrospective paper audit of existing D5 labels and frozen round-two scores."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse, itertools, json, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from scripts.expansion.train_option_value import read_jsonl,sha256_file
from scripts.expansion.run_round1_repair import policy_metrics,paired_interval
from crashbench.models.paired_gain import choose_from_gains


def sign_flip(delta):
    delta=np.asarray(delta)
    signs=np.array(list(itertools.product((-1.,1.),repeat=len(delta))))
    observed=abs(delta.mean());null=abs((signs*delta).mean(1))
    return float((null>=observed-1e-12).mean())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    data=ROOT/'results/expansion/d5_statewise/e2a5264a802a_fe660c168e72_20260830T105726Z/postprocess_a4b1714/merged'
    a={r['block_id']:r for r in read_jsonl(data/'anchors.jsonl') if r['split_role'] in {'train','development'}}
    b=defaultdict(dict)
    for r in read_jsonl(data/'branches.jsonl'):
        if r['split_role'] in {'train','development'}:b[r['block_id']][r['option_id']]=r
    groups=defaultdict(Counter);mismatches=[];quadrants=defaultdict(Counter)
    for block,v in b.items():
        base,ref,stop=[v[k] for k in ['base_continue','observation_refresh','safe_stop']];m=a[block]
        flags={'n':1,'positive_refresh':ref['u0']>base['u0'],
            'rescue':ref['outcome']['task_success'] and not base['outcome']['task_success'],
            'base_cat':base['outcome']['catastrophe'],'base_success':base['outcome']['task_success'],
            'ref_success':ref['outcome']['task_success'],
            'new_cat':ref['outcome']['catastrophe'] and not base['outcome']['catastrophe'],
            'terminal_disagree':any(base['outcome'][k]!=ref['outcome'][k] for k in ['task_success','catastrophe','safe_noncompletion'])}
        for dim in ['condition','task_id','anchor_steps','severity_id']:
            groups[f'{m["split_role"]}/{dim}/{m[dim]}'].update({k:int(v) for k,v in flags.items()})
        risk=bool(base['outcome']['catastrophe']);benefit=max(x['u0'] for x in v.values())>base['u0']
        quadrants[m['split_role']][f'R{int(risk)}_B{int(benefit)}']+=1
        if m['condition']!='stale' and flags['terminal_disagree']:
            mismatches.append({'block_id':block,'source_id':m['physical_source_id'],'role':m['split_role'],'condition':m['condition'],
                'base':base['outcome'],'refresh':ref['outcome']})
    dev=[r for v in b.values() for r in v.values() if r['split_role']=='development']
    base_choices={block:'base_continue' for block in b if a[block]['split_role']=='development'}
    base_metric=policy_metrics(dev,base_choices)
    report={'kind':'retrospective_paper_audit','test_rows_read':0,'confirmatory':False,
            'groups':dict(groups),'risk_benefit_table':dict(quadrants),'neutral_terminal_mismatches':mismatches,
            'neutral_summary':{role:{'mismatch_count':sum(x['role']==role for x in mismatches),
                'source_count':len({x['source_id'] for x in mismatches if x['role']==role})} for role in ['train','development']},
            'policies':{},'input_sha256':{n:sha256_file(data/n) for n in ['anchors.jsonl','branches.jsonl']}}
    run=ROOT/'results/repair_round2/ff233e364e8f_5627906'
    rr=json.loads((run/'report.json').read_text())
    with np.load(run/'decision_index.npz',allow_pickle=False) as x:blocks=x['blocks'];roles=x['roles'];sources=x['sources']
    mask=roles=='development'
    choices_all={'Base':base_choices,'AlwaysRefresh':{block:'observation_refresh' for block in base_choices}}
    with np.load(run/'directq.npz',allow_pickle=False) as z:
        choices_all['DirectQ']=choose_from_gains(z['gains'][mask],blocks[mask])
    for recipe,mode in [('outcome_interaction','all'),('outcome_interaction','refresh_only'),('gain_standardized','all'),('gain_standardized','refresh_only')]:
        with np.load(run/recipe/'ensemble_epoch_500.npz',allow_pickle=False) as z:g=z['gains'][mask]
        choices_all[f'{recipe}_500_{mode}']=choose_from_gains(g,blocks[mask],allow_stop=mode=='all')
    choices_all['OracleDiagnostic']={block:max(v,key=lambda k:(v[k]['u0'],k=='base_continue')) for block,v in b.items() if a[block]['split_role']=='development'}
    # Exposed-label oracle restricted to one action per exact binary Base-risk value.
    risk_actions={}
    for risk in (0,1):
        subset=[r for r in dev if b[r['block_id']]['base_continue']['outcome']['catastrophe']==risk]
        values={o:np.mean([np.mean([r['u0'] for r in subset if r['option_id']==o and r['physical_source_id']==s]) for s in sorted({r['physical_source_id'] for r in subset})]) for o in ['base_continue','observation_refresh','safe_stop']}
        risk_actions[risk]=max(values,key=values.get)
    choices_all['PerfectBinaryRiskRestrictedOracle']={block:risk_actions[b[block]['base_continue']['outcome']['catastrophe']] for block in base_choices}
    report['perfect_risk_oracle_actions']=risk_actions
    for name,choices in choices_all.items():
        metric=policy_metrics(dev,choices);counters=defaultdict(Counter);events=[]
        for block,choice in choices.items():
            v=b[block];base=v['base_continue'];chosen=v[choice];meta=a[block]
            event={'refresh_rescue':choice=='observation_refresh' and chosen['outcome']['task_success'] and not base['outcome']['task_success'],
                'base_success_lost':base['outcome']['task_success'] and not chosen['outcome']['task_success'],
                'new_catastrophe':chosen['outcome']['catastrophe'] and not base['outcome']['catastrophe'],
                'intervention':choice!='base_continue','utility_delta':chosen['u0']-base['u0']}
            for group in [meta['condition'],meta['task_id'],meta['physical_source_id']]:counters[group].update(event)
            if event['refresh_rescue'] or event['base_success_lost'] or event['new_catastrophe']:
                events.append({'block_id':block,'source_id':meta['physical_source_id'],'task_id':meta['task_id'],'condition':meta['condition'],'choice':choice,**event})
        delta=np.array([metric['per_source'][s]['u0']-base_metric['per_source'][s]['u0'] for s in sorted(metric['per_source'])])
        loo=[float(np.delete(delta,i).mean()) for i in range(len(delta))]
        report['policies'][name]={'metrics':metric,'paired_vs_base':paired_interval(metric,base_metric),
            'source_sign_flip_p_two_sided':sign_flip(delta),'leave_one_source_out_mean_delta_range':[min(loo),max(loo)],
            'source_wins_ties_losses':[int((delta>1e-9).sum()),int((abs(delta)<=1e-9).sum()),int((delta< -1e-9).sum())],
            'events':events,'subgroup_counters':dict(counters)}
    report['utility_preference_sensitivity']={}
    for catcost in [1.,2.,5.]:
        values={}
        for name,choices in choices_all.items():
            source_rows=defaultdict(list)
            for block,o in choices.items():
                row=b[block][o];source_rows[row['physical_source_id']].append(row['u0']+(2.-catcost)*row['outcome']['catastrophe'])
            values[name]=float(np.mean([np.mean(v) for v in source_rows.values()]))
        report['utility_preference_sensitivity'][str(catcost)]={'frozen_choices_not_reoptimized':True,'source_macro_u0':values}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'neutral_summary':report['neutral_summary'],'risk_benefit':report['risk_benefit_table'],
        'policies':{k:{'u0':v['metrics']['u0'],'p':v['source_sign_flip_p_two_sided'],'loo':v['leave_one_source_out_mean_delta_range'],
            'events':{c:v['subgroup_counters'].get(c) for c in ['stale','fresh_control','matched_buffer_control']}} for k,v in report['policies'].items()}},indent=2))

if __name__=='__main__':main()
