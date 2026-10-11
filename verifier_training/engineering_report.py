"""Aggregate fixed training and sensitivity results on Quest CPU."""
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *
from .engineering_contract import CONDITIONS,score_rationale


def main(root):
    rows=[json.loads(l) for l in (root/'dataset/dataset.jsonl').read_text().splitlines()]
    lookup={r['sample_id']:r for r in rows};audit=read(root/'audit/AUDIT.json')
    audit_rows={r['sample_id']:r for r in read(root/'audit/samples.json')}
    require(sha(root/'A/TRAINING_ORDER.json')==sha(root/'B/TRAINING_ORDER.json'),'A/B training order differs')
    summary={'scope':'Training-pool engineering fit and sensitivity only; no scientific generalization claim',
             'dataset_sha256':audit['dataset_sha256'],'original_sources':1,'audit':audit,'arms':{}}
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    cases=[];reviews=[]
    for arm_index,v in enumerate(('A','B')):
        result=read(root/v/'RESULT.json');conditions={}
        original=read(root/('diagnostics_'+v)/'original.json');original_by_id={g['sample_id']:g for g in original}
        for condition in CONDITIONS:
            records=read(root/('diagnostics_'+v)/(condition+'.json'))
            require(len(records)==120,'Missing diagnostic samples')
            if v=='B' and condition!='original':
                reference=read(root/'diagnostics_A'/(condition+'.json'))
                require(all(x['input_sha256']==y['input_sha256'] and x['sample_id']==y['sample_id'] for x,y in zip(records,reference)),
                        'A/B diagnostic input fields differ')
            counts=Counter(g['predicted_verdict'] if g.get('predicted_verdict') else 'invalid' for g in records)
            correct=sum(g['verdict_correct'] for g in records)
            stats={'n':120,'correct':correct,'accuracy':correct/120,'predictions':dict(counts),
                   'format_valid':sum(g['format_valid'] for g in records)}
            if condition=='original':
                stats['exact_outputs']=sum(g['exact_match'] for g in records)
                stats['strata']={}
                for name,select in [('milk_hidden',lambda e:e['milk_hidden_both']),('milk_present',lambda e:not e['milk_hidden_both']),
                                    ('near_conflict_screen',lambda e:e['near_conflict_screen'])]:
                    subset=[g for g in records if select(audit_rows[g['sample_id']])]
                    stats['strata'][name]={'n':len(subset),'correct':sum(g['verdict_correct'] for g in subset)}
            else:
                stats['same_prediction_as_original']=sum(g['predicted_verdict']==original_by_id[g['sample_id']]['predicted_verdict'] for g in records)
            if condition=='shuffled_pairs':
                stats['donor_label_correct']=sum(g['donor_verdict_correct'] for g in records)
                stats['matches_original_donor_prediction']=sum(g['predicted_verdict']==original_by_id[g['donor_id']]['predicted_verdict'] for g in records)
                stats['opposite_label_swaps']=sum(lookup[g['sample_id']]['target']['verdict']!=g['donor_verdict'] for g in records)
            if condition=='horizon_0':
                pos=[g for g in records if lookup[g['sample_id']]['target']['verdict']=='feasible']
                stats['original_positive_now_infeasible']={'correct':sum(g['verdict_correct'] for g in pos),'n':len(pos)}
            if v=='B':
                stats['rationale_status']=dict(Counter(g['rationale_assessment']['status'] for g in records))
                stats['exact_observability_templates']=sum(g['rationale_assessment'].get('exact_observability_template',False) for g in records)
                stats['physical_verdict_justification_established']=0
            for g in records:
                entry={'variant':v,'condition':condition,'sample_id':g['sample_id'],'case':lookup[g['sample_id']]['metadata']['case'],
                    'expected_verdict':g['expected_verdict'],'predicted_verdict':g['predicted_verdict'],
                    'correct':g['verdict_correct'],'format_valid':g['format_valid'],
                    'milk_hidden':audit_rows[g['sample_id']]['milk_hidden_both'],
                    'rationale_status':g['rationale_assessment']['status'] if v=='B' else 'not_applicable',
                    'generated':g['generated']}
                cases.append(entry)
                if not g['format_valid'] or (v=='B' and g['rationale_assessment']['status']!='supported_complete'):
                    reviews.append(dict(entry,assessment=g['rationale_assessment']))
            conditions[condition]=stats
        summary['arms'][v]={'training':result,'conditions':conditions}
        curve=[json.loads(l) for l in (root/v/'curve.jsonl').read_text().splitlines()]
        for col,key in enumerate(('total','rationale','verdict')):
            ax=axes[arm_index,col]
            if v=='A' and key=='rationale':
                ax.text(.5,.5,'No rationale target in A',ha='center',va='center',transform=ax.transAxes)
            else:
                ax.plot([x['step'] for x in curve],[max(x[key],1e-8) for x in curve],alpha=.4,label='update batch (dropout on)')
                lossfiles=sorted((root/v).glob('loss_step_*.json'))
                ax.plot([0]+[int(f.stem.split('_')[-1]) for f in lossfiles],
                        [result['before'][key]]+[read(f)[key] for f in lossfiles],marker='o',label='all120, teacher forced')
                ax.set_yscale('log');ax.legend(fontsize=7)
            ax.set(title=f'{v}: {key} CE',xlabel='Optimizer update',ylabel='Mean CE');ax.grid(alpha=.2)
    fig.suptitle('Fixed 120-sample training fit — one source scene; engineering only')
    fig.savefig(root/'training_curves.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    for v in ('A','B'):
        points=[(0,read(root/v/'generations_before.json'))]+[(s,read(root/v/f'generations_step_{s:04d}.json')) for s in (150,300)]
        ax.plot([s for s,g in points],[sum(x['verdict_correct'] for x in g)/120 for s,g in points],'-o',label=v)
    ax.set(ylim=(0,1.02),xlabel='Optimizer update',ylabel='Greedy training verdict accuracy',title='Same120 training items — no held-out evaluation');ax.legend();ax.grid(alpha=.2)
    fig.savefig(root/'training_accuracy.png',dpi=160);plt.close(fig)
    write(root/'SUMMARY.json',summary);write(root/'INVALID_OR_INCONSISTENT.json',reviews)
    with (root/'outputs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(cases[0]));w.writeheader();w.writerows(cases)
    lines=['# Full120 engineering measurements','',summary['scope']+'.','',
           '| Arm | Updates | Training verdicts | Exact outputs | Total CE | Rationale CE | Verdict CE |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for v in ('A','B'):
        r=summary['arms'][v]['training'];lines.append(f'| {v} | {r["steps"]} | {r["correct_verdicts"]}/120 | {r["exact_outputs"]}/120 | {r["after"]["total"]:.7g} | {r["after"]["rationale"]:.7g} | {r["after"]["verdict"]:.7g} |')
    lines+=['','| Condition | A correct | B correct | B rationale status |','|---|---:|---:|---|']
    for c in CONDITIONS:
        a=summary['arms']['A']['conditions'][c];b=summary['arms']['B']['conditions'][c]
        lines.append(f'| {c} | {a["correct"]}/120 | {b["correct"]}/120 | {b["rationale_status"]} |')
    lines+=['','Shuffled correctness uses recipient labels; see SUMMARY.json for donor-label accuracy. Text-only accuracy measures dataset-prior behavior, not identifiable inference. Zero horizon has all-infeasible derived labels and is a trivial contract-logic probe.',
        '', 'Rationale checks are exact finite-grammar evidence checks on the three visibility statements and limitation sentences. Unknown prose is flagged for manual review. They do not certify physical causal reasoning or object recognizability from tiny fragments.',
        '', 'The immutable dataset and all source evidence are retained. Every row remains excluded from claims of scientifically established visual identifiability.']
    (root/'MEASUREMENTS.md').write_text('\n'.join(lines)+'\n')
    write(root/'COMPLETE.json',{'passed':True,'training_steps_per_arm':300,'rows':120,
        'identical_AB_training_order':True,'identical_AB_diagnostic_inputs':True,
        'scientific_generalization_claim':False})
    destination=Path.home()/'crash_bench/results/verifier_training'/root.name
    destination.mkdir(parents=True,exist_ok=True)
    # Deliver lightweight records only; immutable data, model and adapters stay on Quest.
    skip={'source','dataset','adapter'}
    for path in root.rglob('*'):
        relative=path.relative_to(root)
        if path.is_file() and not any(part in skip for part in relative.parts) and path.name!='source.tar':
            target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    files=sorted(p for p in destination.rglob('*') if p.is_file() and p.name!='DELIVERY.sha256')
    (destination/'DELIVERY.sha256').write_text(''.join(sha(p)+'  '+str(p.relative_to(destination))+'\n' for p in files))
    print(json.dumps({v:summary['arms'][v]['conditions'] for v in ('A','B')}),flush=True)
