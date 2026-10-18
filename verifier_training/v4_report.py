"""Preregistered A/B/C report (docs/generative_verifier/PLAN.md section 5; configs/v4_dev_prereg.json).

CPU only, from saved generations: python -m verifier_training.v4_report RUN validation|test
Rows are paired, correctness is averaged over seeds per row, and sources are the clusters.
"""
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
from .common import read,write,sha,require
from .fd1000_contract import readings,validate
from .fd1000_metrics import summarize,clustered_difference,rule_prediction,auroc,weak_shortcut,stratum,KEY_FIELDS

PRIMARY=(('B','facts_free','verdict'),('A','direct_free','verdict'))
SECONDARY={'C facts_forced - C direct_forced':(('C','facts_forced','verdict'),('C','direct_forced','verdict')),
    'C direct_forced - A direct_free':(('C','direct_forced','verdict'),('A','direct_free','verdict')),
    'B facts + program rule - B facts_free verdict':(('B','facts_free','rule'),('B','facts_free','verdict'))}
GREEDY_FACTS={'B':'facts_free','C':'facts_forced'}
SCALARS=('verdict_accuracy','unit_all_correct','rule_flip_consistency','color_renaming_invariance','false_feasible_rate',
         'false_infeasible_rate','auroc_invalid_neutral')
MILESTONE=dict(verdict_accuracy=.85,unit_all_correct=.70,rule_flip_consistency=.90)


def load(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]


def arm_readings(config):
    """Readings that represent an arm for the milestone; oracle, edited and sampled readings are diagnostics."""
    return {f'{a} {r}' for a in config['arms'] for r in readings(a)}


def mean(values):
    values=[v for v in values if v is not None];return sum(values)/len(values) if values else None


def run_records(root,split,label):
    if split=='validation':
        folder=root/label;result=read(folder/'RESULT.json')
        require(result['passed'] and result['test_evaluations']==0,'Incomplete or test-contaminated run')
        records=load(folder/f'validation_step_{result["selected_step"]}.jsonl')
        return records+(load(folder/'validation_diagnostics.jsonl') if (folder/'validation_diagnostics.jsonl').exists() else [])
    done=read(root/'test'/label/'COMPLETE.json');require(done['passed'] and done['evaluations']==1,'Test evaluation incomplete')
    return load(root/'test'/label/'records.jsonl')


def decision(record,mode,k=None):
    if mode=='rule':return rule_prediction(record['facts'])
    if mode=='sampled':
        p=sum(s['p_feasible'] if s['p_feasible'] is not None else .5 for s in record['samples'][:k])/k
        return 1 if p>.5 else 0 if p<.5 else None
    return record['predicted']


def correct_map(records,reading,mode,k=None):
    return {r['sample_id']:float(decision(r,mode,k)==r['truth']) for r in records if r['reading']==reading}


def seed_average(maps):
    keys=set(maps[0]);require(all(set(m)==keys for m in maps),'Seeds cover different rows')
    return {s:sum(m[s] for m in maps)/len(maps) for s in keys}


def comparison(runs,treatment,control,groups,k=None):
    def averaged(spec):
        arm,reading,mode=spec;return seed_average([correct_map(runs[(arm,seed)],reading,mode,k) for seed in seeds(runs,arm)])
    t=averaged(treatment);c=averaged(control)
    result=clustered_difference(c,t,groups)
    result.update(treatment=list(treatment),control=list(control),treatment_accuracy=sum(t.values())/len(t),control_accuracy=sum(c.values())/len(c))
    return result


def seeds(runs,arm):return sorted(s for a,s in runs if a==arm)


def arm_summaries(runs,config):
    """Per reading: seed means of the scalar metrics, strata and fact fields, plus pooled AUROC of seed-averaged P(1)."""
    out={}
    for arm in sorted({a for a,s in runs}):
        per_seed={seed:summarize(runs[(arm,seed)],config) for seed in seeds(runs,arm)}
        for reading in per_seed[seeds(runs,arm)[0]]:
            parts=[per_seed[s][reading] for s in per_seed];entry=dict(seeds=len(parts))
            if 'verdict_accuracy' in parts[0]:
                for key in SCALARS:
                    entry[key]=mean(p[key] for p in parts)
                    entry[key+'_by_seed']=[p[key] for p in parts]
                entry['format_valid']=[p['format_valid'] for p in parts]
                entry['weak_shortcut_accuracy']=mean(p['weak_shortcut_subset']['accuracy'] for p in parts)
                entry['strata_accuracy']={k:mean(p['strata'][k]['accuracy'] for p in parts) for k in parts[0]['strata']}
                if 'facts' in parts[0]:
                    entry['facts']={k:sum(p['facts'][k] for p in parts)/len(parts) for k in parts[0]['facts'] if k.endswith('accuracy')}
                    entry['fact_cases']={k:sum(p['facts']['cases'][k] for p in parts) for k in parts[0]['facts']['cases'] if k!='meaning'}
                pooled=defaultdict(list);truth={}
                for seed in per_seed:
                    for r in runs[(arm,seed)]:
                        if r['reading']==reading:pooled[r['sample_id']].append(r['p_feasible'] if r['p_feasible'] is not None else .5);truth[r['sample_id']]=r['truth']
                entry['pooled_auroc_seed_mean_p']=auroc([truth[s] for s in pooled],[sum(v)/len(v) for v in pooled.values()])
                entry['mean_generation_seconds']=sum(p['mean_generation_seconds'] for p in parts)/len(parts)
            elif reading.endswith('_sampled'):
                entry.update({k:{m:mean(p[k][m] for p in parts) for m in ('mean_p_accuracy','mean_p_auroc','vote_accuracy')}
                              for k in parts[0] if k.isdigit()})
                entry['mean_generation_seconds']=sum(p['mean_generation_seconds'] for p in parts)/len(parts)
            else:
                entry.update({k:sum(p[k] for p in parts)/len(parts) for k in ('follows_edited_text','keeps_image_label')})
            out[f'{arm} {reading}']=entry
    return out


def diagnostics_by_correctness(runs):
    """Oracle and edited-premise diagnostics, split by whether the greedy fact reading was right (prereg)."""
    out={}
    for arm,reading in GREEDY_FACTS.items():
        if not seeds(runs,arm):continue
        counts=defaultdict(lambda:defaultdict(lambda:[0,0]))
        for seed in seeds(runs,arm):
            records=runs[(arm,seed)];greedy={r['sample_id']:r for r in records if r['reading']==reading}
            for r in records:
                if r['reading'] not in ('oracle_facts','edit_target','edit_goal'):continue
                group='greedy_correct' if greedy[r['sample_id']]['predicted']==r['truth'] else 'greedy_wrong'
                hit=r['predicted']==(r['truth'] if r['reading']=='oracle_facts' else r['text_implied_verdict'])
                counts[r['reading']][group][0]+=int(hit);counts[r['reading']][group][1]+=1
            for r in greedy.values():
                group='greedy_correct' if r['predicted']==r['truth'] else 'greedy_wrong'
                for key in KEY_FIELDS:
                    counts['field_'+key][group][0]+=int((r['facts'] or {}).get(key)==r['truth_facts'][key]);counts['field_'+key][group][1]+=1
        out[arm]={name:{g:dict(rate=v[0]/v[1] if v[1] else None,rows=v[1]) for g,v in groups.items()} for name,groups in counts.items()}
        out[arm]['meaning']='oracle_facts: verdict correct after gold facts; edit_*: verdict follows the edited text; field_*: fact accuracy'
    return out


def readout_scores(rows,scores):
    y=[int(r['target']['verdict']=='feasible') for r in rows];strata=defaultdict(lambda:([],[]))
    for r,s,t in zip(rows,scores,y):
        m=r['metadata'];mode,side,checkpoint=stratum(dict(construction_mode=m['construction_mode'],positive_side=m['positive_side'],checkpoint=m['checkpoint']))
        strata[mode+' x '+checkpoint][0].append(t);strata[mode+' x '+checkpoint][1].append(s)
    weak=[(t,s) for r,s,t in zip(rows,scores,y) if weak_shortcut(dict(construction_mode=r['metadata']['construction_mode'],checkpoint=r['metadata']['checkpoint']))]
    return dict(rows=len(rows),auroc=auroc(y,scores),accuracy_at_05=sum((s>=.5)==bool(t) for s,t in zip(scores,y))/len(y),
        strata_auroc={k:auroc(*v) for k,v in sorted(strata.items())},weak_shortcut_auroc=auroc([t for t,s in weak],[s for t,s in weak]))


def readout_baseline(dataset,rows,split):
    """Fixed color/rule linear readouts from the failed screen; refit on train for the test report."""
    by={r['sample_id']:r for r in rows}
    if split=='validation':
        stored=read(dataset/'SHORTCUT_VALIDATION_PREDICTIONS.json')
        return {name:readout_scores([by[s] for s in stored['sample_ids']],stored['predictions'][name])
                for name in ('color_statistics_rule_interactions_linear','rgb_color_rule_interactions_linear')}
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from verifier_dataset.constraint_shortcuts import color_interaction_features
    z=np.load(dataset/'SHORTCUT_FEATURES.npz',allow_pickle=False);ids=[str(s) for s in z['sample_ids']];require(set(ids)==set(by),'Readout rows differ')
    y=np.asarray([int(by[s]['target']['verdict']=='feasible') for s in ids]);part=np.asarray([by[s]['split'] for s in ids])
    stored=read(dataset/'SHORTCUT_VALIDATION_PREDICTIONS.json');out={}
    for name,x in (('color_statistics_rule_interactions_linear',color_interaction_features(z['joint'],'color_statistics')),
                   ('rgb_color_rule_interactions_linear',z['joint'])):
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000,random_state=1014)).fit(x[part=='train'],y[part=='train'])
        column=list(model.classes_).index(1);val={s:p for s,p in zip(np.asarray(ids)[part=='validation'],model.predict_proba(x[part=='validation'])[:,column])}
        # lbfgs tolerance and BLAS threading move probabilities by ~1e-3 between nodes; a different readout would move them far more.
        gap=max(abs(val[s]-p) for s,p in zip(stored['sample_ids'],stored['predictions'][name]))
        require(gap<.02,'Refit readout differs from the frozen screen')
        test_ids=[s for s,t in zip(ids,part) if t=='test']
        out[name]=dict(readout_scores([by[s] for s in test_ids],model.predict_proba(x[part=='test'])[:,column].tolist()),
                       refit_max_abs_gap_to_frozen_validation_predictions=float(gap))
    return out


def markdown(report):
    pct=lambda x:'—' if x is None else f'{100*x:.1f}'
    lines=[f'# V4 A/B/C report: {report["split"]}','',f'Source commit `{report["source_commit"]}`; seeds {report["seeds"]}; rows are seed-averaged and clustered by source.','',
        '| arm reading | accuracy | unit all-correct | rule-flip | color invariance | false-feasible | AUROC (seed mean) | weak subset | all 4 facts |','|---|---|---|---|---|---|---|---|---|']
    for name,e in report['arms'].items():
        if 'verdict_accuracy' in e:
            lines.append(f"| {name} | {pct(e['verdict_accuracy'])} | {pct(e['unit_all_correct'])} | {pct(e['rule_flip_consistency'])} | "
                f"{pct(e['color_renaming_invariance'])} | {pct(e['false_feasible_rate'])} | {pct(e['auroc_invalid_neutral'])} | "
                f"{pct(e['weak_shortcut_accuracy'])} | {pct((e.get('facts') or {}).get('all_key_fields_accuracy'))} |")
    lines+=['','| comparison | treatment | control | difference | source bootstrap 95% | sign-flip p | Durkalski p |','|---|---|---|---|---|---|---|']
    for name,c in [('PRIMARY B facts_free - A direct_free',report['primary'])]+list(report['secondary'].items()):
        lo,hi=c['source_bootstrap_95pct']
        lines.append(f"| {name} | {pct(c['treatment_accuracy'])} | {pct(c['control_accuracy'])} | {100*c['mean_difference']:+.1f} | "
            f"[{100*lo:+.1f}, {100*hi:+.1f}] | {c['sign_flip']['p_two_sided']:.3f} | {c['durkalski_mcnemar']['p']:.3f} |")
    lines+=['',f"Development milestone met: {report['success']['development_milestone_met']}; scientific criterion met: {report['success']['scientific_met']}.",'',
        'Smaller differences than about 15-25 points at the unit level are exploratory (PLAN.md section 5).']
    return '\n'.join(lines)+'\n'


def main(root,split):
    receipt=read(root/'RECEIPT.json');require(os.environ.get('CB_CODE_COMMIT')==receipt['source_commit'],'Report source differs from the campaign')
    require(split in ('validation','test'),'Unknown split');out=root/f'REPORT_{split}.json';require(not out.exists(),'Report already written')
    config=read(root/'config.json');dataset=Path(receipt['dataset']);require(sha(dataset/'DEV_ACCEPTANCE.json')==receipt['dataset_acceptance_sha256'],'Dataset changed')
    rows,_=validate(dataset,mode='development',include_test=split=='test')
    if split=='test':
        freeze=read(root/'TEST_FREEZE.json');require(freeze['prereg_sha256']==sha(root/'source/verifier_training/configs/v4_dev_prereg.json'),'Preregistration changed')
    runs={(a,s):run_records(root,split,f'{a}_seed{s}') for a in config['arms'] for s in config['seeds']}
    groups={r['sample_id']:r['source_group'] for records in runs.values() for r in records}
    secondary={name:comparison(runs,t,c,groups) for name,(t,c) in SECONDARY.items()}
    for k in config['sampling']['k_values']:
        secondary[f'B facts_free_sampled K={k} - B facts_free']=comparison(runs,('B','facts_free_sampled','sampled'),('B','facts_free','verdict'),groups,k=k)
    arms=arm_summaries(runs,config);primary=comparison(runs,*PRIMARY,groups)
    met={name:all(e.get(k) is not None and e[k]>=v for k,v in MILESTONE.items()) for name,e in arms.items() if name in arm_readings(config)}
    report=dict(split=split,source_commit=receipt['source_commit'],seeds=config['seeds'],primary=primary,secondary=secondary,arms=arms,
        diagnostics_by_greedy_correctness=diagnostics_by_correctness(runs),
        baselines=dict(majority_class_accuracy=.5,readouts=readout_baseline(dataset,rows,split),
            zero_shot=summarize(run_records(root,split,'zero_shot'),config) if split=='test' else 'See PHASE1_REPORT.md (validation, Phase 1 campaign)'),
        selection={f'{a}_seed{s}':read(root/f'{a}_seed{s}'/'RESULT.json')['checkpoint_values'] for a in config['arms'] for s in config['seeds']},
        success=dict(milestone_by_reading=met,development_milestone_met=any(met.values()) if split=='test' else None,
            scientific_met=(primary['source_bootstrap_95pct'][0]>0 or primary['source_bootstrap_95pct'][1]<0) if split=='test' else None,
            thresholds=MILESTONE,note='Success criteria apply to the locked test only; validation values are descriptive'),
        prereg_sha256=sha(root/'source/verifier_training/configs/v4_dev_prereg.json'),api_calls=0,slurm_job=os.environ.get('SLURM_JOB_ID'))
    write(out,report);(root/f'REPORT_{split}.md').write_text(markdown(report))


def correct_milestone(root,split,commit):
    """Rejudge the milestone of an existing report without touching it (e03da26 counted oracle diagnostics)."""
    report=read(root/f'REPORT_{split}.json');config=read(root/'config.json');out=root/f'REPORT_{split}_CORRECTION.json'
    require(not out.exists(),'Correction already written')
    met={name:all(e.get(k) is not None and e[k]>=v for k,v in MILESTONE.items()) for name,e in report['arms'].items() if name in arm_readings(config)}
    corrected=dict(report['success'],milestone_by_reading=met,development_milestone_met=any(met.values()) if split=='test' else None)
    write(out,dict(split=split,report_sha256=sha(root/f'REPORT_{split}.json'),original_success=report['success'],corrected_success=corrected,
        reason='The original report judged the milestone over every reading, so the oracle_facts diagnostics (gold facts forced before '
            'Verdict:) counted as arms. The preregistration applies it to arms; only A direct_free, B facts_free, C direct_forced and '
            'C facts_forced qualify. No prediction or other metric changes.',
        correction_commit=commit,report_commit=report['source_commit'],api_calls=0))
    text=markdown(dict(report,success=corrected))
    (root/f'REPORT_{split}_CORRECTED.md').write_text(text+'\nCorrected milestone judgement; see REPORT_'+split+'_CORRECTION.json.\n')


if __name__=='__main__':
    if sys.argv[1]=='correct-milestone':
        from .launch_fd1000 import published_commit
        correct_milestone(Path(sys.argv[2]),sys.argv[3],published_commit());sys.exit()
    root=Path(sys.argv[1]);split=sys.argv[2]
    try:main(root,split)
    except BaseException as error:
        write(root/f'STOP_report_{split}.json',dict(error=type(error).__name__,message=str(error),slurm_job=os.environ.get('SLURM_JOB_ID')));raise
