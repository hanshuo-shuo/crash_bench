"""Dated, explicitly supplemental nuisance checks on the unchanged saved matrix."""
import argparse
import collections
import csv
import json
from pathlib import Path
import numpy as np
import grouped_analysis as ga
import readout as ro

PROTOCOL=json.loads((Path(__file__).parent/'nuisance_protocol.json').read_text())


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def distribution(scores):
    scores=np.asarray(scores)
    if not len(scores):return dict(n=0)
    return dict(n=len(scores), minimum=float(scores.min()), maximum=float(scores.max()),
        mean=float(scores.mean()), q25=float(np.quantile(scores,.25)),
        median=float(np.median(scores)), q75=float(np.quantile(scores,.75)),
        fraction_score_ge_half=float((scores>=.5).mean()),
        interpretation='Unverified readout scores; no UNKNOWN accuracy or inferred true label')


def analyze(analysis, output):
    output.mkdir()
    manifest=json.loads((analysis/'MANIFEST.json').read_text());rows=manifest['rows']
    ro.check_groups(rows)
    for row in rows:ga.validate_outcome(row)
    with np.load(analysis/'FEATURES.npz') as arrays:
        width=arrays['geometry'][:,[6]].copy()
        visibility=arrays['visibility'][:,[0,3]].copy()
    expected=np.asarray([.020 if r['variant'] in ('open','irrelevant') else float(r['variant']) for r in rows])
    deviation=float(np.abs(width[:,0]-expected).max())
    if deviation>1e-8:raise ValueError('Nominal width input does not match the declared source convention')
    inputs=dict(nominal_width_only=width,target_visibility_only=visibility,
        width_plus_target_visibility=np.concatenate((width,visibility),axis=1))
    if any(not np.isfinite(x).all() for x in inputs.values()):raise ValueError('Nonfinite nuisance inputs')
    y=np.asarray([int(r['label']=='infeasible') for r in rows]);groups=np.asarray([r['scene_group'] for r in rows])
    failure=np.asarray([ro.failure_target(r) if ro.failure_target(r) is not None else -1 for r in rows])
    unknown=np.asarray([i for i,r in enumerate(rows) if r['label']=='unknown' and r['initial_valid']])
    records={};table=[];known_predictions=[];unknown_predictions=[];unknown_distributions=[]
    for population in PROTOCOL['populations']:
        def eligible(r):
            return r['initial_valid'] and (population=='all_known_variants' or r['variant'] not in ('open','irrelevant'))
        indices={split:np.asarray([i for i,r in enumerate(rows) if r['split']==split and
            r['family']==('cage' if split=='test' else 'slit') and eligible(r) and r['label']!='unknown'])
            for split in ('train','validation','test')}
        tr,va,te=[indices[s] for s in ('train','validation','test')]
        if any(rows[i]['label']=='unknown' for ids in indices.values() for i in ids):raise RuntimeError('UNKNOWN entered binary fitting/evaluation')
        if set(groups[tr]) & set(groups[te]) or set(groups[va]) & set(groups[te]):raise RuntimeError('Test group leakage')
        coverage={}
        for split in indices:
            selected=[r for r in rows if r['split']==split and r['family']==('cage' if split=='test' else 'slit') and eligible(r)]
            n_unknown=sum(r['label']=='unknown' for r in selected)
            coverage[split]=dict(total=len(selected),known=len(indices[split]),unknown=n_unknown,
                unknown_fraction=n_unknown/len(selected),groups=sorted({r['scene_group'] for r in selected}))
        prior=float(y[tr].mean());prior_score=np.full(len(te),prior);prior_failed=failure[te]==1
        constructor=dict(status='No trained cross-category mapping: fitting constructor is constant slit; test constructor is unseen cage',
            training_prior=prior,constant_reference_test=ro.metrics(y[te],prior_score),
            constant_reference_failed_subset=ro.metrics(y[te][prior_failed],prior_score[prior_failed]),
            width_plus_constructor='Constructor effects are not identifiable in these fitting groups; width-only is the estimable metadata diagnostic')
        models={}
        for name,x in inputs.items():
            model,selection=ro.fit_selected(x[tr],y[tr],x[va],y[va])
            score=ga.probability(model,x[te]);failed=failure[te]==1
            metrics=ro.metrics(y[te],score);ci=ro.cluster_interval(y[te],score,groups[te],seed=713)
            failed_metrics=ro.metrics(y[te][failed],score[failed])
            failed_ci=ro.cluster_interval(y[te][failed],score[failed],groups[te][failed],seed=713)
            families={}
            for kind in ('linear','mlp'):
                chosen=min((t for t in selection['trials'] if t['name'].startswith(kind)),key=lambda t:t['validation_log_loss'])
                head=dict(ro.candidates())[chosen['name']]
                fitted=ga.fit_fixed(x[tr],y[tr],head)
                families[kind]=dict(selection=chosen,test=ro.metrics(y[te],ga.probability(fitted,x[te])))
            models[name]=dict(selection=selection,test=metrics,auroc_interval=ci,
                failed_subset=failed_metrics,failed_subset_interval=failed_ci,head_families=families)
            models[name]['test_layouts']={str(g):dict(all_known=ro.metrics(y[te][groups[te]==g],score[groups[te]==g]),
                defined_failures=ro.metrics(y[te][(groups[te]==g)&failed],score[(groups[te]==g)&failed])) for g in np.unique(groups[te])}
            for endpoint,metric,interval in [('all_known',metrics,ci),('defined_policy_failures',failed_metrics,failed_ci)]:
                bounds=interval['interval']
                table.append(dict(population=population,diagnostic=name,evaluation=endpoint,
                    train_n=len(tr),validation_n=len(va),selected_head=selection['selected'],**metric,
                    auroc_ci_low=bounds[0] if bounds else None,auroc_ci_high=bounds[1] if bounds else None))
            for i,p in zip(te,score):
                known_predictions.append(dict(population=population,diagnostic=name,case=rows[i]['case'],
                    scene_group=rows[i]['scene_group'],label=rows[i]['label'],outcome=rows[i]['outcome'],score=float(p)))
            unknown_scores=ga.probability(model,x[unknown])
            for i,p in zip(unknown,unknown_scores):
                r=rows[i]
                unknown_predictions.append(dict(population=population,diagnostic=name,case=r['case'],
                    scene_group=r['scene_group'],split=r['split'],constructor=r['family'],variant=r['variant'],
                    label='unknown',score=float(p),binary_accuracy=None))
            for split in ('train','validation','test'):
                for family in ('slit','cage'):
                    for variant in ('.034','.060'):
                        keep=np.asarray([rows[i]['split']==split and rows[i]['family']==family and rows[i]['variant']==variant for i in unknown])
                        unknown_distributions.append(dict(population=population,diagnostic=name,
                            split=split,constructor=family,variant=variant,**distribution(unknown_scores[keep])))
            primary_unknown=np.asarray([rows[i]['split']=='test' and rows[i]['family']=='cage' for i in unknown])
            models[name]['primary_test_unknown_scores']=distribution(unknown_scores[primary_unknown])
        records[population]=dict(indices={k:v.tolist() for k,v in indices.items()},coverage=coverage,models=models,
            constructor_diagnostic=constructor)
    dump(output/'NUISANCE_DIAGNOSTICS.json',dict(protocol=PROTOCOL,source_analysis=str(analysis),
        source_width_max_deviation=deviation,all_initial_states=len(rows),all_unknown=sum(r['label']=='unknown' for r in rows),
        populations=records,unknown_accuracy_computed=False))
    write_csv(output/'nuisance_metrics.csv',table)
    write_csv(output/'known_predictions.csv',known_predictions)
    write_csv(output/'unknown_predictions.csv',unknown_predictions)
    dump(output/'UNKNOWN_SCORE_DISTRIBUTIONS.json',unknown_distributions)
    counts=collections.Counter((r['split'],r['family'],r['variant'],r['label'],r['outcome']) for r in rows)
    write_csv(output/'constructor_width_descriptive_counts.csv',[dict(split=k[0],constructor=k[1],variant=k[2],
        initial_label=k[3],policy_outcome=k[4],n=n,scope='Descriptive counts only; no test-fit model') for k,n in sorted(counts.items())])
    np.savez_compressed(output/'NUISANCE_INPUTS.npz',**inputs)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('analysis',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();analyze(args.analysis,args.output)
