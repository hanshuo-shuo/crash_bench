"""Expose every held-out layout and paired difference; no new fitting or selection."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import readout as ro


def write(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def export(analysis,output):
    output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((analysis/'MANIFEST.json').read_text())['rows']
    result=json.loads((analysis/'READOUT.json').read_text());primary=result['primary_vla_representation']
    with np.load(analysis/'TEST_PREDICTIONS.npz') as arrays:
        indices=arrays['indices'];y=arrays['y'];scores={k:arrays[k] for k in arrays.files if k not in ('indices','y')}
    groups=np.asarray([manifest[i]['scene_group'] for i in indices])
    failed=np.asarray([ro.failure_target(manifest[i])==1 for i in indices])
    methods=list(dict.fromkeys([primary,'own_vision_tower','native_final','dino_patch','rgb_pooled_red','visibility','geometry','knows_fixed']))
    metrics=[];paired=[];layouts=[]
    for group in np.unique(groups):
        mask=groups==group;values={}
        for population,sub in [('all_known',mask),('defined_failures',mask&failed)]:
            values[population]={}
            for method in methods:
                measure=ro.metrics(y[sub],scores[method][sub]);values[population][method]=measure
                metrics.append(dict(scene_group=str(group),population=population,representation=method,**measure))
            main=values[population][primary]['auroc']
            for comparator in methods:
                if comparator==primary:continue
                other=values[population][comparator]['auroc']
                paired.append(dict(scene_group=str(group),population=population,primary=primary,comparator=comparator,
                    n=int(sub.sum()),primary_auroc=main,comparator_auroc=other,
                    paired_delta_auroc=main-other if main is not None and other is not None else None,
                    inference='Descriptive within-layout difference; no significance claim'))
        layouts.append(dict(scene_group=str(group),metrics=values))
    incremental=[]
    if (analysis/'INCREMENTAL_PREDICTIONS.npz').exists():
        with np.load(analysis/'INCREMENTAL_PREDICTIONS.npz') as arrays:
            ii=arrays['indices'];yy=arrays['y'];ff=arrays['failed'];first=arrays['safe_only'];second=arrays['safe_plus_feasibility']
        gg=np.asarray([manifest[i]['scene_group'] for i in ii])
        for group in np.unique(gg):
            mask=(gg==group)&ff;a=ro.metrics(yy[mask],first[mask]);b=ro.metrics(yy[mask],second[mask])
            incremental.append(dict(scene_group=str(group),n=a['n'],infeasible=a.get('positive'),
                initial_failure_score_only_auroc=a['auroc'],failure_plus_feasibility_auroc=b['auroc'],
                paired_delta_auroc=b['auroc']-a['auroc'] if a['auroc'] is not None and b['auroc'] is not None else None,
                comparator='SAFE-style initial-state failure probe',scope='t=0, retrospective defined-failure subset; descriptive'))
        write(output/'layout_incremental.csv',incremental)
    write(output/'layout_readouts.csv',metrics);write(output/'layout_paired_differences.csv',paired)
    value=dict(primary_representation=primary,layouts=layouts,paired_differences=paired,incremental=incremental,
        reporting_amendment=json.loads((Path(__file__).parent/'reporting_amendment.json').read_text()),
        interpretation='Four layout-specific comparisons are descriptive. Original bootstrap files are preserved as exploratory appendix output.')
    (output/'LAYOUT_COMPARISONS.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('analysis',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();export(args.analysis,args.output)
