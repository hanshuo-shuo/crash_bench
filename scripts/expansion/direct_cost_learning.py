#!/usr/bin/env python3
"""Existing-data paired Ridge development: source-disjoint A fit, B evaluation."""
from __future__ import annotations
import argparse
import hashlib
import json
import pickle
import platform
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from crashbench.models.advantage import _weighted_ridge
from scripts.expansion.selection_retest import validate_records
EVIDENCE=ROOT/'docs/audits/20260908/candidate_refresh/verified_evidence'
CONTEXTS=ROOT/'results/direct_cost_learning/contexts'
TARGETS=('completion_cost','inference_calls','success','accident')
SCALE=np.array([200.,40.,1.,1.])
RULES=('Base','AlwaysRefresh','age1','age3','age5','period10','age_equal1')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x') as f: json.dump(value,f,indent=2,sort_keys=True);f.write('\n')


def load_context(path):
    with Path(path).open('rb') as f: raw=pickle.load(f)
    # Deliberate allowlist. No fresh field or policy continuation enters features.
    return {k:raw[k] for k in ('delivered','timing','chunk_remaining','nominal_proposal')}


def features(anchor,context,kind):
    age=context['timing']['age_steps']
    if age != anchor['age_steps']: raise ValueError('timing mismatch')
    if context['chunk_remaining'] != 0 or context['nominal_proposal'] is not None:
        raise ValueError('unexpected action context: protocol must be reviewed')
    meta=np.array([float(anchor['task_id']=='libero_spatial:2'),age,anchor['anchor_steps']],float)
    if kind=='metadata': return meta
    if kind!='observation': raise ValueError(kind)
    obs=context['delivered'];blocks=[meta,np.asarray(obs['state'],float)]
    if blocks[-1].shape != (8,): raise ValueError('unexpected proprio')
    for key in ('full_image','wrist_image'):
        image=np.asarray(obs[key])
        if image.shape!=(256,256,3) or image.dtype!=np.uint8: raise ValueError('unexpected image contract')
        blocks.append(image.reshape(4,64,4,64,3).mean(axis=(1,3)).ravel()/255.)
    blocks.append(np.array([context['chunk_remaining']],float))
    return np.concatenate(blocks)


def block_scales(kind):
    widths=[3] if kind=='metadata' else [3,8,48,48,1]
    return np.concatenate([np.full(w,np.sqrt(w)) for w in widths])


def targets(record,horizon=200):
    r=record['horizons'][str(horizon)]
    return np.array([r['steps'] if r['task_success'] else horizon,
                     r['inference_calls'],r['task_success'],r['catastrophe']],float)


def paired_means(anchors,records,phase):
    validate_records(anchors,records,phase)
    return np.array([np.mean([targets(r) for r in records if r['panel_id']==a['panel_id'] and r['option']==1],axis=0)-
        np.mean([targets(r) for r in records if r['panel_id']==a['panel_id'] and r['option']==0],axis=0) for a in anchors])


def fit_fold(x,y,sources,heldout,kind):
    mask=np.asarray(sources)!=heldout
    src=np.asarray(sources)[mask]; unique=sorted(set(src))
    if heldout in unique or not len(unique): raise ValueError('source leakage')
    w=np.array([len(src)/(len(unique)*sum(src==s)) for s in src])
    mean=np.average(x[mask],axis=0,weights=w)
    scale=np.sqrt(np.average((x[mask]-mean)**2,axis=0,weights=w))
    scale[scale<1e-8]=1.
    scale=scale*block_scales(kind)
    z=(x[mask]-mean)/scale
    coefficients=np.stack([_weighted_ridge(z,y[mask,j]/SCALE[j],w,1.) for j in range(4)],axis=1)
    return {'heldout_source':heldout,'training_sources':unique,'training_indices':np.flatnonzero(mask).tolist(),
            'mean':mean.tolist(),'scale':scale.tolist(),'coefficients':coefficients.tolist(),'kind':kind,'ridge':1.}


def predict(model,x):
    c=np.asarray(model['coefficients']);return (c[0]+((x-np.asarray(model['mean']))/np.asarray(model['scale']))@c[1:])*SCALE


def choose(prediction,guarded=True):
    p=np.asarray(prediction)/SCALE
    return int(p[0]<-1e-8 and p[1]<-1e-8 and (not guarded or p[2]>=-1e-8 and p[3]<=1e-8))


def rule(name,anchor):
    if name=='Base':return 0
    if name=='AlwaysRefresh':return 1
    if name=='period10':return int(anchor['anchor_steps']%10==0)
    if name=='age_equal1':return int(anchor['age_steps']==1)
    if name in ('age1','age3','age5'):return int(anchor['age_steps']>=int(name[3:]))
    raise ValueError(name)


def fit(anchors,a,contexts):
    y=paired_means(anchors,a,'A');sources=[r['physical_source_id'] for r in anchors]
    if len(set(sources))!=8 or len(anchors)!=18:raise ValueError('panel changed')
    result={'models':{},'predictions':{},'choices':{},'targets':list(TARGETS),'A_target_deltas':y.tolist()}
    for name in RULES:result['choices'][name]={r['panel_id']:rule(name,r) for r in anchors}
    for kind in ('metadata','observation'):
        x=np.array([features(r,contexts[r['panel_id']],kind) for r in anchors])
        models={};predictions={}
        for source in sorted(set(sources)):
            model=fit_fold(x,y,sources,source,kind);models[source]=model
            for i,r in enumerate(anchors):
                if r['physical_source_id']==source:predictions[r['panel_id']]=predict(model,x[i]).tolist()
        result['models'][kind]=models;result['predictions'][kind]=predictions
        for suffix,guard in [('',True),('_ungated_diagnostic',False)]:
            result['choices'][kind+suffix]={k:choose(v,guard) for k,v in predictions.items()}
    return result


def evaluate(anchors,b,frozen):
    validate_records(anchors,b,'B')
    lookup={(r['panel_id'],r['repeat'],r['option']):r for r in b}
    actual=paired_means(anchors,b,'B')
    result={'methods':{},'anchor_predictions':[]}
    for name,choices in frozen['choices'].items():
        source_rows=[]
        for source in sorted({a['physical_source_id'] for a in anchors}):
            for condition in ('all','stale','matched_buffer_control'):
                selected=[a for a in anchors if a['physical_source_id']==source and (condition=='all' or a['condition']==condition)]
                rows=[]
                for anchor in selected:
                    key=anchor['panel_id'];option=choices[key]
                    for repeat in range(4,8):
                        base=lookup[key,repeat,0];r=lookup[key,repeat,option]
                        v=targets(r);v0=targets(base)
                        rows.append(np.r_[v,option,r['steps'],v-v0,
                            int(v0[2] and not v[2]),int(not v0[3] and v[3]),
                            r['horizons']['100']['task_success'],r['horizons']['100']['catastrophe']])
                values=np.asarray(rows)
                keys=[*TARGETS,'refresh','raw_steps',*[x+'_delta' for x in TARGETS],
                      'lost_success','new_accident','success_100','accident_100']
                source_rows.append(dict(source=source,condition=condition,n=len(rows),**dict(zip(keys,values.mean(axis=0).tolist()))))
        summary={}
        for condition in ('all','stale','matched_buffer_control'):
            rows=[r for r in source_rows if r['condition']==condition]
            numeric=[k for k in rows[0] if k not in ('source','condition','n')]
            summary[condition]={k:float(np.mean([r[k] for r in rows])) for k in numeric}
            summary[condition]['drop_one_source_sensitivity']={s:{k:float(np.mean([r[k] for r in rows if r['source']!=s])) for k in ('completion_cost_delta','inference_calls_delta','success_delta','accident_delta')} for s in sorted({r['source'] for r in rows})}
        result['methods'][name]={'source_macro':summary,'per_source':source_rows}
    for i,anchor in enumerate(anchors):
        result['anchor_predictions'].append({'panel_id':anchor['panel_id'],'source':anchor['physical_source_id'],
            'condition':anchor['condition'],'actual_B_delta':actual[i].tolist(),
            'predicted':{k:v[anchor['panel_id']] for k,v in frozen['predictions'].items()},
            'choices':{k:v[anchor['panel_id']] for k,v in frozen['choices'].items()}})
    return result


def benchmark(anchors,contexts,frozen):
    rows=[]
    for name in frozen['choices']:
        for anchor in anchors:
            context=contexts[anchor['panel_id']]
            if name in RULES: action=lambda:rule(name,anchor)
            else:
                kind=name.split('_')[0];model=frozen['models'][kind][anchor['physical_source_id']]
                action=lambda:choose(predict(model,features(anchor,context,kind)),not name.endswith('_diagnostic'))
            for _ in range(10):action()
            timings=[]
            for _ in range(200):
                start=time.perf_counter_ns();action();timings.append((time.perf_counter_ns()-start)/1e6)
            rows.append({'method':name,'panel_id':anchor['panel_id'],'source':anchor['physical_source_id'],
                         'median_ms':float(np.median(timings)),'p95_ms':float(np.quantile(timings,.95))})
    return {'platform':platform.platform(),'python':sys.version,'numpy':np.__version__,
            'scope':'warm in-memory feature extraction + preprocessing + four heads + decision; excludes disk and sensor acquisition',
            'rows':rows}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['fit','evaluate']);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output
    anchors=read(EVIDENCE/'anchors.json')
    if args.stage=='fit':
        out.mkdir(parents=True,exist_ok=False)
        contexts={r['panel_id']:load_context(CONTEXTS/r['panel_id']/'selector_context.pkl') for r in anchors}
        frozen=fit(anchors,read(EVIDENCE/'A.json'),contexts)
        paths=[EVIDENCE/'anchors.json',EVIDENCE/'A.json',Path(__file__),ROOT/'crashbench/models/advantage.py',ROOT/'docs/audits/20260908/direct_cost_learning/PLAN.md']+[CONTEXTS/r['panel_id']/'selector_context.pkl' for r in anchors]
        frozen['input_sha256']={str(p.relative_to(ROOT)):sha(p) for p in paths}
        frozen['protocol']='post-B-objective development; fixed LOSO A-only fit; paired Ridge alpha=1'
        write(out/'freeze.json',frozen);(out/'freeze.sha256').write_text(sha(out/'freeze.json')+'\n')
        print('Frozen 8 folds, 2 feature ablations; B not read')
    else:
        if sha(out/'freeze.json')!=(out/'freeze.sha256').read_text().strip():raise ValueError('freeze changed')
        frozen=read(out/'freeze.json')
        for rel,digest in frozen['input_sha256'].items():
            if sha(ROOT/rel)!=digest:raise ValueError('input changed: '+rel)
        metrics=evaluate(anchors,read(EVIDENCE/'B.json'),frozen)
        write(out/'metrics.json',metrics)
        contexts={r['panel_id']:load_context(CONTEXTS/r['panel_id']/'selector_context.pkl') for r in anchors}
        write(out/'timing.json',benchmark(anchors,contexts,frozen))
        write(out/'evaluation_manifest.json',{'B_sha256':sha(EVIDENCE/'B.json'),'freeze_sha256':sha(out/'freeze.json'),
            'metrics_sha256':sha(out/'metrics.json'),'timing_sha256':sha(out/'timing.json'),'new_rollouts':0,'test_outcomes_read':0})
        for name,m in metrics['methods'].items():print(name,json.dumps(m['source_macro']['all'],sort_keys=True).split('drop_one_source_sensitivity')[0])

if __name__=='__main__':main()
