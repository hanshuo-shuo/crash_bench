"""Grouped external-probe fitting. UNKNOWN labels are retained, never binarized.

This is analysis machinery, not a claim that a valid held-out matrix exists.
Feature/layer selection and thresholds use train/validation groups only.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def check_groups(rows):
    seen = {}
    for row in rows:
        group, split = row['scene_group'], row['split']
        if split not in ('train','validation','test'):raise ValueError('Unrecognized split')
        if group in seen and seen[group] != split:raise ValueError('Paired scene group crosses split')
        seen[group] = split
    if set(seen.values()) != {'train','validation','test'}:raise ValueError('Need separate train/validation/test groups')
    return seen


def metrics(y, score, threshold=.5):
    y=np.asarray(y);score=np.asarray(score)
    if not len(y):return dict(n=0,auroc=None,balanced_accuracy=None,brier=None,log_loss=None)
    if not set(np.unique(y)).issubset({0,1}):raise ValueError('Nonbinary/undefined metric label')
    both=len(np.unique(y))==2
    return dict(n=len(y),positive=int(y.sum()),
        auroc=float(roc_auc_score(y,score)) if both else None,
        balanced_accuracy=float(balanced_accuracy_score(y,score>=threshold)) if both else None,
        brier=float(brier_score_loss(y,score)),log_loss=float(log_loss(y,score,labels=[0,1])))


def cluster_interval(y, score, groups, statistic='auroc', repetitions=2000, seed=711):
    y=np.asarray(y);score=np.asarray(score);groups=np.asarray(groups)
    unique=np.unique(groups);rng=np.random.default_rng(seed);values=[]
    if len(unique)<2:return dict(interval=None,groups=len(unique),valid_resamples=0,reason='Fewer than two independent test groups')
    for _ in range(repetitions):
        selected=rng.choice(unique,len(unique),replace=True)
        ids=np.concatenate([np.flatnonzero(groups==group) for group in selected])
        value=metrics(y[ids],score[ids])[statistic]
        if value is not None:values.append(value)
    return dict(interval=np.quantile(values,[.025,.975]).tolist() if values else None,
        groups=len(unique),valid_resamples=len(values),repetitions=repetitions,
        caution='Exploratory cluster interval; few structural groups imply limited precision' if len(unique)<10 else None)


def candidates(seed=713):
    # Identical tuning count for every representation: 4 linear, 4 nonlinear.
    for c in (.01,.1,1.,10.):
        yield 'linear_C'+str(c),LogisticRegression(C=c,max_iter=1000,random_state=seed)
    for alpha in (.001,.01,.1,1.):
        yield 'mlp32_alpha'+str(alpha),MLPClassifier(hidden_layer_sizes=(32,),alpha=alpha,
            solver='lbfgs',max_iter=500,random_state=seed)


def fit_selected(train_x, train_y, val_x, val_y, dimension=32):
    if len(np.unique(train_y))<2 or len(np.unique(val_y))<2:
        raise ValueError('Both labels are required in train and validation')
    dim=min(dimension,len(train_x)-1,train_x.shape[1])
    if dim<1:raise ValueError('Insufficient training examples')
    records=[];best=None
    for name,head in candidates():
        model=make_pipeline(StandardScaler(),PCA(n_components=dim,svd_solver='full'),head)
        model.fit(train_x,train_y);score=model.predict_proba(val_x)[:,1]
        loss=float(log_loss(val_y,score,labels=[0,1]));records.append(dict(name=name,validation_log_loss=loss))
        if best is None or loss<best[0]:best=(loss,name,model)
    return best[2],dict(selected=best[1],selection='minimum validation log loss; deterministic listed-order ties',
        common_pca_dimension=dim,trials=records)


def failure_target(row):
    # Full T failure is distinct from a short-horizon collision endpoint.
    if row['outcome']=='safe_completion':return 0
    if row['outcome'] in ('collision','safe_timeout','shield_blocked'):return 1
    return None


def evaluate(rows, features, heldout_family):
    check_groups(rows)
    # Selection/evaluation filtering never silently changes group assignment.
    train=[i for i,r in enumerate(rows) if r['split']=='train' and r['family']!=heldout_family and r['label']!='unknown']
    val=[i for i,r in enumerate(rows) if r['split']=='validation' and r['family']!=heldout_family and r['label']!='unknown']
    test=[i for i,r in enumerate(rows) if r['split']=='test' and r['family']==heldout_family and r['label']!='unknown']
    if not train or not val or not test:raise ValueError('Missing preregistered family-holdout cells')
    y=np.asarray([int(r['label']=='infeasible') for r in rows]);results={};scores={}
    for key,x in sorted(features.items()):
        x=np.asarray(x,dtype=float)
        if len(x)!=len(rows) or not np.isfinite(x).all():raise ValueError('Feature coverage/nonfinite values')
        model,selection=fit_selected(x[train],y[train],x[val],y[val])
        prediction=model.predict_proba(x[test])[:,1];scores[key]=prediction
        failed=np.asarray([failure_target(rows[i])==1 for i in test])
        groups=[rows[i]['scene_group'] for i in test]
        results[key]=dict(selection=selection,test=metrics(y[test],prediction),
            failed_subset=metrics(y[test][failed],prediction[failed]),
            test_auroc_interval=cluster_interval(y[test],prediction,groups),
            failed_subset_auroc_interval=cluster_interval(y[test][failed],prediction[failed],np.asarray(groups)[failed]),
            decision_time='initial state',estimand='RA-SEP-1 infeasibility versus witnessed feasibility')
    return dict(results=results,unknown_counts={split:sum(r['split']==split and r['label']=='unknown' for r in rows)
        for split in ('train','validation','test')},known_indices=dict(train=train,validation=val,test=test),
        heldout_family=heldout_family,claim='External supervised readout; no native understanding or information-destruction claim'),scores


def main():
    parser=argparse.ArgumentParser();parser.add_argument('manifest',type=Path);parser.add_argument('features',type=Path)
    parser.add_argument('output',type=Path);parser.add_argument('--heldout-family',required=True);args=parser.parse_args()
    rows=json.loads(args.manifest.read_text())['rows']
    with np.load(args.features) as arrays:features={k:arrays[k] for k in arrays.files}
    results,scores=evaluate(rows,features,args.heldout_family);args.output.mkdir()
    (args.output/'READOUT.json').write_text(json.dumps(results,indent=2)+'\n')
    np.savez_compressed(args.output/'test_predictions.npz',**scores)


if __name__=='__main__':main()
