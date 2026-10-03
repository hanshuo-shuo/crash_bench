"""Preregistered scene-group transfer and same-time incremental evaluation."""
import argparse,json,warnings
from pathlib import Path
import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import readout as ro
P=json.loads((Path(__file__).parent/'matrix_protocol.json').read_text())


def save(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def validate_outcome(summary):
    outcome=summary['outcome'];steps=summary['steps'];T=summary['execution_horizon_T']
    if outcome not in ('collision','safe_completion','safe_timeout','invalid','not_run'):
        raise ValueError('Undefined policy outcome domain')
    if T!=P['execution_horizon_T'] or not 0<=steps<=T:raise ValueError('Execution horizon mismatch')
    if outcome=='safe_timeout' and (steps!=T or not summary['safe_history'] or summary['safe_success']):
        raise ValueError('Safe timeout requires the complete T-step endpoint')
    if outcome=='collision' and summary['safe_history']:raise ValueError('Collision without safety violation')
    if outcome=='safe_completion' and not (summary['safe_history'] and summary['safe_success']):
        raise ValueError('Safe completion endpoint mismatch')
    if outcome=='not_run' and summary['initial_valid'] and summary['initial_label']!='unknown':
        raise ValueError('Known valid state missing policy outcome')
    return ro.failure_target(summary)


def fit_fixed(x,y,head):
    if not len(y):raise ValueError('No defined fitting outcomes')
    if not set(np.unique(y)).issubset({0,1}):raise ValueError('Nonbinary/undefined fitting label')
    if len(np.unique(y))<2:
        model=DummyClassifier(strategy='prior');model.fit(x,y);return model
    dim=min(P['readouts']['pca_max_dimension'],len(x)-1,x.shape[1])
    model=make_pipeline(StandardScaler(),PCA(n_components=max(1,dim),svd_solver='full'),clone(head))
    model.fit(x,y);return model


def probability(model,x):
    classes=list(model.classes_)
    return model.predict_proba(x)[:,classes.index(1)] if 1 in classes else np.zeros(len(x))


def nested_feasibility_select(features,candidates,fitting,validation,y):
    records=[];best=None
    for key in candidates:
        x=features[key];model,selection=ro.fit_selected(x[fitting],y[fitting],x[validation],y[validation])
        loss=min(t['validation_log_loss'] for t in selection['trials'])
        records.append(dict(representation=key,head=selection['selected'],validation_log_loss=loss))
        if best is None or loss<best[0]:best=(loss,key,model,selection)
    return best,records


def paired_interval(y,first,second,groups,repetitions=2000):
    rng=np.random.default_rng(713);groups=np.asarray(groups);unique=np.unique(groups);values=[]
    for _ in range(repetitions):
        selected=rng.choice(unique,len(unique),replace=True)
        ids=np.concatenate([np.flatnonzero(groups==g) for g in selected])
        a=ro.metrics(y[ids],first[ids])['auroc'];b=ro.metrics(y[ids],second[ids])['auroc']
        if a is not None and b is not None:values.append(b-a)
    return dict(delta_auroc_interval=np.quantile(values,[.025,.975]).tolist() if len(unique)>=2 and values else None,
        valid_resamples=len(values),groups=len(unique),repetitions=repetitions,
        caution='Four predefined test groups; exploratory interval, not benchmark/population precision')


def load(root):
    rows=[];features={}
    for layout in P['layouts']:
        for family in P['mechanisms']:
            for variant in P['variants']:
                case=layout['id']+'_'+family+'_'+variant
                found=list(root.glob('**/output/'+case+'/summary.json'))+list(root.glob('**/shard_*/'+case+'/summary.json'))
                # Input root contains only GPU results; reject duplicate replicas.
                if len(found)!=1:raise ValueError(f'Expected one GPU result for {case}, got {found}')
                folder=found[0].parent;summary=json.loads(found[0].read_text());validate_outcome(summary)
                if summary['outcome'] in ('collision','safe_completion','safe_timeout') and not json.loads((folder/'VERIFIED.json').read_text())['passed']:raise ValueError('Missing verified policy endpoint')
                feature_audit=json.loads((folder/'FEATURE_AUDIT.json').read_text())
                if not feature_audit['physics_unchanged']:raise RuntimeError('Changed feature state')
                label=json.loads((folder/'INHERITED.json').read_text())['label']
                row=dict(scene_group=layout['id'],split=layout['split'],family=family,variant=variant,case=case,label=label,
                    outcome=summary['outcome'],steps=summary['steps'],initial_valid=summary['initial_valid'],source=str(folder))
                row.update({k:summary[k] for k in ('execution_horizon_T','safe_history','safe_success','initial_label')})
                row['failure_target']=ro.failure_target(row)
                rows.append(row);values={}
                with np.load(folder/'layer_features.npz') as a:
                    for key in ('own_vision_tower','projected_vision','native_final'):values[key]=a[key].astype(float)
                    for key in ('layers','image_layers'):
                        for i,value in enumerate(a[key]):values[key+'_%02d'%(i+1)]=value.astype(float)
                with np.load(folder/'general_vision.npz') as a:
                    for key in a.files:values[key]=a[key].astype(float)
                with np.load(folder/'cue_baselines.npz') as a:values['rgb_pooled_red']=a['rgb_pooled_and_red']
                with np.load(folder/'attention_cues.npz') as a:
                    for key in ('visibility','geometry','knows_fixed'):values[key]=a[key]
                with np.load(folder/'actual_policy_input.npz') as a:values['proprioception']=a['observation/state']
                for key,value in values.items():features.setdefault(key,[]).append(value)
                row['layer_accepted']=feature_audit['layer_accepted'];row['attention_accepted']=feature_audit['attention_accepted']
    return rows,{k:np.asarray(v,dtype=float) for k,v in features.items()}


def analyze(rows,features,output):
    output.mkdir();ro.check_groups(rows)
    for row in rows:validate_outcome(row)
    save(output/'MANIFEST.json',dict(rows=rows,protocol=P))
    np.savez_compressed(output/'FEATURES.npz',**features)
    indices={split:np.asarray([i for i,r in enumerate(rows) if r['split']==split and
        r['family']==('cage' if split=='test' else 'slit') and r['label']!='unknown' and r['initial_valid']])
        for split in ('train','validation','test')}
    tr,va,te=[indices[s] for s in ('train','validation','test')]
    y=np.asarray([int(r['label']=='infeasible') for r in rows]);group=np.asarray([r['scene_group'] for r in rows])
    failure=np.asarray([ro.failure_target(r) if ro.failure_target(r) is not None else -1 for r in rows])
    models={};selection={};predictions={};results={};curves={}
    for key,x in sorted(features.items()):
        if (key.startswith(('layers_','image_layers_')) and not all(r['layer_accepted'] for r in rows)) or (key=='knows_fixed' and not all(r['attention_accepted'] for r in rows)):
            results[key]=dict(unavailable='Native feature audit unresolved');continue
        if not np.isfinite(x).all():raise ValueError('Nonfinite '+key)
        model,sel=ro.fit_selected(x[tr],y[tr],x[va],y[va]);models[key]=model;selection[key]=sel
        score=probability(model,x[te]);predictions[key]=score
        failed=failure[te]==1
        # Fixed .5 threshold; layer/head tuning uses validation loss only.
        results[key]=dict(selection=sel,test=ro.metrics(y[te],score),
            failed_subset=ro.metrics(y[te][failed],score[failed]),
            auroc_interval=ro.cluster_interval(y[te],score,group[te],seed=713),
            failed_subset_interval=ro.cluster_interval(y[te][failed],score[failed],group[te][failed],seed=713),
            groups={str(g):ro.metrics(y[te][group[te]==g],score[group[te]==g]) for g in np.unique(group[te])})
        # Report each head family separately without selecting against test data.
        head_results={}
        for kind in ('linear','mlp'):
            allowed=[q for q in sel['trials'] if q['name'].startswith(kind)]
            chosen=min(allowed,key=lambda q:q['validation_log_loss'])
            head=dict(ro.candidates())[chosen['name']];fitted=fit_fixed(x[tr],y[tr],head)
            head_results[kind]=dict(selected=chosen,prediction=probability(fitted,x[te]).tolist(),test=ro.metrics(y[te],probability(fitted,x[te])))
        results[key]['head_families']=head_results
        # Fixed training prefixes; identical group counts for every representation.
        train_order=[g for g in P['group_permutation'] if any(r['scene_group']==g and r['split']=='train' for r in rows)]
        points=[]
        for count in P['readouts']['learning_curve_group_counts']:
            ids=tr[np.isin(group[tr],train_order[:count])]
            if len(np.unique(y[ids]))<2:points.append(dict(groups=count,reason='single training label'));continue
            cm,cs=ro.fit_selected(x[ids],y[ids],x[va],y[va])
            points.append(dict(groups=count,n=len(ids),selection=cs['selected'],test=ro.metrics(y[te],probability(cm,x[te]))))
        curves[key]=points
    candidates=[k for k in models if k.startswith(('layers_','image_layers_')) or k=='native_final']
    chosen=min(candidates,key=lambda k:min(t['validation_log_loss'] for t in selection[k]['trials']))
    save(output/'READOUT.json',dict(results=results,primary_vla_representation=chosen,indices={k:v.tolist() for k,v in indices.items()},
        unknown_counts={s:sum(r['split']==s and r['label']=='unknown' for r in rows) for s in indices},
        claim=P['claim'],decision_time='initial state',unknowns_excluded_only_from_binary_metrics=True))
    save(output/'LEARNING_CURVES.json',curves);np.savez_compressed(output/'TEST_PREDICTIONS.npz',indices=te,y=y[te],**predictions)
    save(output/'PAIRED_REPRESENTATION_COMPARISONS.json',{key:paired_interval(y[te],predictions[key],predictions[chosen],group[te]) for key in ('own_vision_tower','dino_patch','rgb_pooled_red','visibility','geometry','knows_fixed') if key in predictions})
    # One fixed SAFE-style architecture predicts full-T policy failure.
    safe_head=MLPClassifier(hidden_layer_sizes=(32,),alpha=.1,solver='lbfgs',max_iter=500,random_state=713)
    safe_x=features['native_final']
    defined={s:ids[failure[ids]>=0] for s,ids in indices.items()}
    exclusions={s:[rows[i]['case'] for i in ids if failure[i]<0] for s,ids in indices.items()}
    oof=np.full((len(rows),2),np.nan);folds=[]
    for g in np.unique(group[tr]):
        fitting=tr[group[tr]!=g];held=tr[group[tr]==g]
        safe_fitting=fitting[failure[fitting]>=0]
        sf=fit_fixed(safe_x[safe_fitting],failure[safe_fitting],safe_head)
        # Nested representation AND head selection: held-group labels never
        # influence the model, transforms, or selection producing its OOF score.
        best,nested=nested_feasibility_select(features,candidates,fitting,va,y)
        _,fold_key,ff,fold_selection=best
        oof[held,0]=probability(sf,safe_x[held]);oof[held,1]=probability(ff,features[fold_key][held])
        folds.append(dict(held_group=str(g),training_groups=np.unique(group[fitting]).tolist(),
            validation_groups=np.unique(group[va]).tolist(),failure_classes=np.unique(failure[safe_fitting]).tolist(),
            feasibility_classes=np.unique(y[fitting]).tolist(),nested_selected_representation=fold_key,
            nested_selected_head=fold_selection['selected'],nested_selection_trials=nested))
    safe=fit_fixed(safe_x[defined['train']],failure[defined['train']],safe_head)
    safe_test=probability(safe,safe_x[te]);safe_val=probability(safe,safe_x[va]);feas_test=predictions[chosen]
    failed_train=tr[failure[tr]==1];failed_test=failure[te]==1
    incremental=dict(primary_feasibility_representation=chosen,decision_time='initial state',folds=folds,
        safe_failure_probe=ro.metrics(failure[te][failure[te]>=0],safe_test[failure[te]>=0]),
        safe_validation_failure=ro.metrics(failure[va][failure[va]>=0],safe_val[failure[va]>=0]),
        undefined_outcome_exclusions=exclusions,undefined_outcome_exclusion_counts={s:len(v) for s,v in exclusions.items()},
        training_failure_classes=np.unique(failure[defined['train']]).tolist(),safe_description=P['failure_probe'],
        raw_safe_score_for_infeasibility=ro.metrics(y[te][failed_test],safe_test[failed_test]),
        failed_train_n=len(failed_train),failed_test_n=int(failed_test.sum()),estimand='Independently infeasible versus witnessed feasible among fixed-policy failures')
    if len(np.unique(y[failed_train]))<2 or len(np.unique(y[te][failed_test]))<2:
        incremental['not_estimable']='Both feasibility labels are required among train and test policy failures; scenes and thresholds were not retuned'
    else:
        base=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=713))
        plus=clone(base);base.fit(oof[failed_train,:1],y[failed_train]);plus.fit(oof[failed_train],y[failed_train])
        test_x=np.c_[safe_test,feas_test]
        first=probability(base,test_x[:,:1]);second=probability(plus,test_x)
        incremental.update(safe_only=ro.metrics(y[te][failed_test],first[failed_test]),
            safe_plus_feasibility=ro.metrics(y[te][failed_test],second[failed_test]),
            paired_group_bootstrap=paired_interval(y[te][failed_test],first[failed_test],second[failed_test],group[te][failed_test]),
            raw_groups={str(g):dict(safe_only=ro.metrics(y[te][(group[te]==g)&failed_test],first[(group[te]==g)&failed_test]),
                safe_plus_feasibility=ro.metrics(y[te][(group[te]==g)&failed_test],second[(group[te]==g)&failed_test])) for g in np.unique(group[te])})
        np.savez_compressed(output/'INCREMENTAL_PREDICTIONS.npz',indices=te,y=y[te],failed=failed_test,safe_failure=safe_test,feasibility=feas_test,safe_only=first,safe_plus_feasibility=second)
    np.savez_compressed(output/'TRAIN_CROSSFIT.npz',indices=tr,values=oof[tr],failure=failure[tr],y=y[tr])
    save(output/'INCREMENTAL.json',incremental)
    # KNOWS raw success association remains distinct from feasibility fitting.
    knows=features['knows_fixed'][te]
    save(output/'KNOWS_INITIAL.json',dict(scope='Initial K=1 oracle localization adaptation, no trajectory smoothing or official tracker',
        failure_by_negative_target_mass=ro.metrics(failure[te][failure[te]>=0],1-np.clip(knows[failure[te]>=0,0],0,1)),
        target_mass=knows[:,0].tolist(),density=knows[:,1].tolist(),entropy=knows[:,2].tolist(),indices=te.tolist()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    rows,features=load(a.root);analyze(rows,features,a.output)
