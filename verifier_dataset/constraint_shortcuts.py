"""Frozen low-dimensional diagnostics; the final test split stays unscored."""
from collections import Counter
import json
import re
from .inventory import write
from .pool_audit import image_statistics


def features(student):
    import numpy as np
    from PIL import Image
    images=[];color=[]
    for spec in student['images']:
        with Image.open(spec['path']) as im:images.append(np.asarray(im).copy())
    for rgb in images:
        im=rgb.astype(float)/255.;r,g,b=np.moveaxis(im,-1,0);h,w=r.shape
        y,x=np.mgrid[0:h,0:w];x=x/w;y=y/h
        masks=[(r>.6)&(g>.18)&(g<.75)&(b<.35),(r>.45)&(b>.45)&(g<.45)]
        for mask in masks:
            color.extend([float(mask.mean())]+([float(v[mask].mean()) for v in (x,y)]+[float(v[mask].std()) for v in (x,y)] if mask.any() else [0.]*4))
    rgb=np.asarray(image_statistics(images));color=np.asarray(color)
    match=re.findall(r'Only the (orange|purple) boundary is protected',student['constraints'])
    if len(match)!=1:raise RuntimeError('Ambiguous rule color')
    sign=1. if match[0]=='orange' else -1.
    static=np.concatenate((rgb,color))
    joint=np.concatenate((static,sign*static,[sign]))
    return rgb,joint


def measured(y,score,groups):
    import numpy as np
    from sklearn.metrics import roc_auc_score,balanced_accuracy_score,brier_score_loss,log_loss
    result=dict(n=len(y),source_groups=len(set(groups)),auroc=float(roc_auc_score(y,score)),
        balanced_accuracy_at_05=float(balanced_accuracy_score(y,score>=.5)),
        brier=float(brier_score_loss(y,score)),log_loss=float(log_loss(y,score,labels=[0,1])))
    rng=np.random.default_rng(1014);unique=sorted(set(groups));values=[]
    for _ in range(1000):
        drawn=rng.choice(unique,len(unique),replace=True)
        idx=np.concatenate([np.flatnonzero(groups==s) for s in drawn])
        values.append(roc_auc_score(y[idx],score[idx]))
    result['source_bootstrap_auroc_95pct']=np.quantile(values,[.025,.975]).tolist()
    return result


def color_interaction_features(joint,kind):
    """The same color-only columns used by the pre-existing failed-screen diagnostic."""
    if joint.ndim!=2 or joint.shape[1]!=261:raise RuntimeError('Unexpected fixed joint features')
    columns={'color_statistics':list(range(110,130)),
             'color_centroids':[111,112,116,117,121,122,126,127]}
    if kind not in columns:raise RuntimeError('Unknown predeclared color-only readout')
    base=columns[kind]
    return joint[:,base+[130+i for i in base]+[260]]


def main(root,rows):
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    y=np.asarray([int(r['evidence']['feasibility']=='feasible') for r in rows]);all_features=[features(r['student']) for r in rows]
    rgb=np.asarray([r[0] for r in all_features]);joint=np.asarray([r[1] for r in all_features])
    text=[r['student']['task']+'\n'+r['student']['constraints'] for r in rows]
    groups=np.asarray([r['metadata']['ancestry_component'] for r in rows])
    train=np.asarray([r['split']=='train' for r in rows]);val=np.asarray([r['split']=='validation' for r in rows])
    test=np.asarray([r['split']=='test' for r in rows])
    if set(groups[train])&set(groups[val]) or set(groups[train|val])&set(groups[test]):
        raise RuntimeError('Shortcut readout source leakage')
    if len(set(groups[val]))<5 or len(set(groups[train]))<15:raise RuntimeError('Insufficient independent split coverage')
    models=[('rgb_red_statistics',make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000,random_state=1014)),rgb),
        ('task_rules_only',make_pipeline(TfidfVectorizer(ngram_range=(1,2)),LogisticRegression(C=1.,max_iter=1000,random_state=1014)),np.asarray(text)),
        ('rgb_color_rule_interactions_linear',make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000,random_state=1014)),joint),
        ('rgb_color_rule_interactions_forest',RandomForestClassifier(n_estimators=200,max_depth=4,min_samples_leaf=8,random_state=1014,n_jobs=2),joint)]
    protocol=json.loads((__import__('pathlib').Path(__file__).with_name('constraint_protocol.json')).read_text())
    limits=protocol['shortcut_screen']
    for kind in limits.get('additional_color_only_interaction_readouts',[]):
        x=color_interaction_features(joint,kind)
        models.extend([(kind+'_rule_interactions_linear',make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000,random_state=1014)),x),
            (kind+'_rule_interactions_forest',RandomForestClassifier(n_estimators=200,max_depth=4,min_samples_leaf=8,random_state=1014,n_jobs=2),x)])
    results={};predictions={}
    for name,model,x in models:
        model.fit(x[train],y[train]);p=model.predict_proba(x[val])[:,list(model.classes_).index(1)]
        results[name]=measured(y[val],p,groups[val]);predictions[name]=p.tolist()
    image_ok=abs(results['rgb_red_statistics']['auroc']-.5)<1e-8
    text_ok=abs(results['task_rules_only']['auroc']-.5)<1e-8
    joint_ok=max(results[k]['auroc'] for k in results if 'interactions' in k)<=limits['maximum_validation_joint_statistics_auroc']
    result=dict(stage='validation_only_shortcut_screen',results=results,
        gates=dict(image_only_exact_chance=image_ok,text_only_exact_chance=text_ok,joint_statistics_below_limit=joint_ok),
        passed=image_ok and text_ok and joint_ok,configuration=limits,
        train_rows=int(train.sum()),validation_rows=int(val.sum()),locked_test_rows=int(test.sum()),
        locked_test_scored=False,threshold_search=False,hyperparameter_search=False,
        note='Passing excludes these specified shortcuts only. It does not establish that generative reasoning is necessary.',
        scientific_training_allowed=False)
    write(root/'SHORTCUT_AUDIT.json',result)
    write(root/'SHORTCUT_VALIDATION_PREDICTIONS.json',dict(sample_ids=[r['sample_id'] for r in rows if r['split']=='validation'],predictions=predictions))
    np.savez_compressed(root/'SHORTCUT_FEATURES.npz',rgb=rgb,joint=joint,sample_ids=np.asarray([r['sample_id'] for r in rows]))
    return result
