"""Exploratory train/validation camera ablations after a failed dataset screen."""
import os
from pathlib import Path
import json
import numpy as np
from .inventory import read,write,sha
from .constraint_shortcuts import measured


def main(root):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Shortcut fitting belongs on a compute node')
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    parent=read(root/'DIAGNOSTIC_SOURCE.json');source=Path(parent['root'])
    for name,digest in parent['files'].items():
        if sha(source/name)!=digest:raise RuntimeError('Diagnostic source changed')
    rows=[json.loads(x) for x in (source/'PROPOSED_DATASET.jsonl').read_text().splitlines()]
    with np.load(source/'SHORTCUT_FEATURES.npz',allow_pickle=False) as archive:
        if archive['sample_ids'].tolist()!=[r['sample_id'] for r in rows]:raise RuntimeError('Feature/sample order differs')
        joint=archive['joint'].copy()
    if joint.shape!=(len(rows),261):raise RuntimeError('Unexpected frozen diagnostic features')
    y=np.asarray([int(r['evidence']['feasibility']=='feasible') for r in rows])
    train=np.asarray([r['split']=='train' for r in rows]);val=np.asarray([r['split']=='validation' for r in rows])
    groups=np.asarray([r['metadata']['ancestry_component'] for r in rows])
    if set(groups[train])&set(groups[~train]) or set(groups[val])&set(groups[~(train|val)]):raise RuntimeError('Source leakage')
    sets={
        'view_1_only':list(range(55))+list(range(110,120)),
        'view_2_only':list(range(55,110))+list(range(120,130)),
        'two_view_color_statistics_only':list(range(110,130)),
        'two_view_color_area_only':[110,115,120,125],
        'two_view_color_centroids_only':[111,112,116,117,121,122,126,127],
    }
    results={}
    for name,columns in sets.items():
        indices=columns+[130+i for i in columns]+[260];x=joint[:,indices]
        for family,model in (
            ('linear',make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000,random_state=1014))),
            ('forest',RandomForestClassifier(n_estimators=200,max_depth=4,min_samples_leaf=8,random_state=1014,n_jobs=2))):
            model.fit(x[train],y[train]);scores=model.predict_proba(x[val])[:,list(model.classes_).index(1)]
            results[name+'_'+family]=dict(features=len(indices),**measured(y[val],scores,groups[val]))
    report=dict(execution_passed=True,exploratory=True,
        purpose='Diagnose an already failed shortcut screen; these ablations cannot authorize training or change the fixed acceptance limit',
        source=str(source),source_shortcut_passed=read(source/'SHORTCUT_AUDIT.json')['passed'],
        input_views=[im['camera'] for im in rows[0]['student']['images']],results=results,
        train_rows=int(train.sum()),validation_rows=int(val.sum()),test_predictions=0,
        hyperparameter_search=False,scientific_training_allowed=False,api_calls=0,slurm_job=os.environ['SLURM_JOB_ID'])
    write(root/'DIAGNOSTIC_COMPLETE.json',report)
    write(root/'HASHES.json',{name:sha(root/name) for name in ('DIAGNOSTIC_COMPLETE.json','DIAGNOSTIC_SOURCE.json','STATISTICS_READY.json')})
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
