import copy
import numpy as np
import pytest
from scripts.expansion.direct_cost_learning import features,fit_fold,predict,choose,targets,rule


def context():
    return {'timing':{'age_steps':1},'chunk_remaining':0,'nominal_proposal':None,
            'delivered':{'state':np.arange(8),'full_image':np.zeros((256,256,3),dtype=np.uint8),
                         'wrist_image':np.ones((256,256,3),dtype=np.uint8)}}


def test_allowlist_and_nonexistent_proposal():
    a={'age_steps':1,'anchor_steps':15,'task_id':'libero_spatial:0'};c=context()
    before=features(a,c,'observation')
    c['fresh']=object();c['policy_continuation']=object();c['future_success']=1
    assert np.array_equal(before,features(a,c,'observation'))
    assert len(before)==108 and len(features(a,c,'metadata'))==3
    c['nominal_proposal']=np.zeros(7)
    with pytest.raises(ValueError):features(a,c,'observation')


def test_heldout_all_conditions_do_not_influence_fit_or_scaler():
    x=np.arange(24,dtype=float).reshape(8,3);y=np.arange(32,dtype=float).reshape(8,4)
    sources=['a','a','b','b','c','c','d','d']
    first=fit_fold(x,y,sources,'a','metadata')
    x2=x.copy();y2=y.copy();x2[:2]=99999;y2[:2]=-99999
    assert first==fit_fold(x2,y2,sources,'a','metadata')
    assert 'a' not in first['training_sources'] and first['training_indices']==list(range(2,8))
    assert np.isfinite(predict(first,x[0])).all()


def test_constant_features_intercept_and_source_weights():
    x=np.zeros((5,3)); y=np.zeros((5,4));y[1:4]=[200,40,1,0]
    # train b: three identical anchors, train c: one. Source-balanced mean is .5.
    model=fit_fold(x,y,['a','b','b','b','c'],'a','metadata')
    assert np.allclose(predict(model,x[0]),[100,20,.5,0])


def test_cost_target_does_not_reward_early_accident_and_gate_requires_both():
    r={'horizons':{'200':{'steps':10,'task_success':0,'catastrophe':1,'inference_calls':2}}}
    assert targets(r).tolist()==[200,2,0,1]
    assert choose([-5,-1,0,0])==1
    for p in ([-5,0,0,0],[-5,-1,-.1,0],[-5,-1,0,.1],[0,-1,0,0]):assert choose(p)==0
    assert choose([-5,-1,-.1,.1],False)==1
    assert rule('age_equal1',{'age_steps':0})==0
