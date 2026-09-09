import copy
import numpy as np
import pytest
from crashbench.detour_benefit import (Opportunity,PendingAction,terminal,readout,validate_panel,freeze,ridge,predict)

def panel():
    return [{'episode_id':f'e{i:02d}','source':str(i//3),'role':'fitting' if i<36 else 'calibration','condition':('glass','offpath','noglass')[i%3],'triggered':False,'horizons':{str(h):readout([],h) for h in (220,440)}} for i in range(48)]

def test_no_trigger_and_rejection_consumes_only_opportunity():
    gate=Opportunity();assert not gate.observe(.1,.3,1);assert gate.observe(.5,.3,2)
    assert not gate.observe(.9,.3,3) # Rejection cannot lead to a second opportunity.
    assert not Opportunity().observe(1,.3,220)

def test_early_success_accident_and_simultaneous_priority():
    assert terminal(True,True)=='accident'
    for success,accident in [(True,False),(False,True),(True,True)]:
        e=[{'step':7,'reason':terminal(success,accident),'calls':1}]
        r=readout(e,220);assert r['success']==int(success and not accident);assert r['accident']==int(accident)

def test_h_readout_does_not_interrupt_continuous_trajectory():
    events=[{'step':i,'reason':None,'calls':1} for i in range(1,221)]
    original=copy.deepcopy(events);assert readout(events,220)['success']==0;assert events==original
    events.extend([{'step':i,'reason':'success' if i==260 else None,'calls':1} for i in range(221,261)])
    assert readout(events,220)['success']==0 and readout(events,440)['success_step']==260

def test_controller_steps_and_failed_early_stop_not_acceleration():
    events=[{'step':i,'reason':'accident' if i==5 else None,'calls':0,'controller_steps':1} for i in range(1,6)]
    r=readout(events,440);assert r['completion_cost']==440 and r['controller_steps']==5

def test_pending_restore_and_switch():
    action=np.arange(7);b=PendingAction(action);r=PendingAction(action)
    assert np.array_equal(b.reject_intervention(),action);assert b.take() is None
    r.accept_intervention();assert r.take() is None;assert np.array_equal(action,np.arange(7))

def test_source_conditions_never_leak():
    p=panel();validate_panel(p);p[0]['role']='calibration'
    with pytest.raises(ValueError):validate_panel(p)

def test_freeze_all_base_and_never_accepts_test_labels():
    p=panel();config={'model':{'pca_components':4}}
    f=freeze(p,[],config);assert not any(f['choices']['BenefitGate'].values())
    with pytest.raises(ValueError):freeze(p,[{'episode_id':'e00','repeat':2,'option':0}],config)

def test_ridge_difference_equals_direct_two_arm_q():
    rng=np.random.default_rng(9);x=rng.normal(size=(20,5));y=rng.normal(size=(20,2));s=np.repeat(['a','b','c','d'],5)
    q=predict(ridge(x,y,s),x);delta=predict(ridge(x,y[:,1:]-y[:,:1],s),x)[:,0]
    np.testing.assert_allclose(q[:,1]-q[:,0],delta,atol=1e-12)

def test_calibration_constraints_and_phase_freeze():
    p=panel();rng=np.random.default_rng(10);a=[]
    for e in p:
        e.update(triggered=True,risk=.6,hidden=rng.normal(size=12).tolist(),robot_state=[0]*8,nominal_action=[0]*7)
        for repeat in (0,1):
            for option in (0,1):
                # R harms controls and succeeds in glass; calibration cannot accept controls.
                success=option if e['condition']=='glass' else 1-option
                a.append({'episode_id':e['episode_id'],'repeat':repeat,'option':option,'horizons':{str(h):dict(success=success,accident=0,calls=1,controller_steps=option,completion_cost=10 if success else h) for h in (220,440)}})
    f=freeze(p,a,{'model':{'pca_components':4}})
    assert f['info']['direct_q_max_error']<1e-8
    assert not any(f['choices']['RiskDetour'].values())
    assert f['choices']['DirectQ_sanity']==f['choices']['BenefitGate']

def test_full_no_trigger_analysis_preserves_all_denominators(tmp_path):
    import hashlib,json,pickle
    from scripts.expansion.analyze_detour_benefit import analyze
    p=panel()
    for e in p:
        e.update(events=[],prefix_calls=0,prefix_elapsed_seconds=0,risk_call_count=0,risk_elapsed_seconds=0)
        folder=tmp_path/e['episode_id'];folder.mkdir();trace=folder/'prefix.pkl.gz';trace.write_bytes(b'no trigger fixture')
        e['prefix_sha256']=hashlib.sha256(trace.read_bytes()).hexdigest()
    for name,v in [('anchors.json',p),('A.json',[]),('B.json',[]),('complete.json',{'status':'COMPLETE'}),('provenance.json',{})]:
        (tmp_path/name).write_text(json.dumps(v))
    (tmp_path/'models.pkl').write_bytes(pickle.dumps({}))
    f=freeze(p,[],{'model':{'pca_components':4}});f.pop('models')
    for name,key in [('anchors.json','anchors_sha256'),('A.json','A_sha256'),('models.pkl','models_sha256')]:f[key]=hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()
    (tmp_path/'freeze.json').write_text(json.dumps(f));(tmp_path/'freeze.sha256').write_text(hashlib.sha256((tmp_path/'freeze.json').read_bytes()).hexdigest())
    analyze(tmp_path,bootstrap=10)
    metrics=json.loads((tmp_path/'analysis/metrics.json').read_text())
    assert metrics['cost']['prefixes']==48 and metrics['cost']['scored_branches']==0
    assert (tmp_path/'analysis/success_cumulative_curve.csv').read_text().count('\n')==442
