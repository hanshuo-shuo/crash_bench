"""Pure measurement and frozen-learning contract for the bounded Detour study."""
from collections import Counter
import time
import numpy as np

HORIZONS=(220,440)
METHODS=('Base','AlwaysDetour','RiskDetour','DirectQ_sanity','BenefitGate','RiskOnlyBenefit','A_reference','A_long_reference')

def terminal(success, accident, terminated=False):
    return 'accident' if accident else 'success' if success else 'environment_termination' if terminated else None

def readout(events, horizon, calls=0, controller_steps=0):
    seen=[e for e in events if e['step']<=horizon]
    end=next((e for e in seen if e['reason']),None)
    success=bool(end and end['reason']=='success'); accident=bool(end and end['reason']=='accident')
    return {'success':int(success),'accident':int(accident),'reason':end['reason'] if end else 'timeout',
        'steps':end['step'] if end else horizon,'success_step':end['step'] if success else None,
        'completion_cost':end['step'] if success else horizon,
        'calls':sum(e.get('calls',0) for e in seen),'controller_steps':sum(e.get('controller_steps',0) for e in seen)}

class Opportunity:
    def __init__(self):self.consumed=False
    def observe(self, risk, threshold, step, last=219):
        hit=not self.consumed and step<=last and risk>threshold
        if hit:self.consumed=True
        return hit

class PendingAction:
    """An already paid-for one-action OpenVLA proposal; discard only on acceptance."""
    def __init__(self, action):self.action=np.asarray(action).copy() if action is not None else None
    def take(self):
        action=self.action;self.action=None;return action
    def reject_intervention(self):return self.take()
    def accept_intervention(self):self.action=None

def validate_panel(panel):
    roles={}
    assert len(panel)==48 and len({x['episode_id'] for x in panel})==48
    for e in panel:
        old=roles.setdefault(e['source'],e['role'])
        if old!=e['role']:raise ValueError('source leakage across conditions')
    if Counter(roles.values())!={'fitting':12,'calibration':4}:raise ValueError('source allocation changed')
    for s in roles:
        if sorted(e['condition'] for e in panel if e['source']==s)!=['glass','noglass','offpath']:raise ValueError('condition coverage')

def validate_phase(records, anchors, phase):
    repeats=(0,1) if phase=='A' else (2,3)
    expected={(e['episode_id'],r,o) for e in anchors if e['triggered'] for r in repeats for o in (0,1)}
    actual=[(r['episode_id'],r['repeat'],r['option']) for r in records]
    if len(set(actual))!=len(actual) or set(actual)!=expected:raise ValueError('phase incomplete, duplicate, or crossed A/B labels')

def episode_outcome(e, records, option, horizon):
    if not e['triggered']:return {k:float(v) for k,v in e['horizons'][str(horizon)].items() if isinstance(v,(int,float))}
    rows=[r['horizons'][str(horizon)] for r in records if r['episode_id']==e['episode_id'] and r['option']==option]
    if len(rows)!=2:raise ValueError('expected two repeats per phase')
    return {k:float(np.mean([r[k] for r in rows])) for k in ('success','accident','calls','controller_steps','completion_cost')}

def source_macro(panel, values):
    by={}
    for e,v in zip(panel,values):by.setdefault(e['source'],[]).append(v)
    return float(np.mean([np.mean(v) for v in by.values()]))

def weights(sources):
    counts=Counter(sources);w=np.array([1/counts[s] for s in sources],float)
    return w/w.mean()

def ridge(x,y,sources,alpha=1.):
    w=weights(sources);mean=np.average(x,axis=0,weights=w);scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=w))+1e-6
    z=np.column_stack(((x-mean)/scale,np.ones(len(x))));p=np.eye(z.shape[1])*alpha;p[-1,-1]=0
    coef=np.linalg.solve(z.T@(w[:,None]*z)+p,z.T@(w[:,None]*y))
    return {'mean':mean,'scale':scale,'coef':coef}

def predict(model,x):return np.column_stack(((x-model['mean'])/model['scale'],np.ones(len(x))))@model['coef']

def freeze(anchors, a, config):
    """Only A outcomes are accepted. Caller writes freeze before collecting any B."""
    validate_phase(a,anchors,'A');start=time.perf_counter()
    n=len(anchors);fit=np.array([e['role']=='fitting' and e['triggered'] for e in anchors]);cal=np.array([e['role']=='calibration' for e in anchors]);trigger=np.array([e['triggered'] for e in anchors])
    sources=np.array([e['source'] for e in anchors]);risks=np.array([e.get('risk',0) for e in anchors])[:,None]
    labels=np.zeros((n,4));arms=np.zeros((n,2,4))
    for i,e in enumerate(anchors):
        for o in (0,1):
            for hi,h in enumerate(HORIZONS):
                v=episode_outcome(e,a,o,h);arms[i,o,hi*2:hi*2+2]=v['success'],v['accident']
        labels[i]=arms[i,1]-arms[i,0]
    choices={'Base':np.zeros(n,int),'AlwaysDetour':trigger.astype(int)};info={};models={};scores={'RiskDetour':risks[:,0]}
    allowed={'RiskDetour':trigger.copy()}
    k=config['model']['pca_components']
    if fit.sum()>=k+1:
        hidden=np.array([e.get('hidden',np.zeros(4096)) for e in anchors]);mean=np.average(hidden[fit],axis=0,weights=weights(sources[fit]));centered=hidden[fit]-mean
        _,sv,vt=np.linalg.svd(centered,full_matrices=False);rank=int(np.sum(sv>sv[0]*max(centered.shape)*np.finfo(float).eps))
        info['fitting_hidden_rank']=rank
    else:rank=0
    info['fitting_anchor_count']=int(fit.sum());info['pca_components']=k
    if rank>=k:
        pcs=vt[:k].T
        x=np.column_stack(((hidden-mean)@pcs,np.array([e.get('robot_state',[0]*8) for e in anchors]),np.array([e.get('nominal_action',[0]*7) for e in anchors])))
        for name,xx in [('BenefitGate',x),('RiskOnlyBenefit',risks)]:
            model=ridge(xx[fit],labels[fit],sources[fit]);pred=predict(model,xx);models[name]=model
            scores[name]=pred[:,0];allowed[name]=trigger&(pred[:,1]<=0)&(pred[:,3]<=0)
            info[name+'_predictions']=pred.tolist()
            times=[];example_hidden=hidden[fit][0].copy();example_tail=x[fit][0,k:].copy();example_risk=risks[fit][:1].copy()
            for _ in range(100):
                tick=time.perf_counter()
                if name=='BenefitGate':
                    one=np.concatenate(((example_hidden-mean)@pcs,example_tail))[None,:]
                else:one=example_risk
                predict(model,one)
                times.append((time.perf_counter()-tick)*1000)
            info[name+'_gate_latency_ms']={'median':float(np.median(times)),'p95':float(np.percentile(times,95)),'scope':'warm CPU projection/scaler/ridge, one candidate; no VLA'}
        qm=ridge(x[fit],arms[fit].reshape((-1,8)),sources[fit]);qpred=predict(qm,x).reshape((-1,2,4));err=float(np.max(np.abs(qpred[:,1]-qpred[:,0]-np.array(info['BenefitGate_predictions']))))
        if err>1e-8:raise ValueError('DirectQ algebra mismatch')
        info['direct_q_max_error']=err;models.update(pca_mean=mean,pca_components=pcs)
    else:
        info['support_failure']='insufficient fitting count/rank; both learned gates all Base'
        for name in ('BenefitGate','RiskOnlyBenefit'):scores[name]=np.zeros(n);allowed[name]=np.zeros(n,bool)
    for name,score in scores.items():
        # Calibrate only on four sources; no B outcomes or fitting outcomes here.
        thresholds=[float('inf')]+sorted(set(float(s) for s in score[cal&trigger]))
        if (cal&trigger).any():thresholds.append(float(np.min(score[cal&trigger])-1e-8))
        best=None
        for threshold in thresholds:
            take=(score>threshold)&allowed[name];selected=np.where(take[:,None],labels,0.)
            normal=cal&np.array([e['condition']!='glass' for e in anchors])
            ok=all(source_macro([e for e,c in zip(anchors,cal) if c],selected[cal,j])<=1e-12 for j in (1,3))
            ok &= not np.any(take[normal,None] & (labels[normal][:,[0,2]]<0))
            if not ok:continue
            value=source_macro([e for e,c in zip(anchors,cal) if c],selected[cal,0]);acc=source_macro([e for e,c in zip(anchors,cal) if c],selected[cal,1]);rate=source_macro([e for e,c in zip(anchors,cal) if c],take[cal])
            key=(value,-acc,-rate,threshold)
            if best is None or key>best[0]:best=(key,take,threshold)
        assert best is not None
        choices[name]=best[1].astype(int);info[name+'_threshold']=None if np.isinf(best[2]) else best[2]
    choices['DirectQ_sanity']=choices['BenefitGate'].copy()
    choices['A_reference']=(trigger&(labels[:,0]>0)).astype(int)
    choices['A_long_reference']=(trigger&(labels[:,2]>0)).astype(int)
    info['fit_elapsed_seconds']=time.perf_counter()-start
    return {'choices':{name:{e['episode_id']:int(v) for e,v in zip(anchors,c)} for name,c in choices.items()},'info':info,'models':models}
