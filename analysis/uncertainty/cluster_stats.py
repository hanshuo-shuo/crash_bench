"""Exact fixed-weight cluster bootstrap reductions; no training or data loading."""


def draws(states, tasks, repetitions, seed):
    import numpy as np
    if len(states)!=len(set(states)) or len(states)!=len(tasks):
        raise ValueError('Unique paired state/task identities required')
    rng=np.random.Generator(np.random.PCG64(seed))
    result=np.zeros((repetitions,len(states)),dtype=np.int64)
    for task in sorted(set(tasks)):
        indices=sorted((i for i,t in enumerate(tasks) if t==task),key=lambda i:states[i])
        sampled=rng.choice(indices,size=(repetitions,len(indices)),replace=True)
        for column in indices:result[:,column]=(sampled==column).sum(axis=1)
    return result


def interval(values):
    import numpy as np
    values=np.asarray(values,dtype=np.float64);finite=values[np.isfinite(values)]
    low,high=np.percentile(finite,[2.5,97.5]) if len(finite) else [None,None]
    return dict(low=None if low is None else float(low),high=None if high is None else float(high),
                defined=len(finite),undefined=len(values)-len(finite))


def _arrays(scene_ids, weights, states):
    import numpy as np
    index={s:i for i,s in enumerate(states)}
    if len(index)!=len(states) or len(scene_ids)!=len(weights):raise ValueError('Cluster identities/weights differ')
    w=np.asarray(weights,dtype=np.float64)
    if not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:raise ValueError('Nonnegative finite positive-total weights required')
    return np.asarray([index[s] for s in scene_ids]),w/w.sum()


def mean(values, scene_ids, weights, states, multiplicities):
    import numpy as np
    ids,w=_arrays(scene_ids,weights,states);x=np.asarray(values,dtype=np.float64)
    if x.shape!=w.shape or not np.isfinite(x).all():raise ValueError('Finite scalar values required')
    mass=np.bincount(ids,weights=w,minlength=len(states))
    total=np.bincount(ids,weights=w*x,minlength=len(states))
    point=float(total.sum()/mass.sum());denominator=multiplicities@mass
    replicates=np.divide(multiplicities@total,denominator,out=np.full(len(multiplicities),np.nan),where=denominator>0)
    return point,interval(replicates)


def classification(labels, scores, probabilities, scene_ids, weights, states, multiplicities):
    """AUC numerator is a state-pair matrix, preserving ties and repeat draws.

    This equals reweighting every original row by the state's draw multiplicity;
    neither regrouping duplicated state ids nor resampling individual rows occurs.
    """
    import numpy as np
    ids,w=_arrays(scene_ids,weights,states);y=np.asarray(labels);x=np.asarray(scores,dtype=np.float64);p=np.asarray(probabilities,dtype=np.float64)
    if y.shape!=w.shape or x.shape!=w.shape or p.shape!=w.shape or not np.isin(y,[0,1]).all():raise ValueError('Matching binary/scalar arrays required')
    if not np.isfinite(x).all() or not np.isfinite(p).all() or (p<0).any() or (p>1).any():raise ValueError('Finite scores and probabilities in [0,1] required')
    size=len(states);_,bins=np.unique(x,return_inverse=True)
    positive=np.zeros((int(bins.max())+1,size));negative=np.zeros_like(positive)
    np.add.at(positive,(bins[y==1],ids[y==1]),w[y==1])
    np.add.at(negative,(bins[y==0],ids[y==0]),w[y==0])
    pairs=positive.T@(np.cumsum(negative,axis=0)-.5*negative)
    pos=positive.sum(axis=0);neg=negative.sum(axis=0)
    tp=np.bincount(ids,weights=w*((y==1)&(p>=.5)),minlength=size)
    tn=np.bincount(ids,weights=w*((y==0)&(p<.5)),minlength=size)
    squared=np.bincount(ids,weights=w*(p-y)**2,minlength=size)
    mass=np.bincount(ids,weights=w,minlength=size)
    # The same reduction yields the point estimate and all bootstrap replicates.
    m=np.vstack([np.ones(size),np.asarray(multiplicities,dtype=np.float64)])
    pm=m@pos;nm=m@neg
    auc_den=pm*nm
    auc=np.divide(np.einsum('bi,ij,bj->b',m,pairs,m),auc_den,out=np.full(len(m),np.nan),where=auc_den>0)
    tpr=np.divide(m@tp,pm,out=np.full(len(m),np.nan),where=pm>0)
    tnr=np.divide(m@tn,nm,out=np.full(len(m),np.nan),where=nm>0)
    ba=.5*(tpr+tnr);den=m@mass
    brier=np.divide(m@squared,den,out=np.full(len(m),np.nan),where=den>0)
    return {key:dict(value=float(v[0]) if np.isfinite(v[0]) else None,ci=interval(v[1:]))
            for key,v in [('AUROC',auc),('BA',ba),('Brier',brier)]}
