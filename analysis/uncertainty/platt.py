"""Frozen scalar calibration; caller supplies train data only, never test labels."""
import math


def fit(values, labels, weights, parameters):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import expit
    x=np.asarray(values,dtype=np.float64);y=np.asarray(labels,dtype=np.float64)
    w=np.asarray(weights,dtype=np.float64)
    if x.ndim!=1 or y.shape!=x.shape or w.shape!=x.shape or not len(x):
        raise ValueError('Nonempty matching scalar training arrays required')
    if not np.isfinite(x).all() or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:
        raise ValueError('Invalid training values/weights')
    if not np.isin(y,[0.,1.]).all():raise ValueError('Binary labels required')
    if parameters['solver']!='scipy_L-BFGS-B_analytic_gradient' or parameters['slope_l2']<=0 or parameters['intercept_l2']!=0:
        raise ValueError('Frozen regularized solver required')
    keep=w>0;x=x[keep];y=y[keep];w=w[keep];w=w/w.sum()
    center=float(np.dot(w,x));scale=float(np.sqrt(np.dot(w,(x-center)**2)))
    prevalence=float(np.dot(w,y));clip=parameters['fallback_prevalence_clip']
    p=min(1.-clip,max(clip,prevalence));intercept=math.log(p/(1.-p))
    model=dict(mean=center,std=scale,prevalence=prevalence,slope=0.,intercept=intercept,
               slope_l2=parameters['slope_l2'],rows=len(x),weight_sum=1.)
    if np.unique(y).size<2 or scale<=parameters['minimum_std']:
        model.update(status='single_class' if np.unique(y).size<2 else 'constant_input',
                     fallback=True,constant_probability=p,iterations=0,converged=True)
        return model
    z=(x-center)/scale;l2=parameters['slope_l2']

    def objective(coefficient):
        slope,offset=coefficient;linear=slope*z+offset;error=expit(linear)-y
        loss=float(np.dot(w,np.logaddexp(0.,linear)-y*linear)+.5*l2*slope*slope)
        gradient=np.array([np.dot(w,error*z)+l2*slope,np.dot(w,error)])
        return loss,gradient

    result=minimize(objective,np.array([parameters['initial_slope'],intercept]),
                    method='L-BFGS-B',jac=True,options={k:parameters[k] for k in ['maxiter','maxls','gtol','ftol']})
    loss,gradient=objective(result.x)
    if not result.success or not np.isfinite(result.x).all() or not math.isfinite(loss):
        raise RuntimeError('Frozen Platt solver failed: '+str(dict(message=str(result.message),
                           iterations=int(result.nit),loss=loss,gradient=gradient.tolist())))
    model.update(status='regularized_fit',fallback=False,slope=float(result.x[0]),
                 intercept=float(result.x[1]),iterations=int(result.nit),converged=True,
                 loss=loss,gradient=gradient.tolist(),solver_message=str(result.message))
    return model


def predict(model, values):
    import numpy as np
    from scipy.special import expit
    x=np.asarray(values,dtype=np.float64)
    if not np.isfinite(x).all():raise ValueError('Nonfinite prediction input')
    if model['fallback']:return np.full(x.shape,model['constant_probability'],dtype=np.float64)
    return expit(model['slope']*(x-model['mean'])/model['std']+model['intercept'])
