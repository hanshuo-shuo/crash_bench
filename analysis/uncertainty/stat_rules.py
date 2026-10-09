"""Pre-action forecast and grouped metric definitions, independent of model code."""
import math


def collision_target(step, collision, terminal_step, safe_success, horizon):
    """None means post-event/censored; current action belongs to the event window."""
    if collision is not None and collision < step:return None
    if collision is not None and step <= collision <= step+horizon-1:return 1
    if safe_success or terminal_step >= step+horizon-1:return 0
    return None


def first_five_eligible(inferences, collision, terminal_step):
    # Fifth predictor precedes action21. Outcomes already known before21 are
    # not prospective predictors; collision at21 can be a legitimate target.
    return len(inferences)>=5 and [x['step'] for x in inferences[:5]]==[1,6,11,16,21] and terminal_step>=21 and (collision is None or collision>=21)


def grouped_weights(scene_ids, rollout_ids):
    groups={}
    for scene,run in zip(scene_ids,rollout_ids):groups.setdefault(scene,{}).setdefault(run,0);groups[scene][run]+=1
    return [1./len(groups)/len(groups[scene])/groups[scene][run] for scene,run in zip(scene_ids,rollout_ids)]


def weighted_auc(labels,scores,weights):
    """Weighted pairwise concordance with half credit for every score tie."""
    positive=sum(w for y,w in zip(labels,weights) if y==1)
    negative=sum(w for y,w in zip(labels,weights) if y==0)
    if not positive or not negative:return None
    bins={}
    for y,s,w in zip(labels,scores,weights):
        value=bins.setdefault(float(s),[0.,0.]);value[int(y)]+=w
    numerator=0.;below_negative=0.
    for score in sorted(bins):
        n,p=bins[score];numerator+=p*(below_negative+.5*n);below_negative+=n
    return numerator/(positive*negative)


def weighted_quantile(values,weights,quantile):
    if not 0<quantile<1 or len(values)!=len(weights) or not values:raise ValueError('Invalid quantile inputs')
    total=sum(weights);target=total*quantile;running=0.
    if total<=0 or any(w<0 for w in weights):raise ValueError('Invalid quantile weights')
    for value,weight in sorted(zip(values,weights)):
        running+=weight
        if running>=target:return float(value)
    return float(max(values))


def classification_metrics(labels,scores,probabilities,weights):
    positive=sum(w for y,w in zip(labels,weights) if y==1)
    negative=sum(w for y,w in zip(labels,weights) if y==0)
    tp=sum(w for y,p,w in zip(labels,probabilities,weights) if y==1 and p>=.5)
    tn=sum(w for y,p,w in zip(labels,probabilities,weights) if y==0 and p<.5)
    brier=sum(w*(p-y)**2 for y,p,w in zip(labels,probabilities,weights))/sum(weights) if weights else None
    return dict(AUROC=weighted_auc(labels,scores,weights),
                BA=.5*(tp/positive+tn/negative) if positive and negative else None,Brier=brier)
