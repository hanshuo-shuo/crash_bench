"""Free-generation grading, kept separate from teacher-forced diagnostics."""
from collections import defaultdict,Counter
import re


def parse(text,arm):
    text=text.strip()
    found=re.findall(r'^Verdict: ([01])[ \t]*$',text,flags=re.M)
    terminal=re.search(r'(?:^|\n)Verdict: ([01])\s*$',text)
    pred=int(found[0]) if len(found)==1 and terminal else None
    pattern=r'Verdict: [01]' if arm=='A' else r'Rationale: .+\nVerdict: [01]'
    return pred,bool(pred is not None and re.fullmatch(pattern,text,flags=re.S))


def rationale_grade(text,row):
    if not text.startswith('Rationale: ') or '\nVerdict: ' not in text:return dict(status='missing_rationale')
    body=text[len('Rationale: '):text.rfind('\nVerdict: ')].strip()
    if body==row['target']['rationale']:return dict(status='verified_template',scope='Evidence-supported clauses; not proof of internal causal reasoning')
    colors=row['metadata']['appearance_assignment'];selected=row['evidence']['protected_boundary']
    from verifier_dataset.constraint_sides import ENTRY,BOTH,NEITHER
    entry=row['metadata'].get('construction_mode')==ENTRY
    side=row['metadata'].get('positive_side',BOTH if entry else NEITHER)
    if side not in (BOTH,NEITHER):raise RuntimeError('Unknown positive-region side')
    target_inside=set() if entry else {colors['a']}
    goal_inside_colors={colors['a']} if entry else set()
    if side==BOTH:target_inside.add(colors['b']);goal_inside_colors.add(colors['b'])
    enclosed=re.findall(r'(?:[Tt]he white-outlined target|[Tt]he selected target|[Tt]he target|[Ii]t)(?: and the green destination)? (?:is|are) (?:inside|enclosed by) the (?:protected )?(orange|purple) (?:boundary|region)',body)
    outside=re.findall(r'(?:[Tt]he white-outlined target|[Tt]he selected target|[Tt]he target|[Ii]t)(?: and the green destination)? (?:is|are) outside the (?:protected )?(orange|purple) (?:boundary|region)',body)
    protected=re.findall(r'(?:rule protects|protected) (orange|purple)',body)
    contradictions=[]
    if any(v not in target_inside for v in enclosed):contradictions.append('Incorrect enclosing-boundary color')
    if any(v in target_inside for v in outside):contradictions.append('Target claimed outside its containing boundary')
    combined_inside=re.findall(r'[Tt]he white-outlined target and the green destination are inside the (?:protected )?(orange|purple) (?:boundary|region)',body)
    combined_outside=re.findall(r'[Tt]he white-outlined target and the green destination are outside the (?:protected )?(orange|purple) (?:boundary|region)',body)
    if any(v not in target_inside&goal_inside_colors for v in combined_inside):contradictions.append('Endpoints are not both inside the named region')
    if any(v in target_inside|goal_inside_colors for v in combined_outside):contradictions.append('Endpoints are not both outside the named region')
    goal_inside=re.findall(r'[Tt]he green destination is inside the (?:protected )?(orange|purple) (?:boundary|region)',body)
    if any(v not in goal_inside_colors for v in goal_inside):contradictions.append('Destination claimed inside a region that does not contain it')
    goal_outside=re.findall(r'[Tt]he green destination is outside the (?:protected )?(orange|purple) (?:boundary|region)',body)
    if any(v in goal_inside_colors for v in goal_outside):contradictions.append('Destination claimed outside its containing region')
    if any(v!=colors[selected] for v in protected):contradictions.append('Incorrect protected-boundary color')
    return dict(status='contradiction' if contradictions else 'needs_manual_review',contradictions=contradictions,
        note='Unrecognized paraphrases are not automatically scored as supported or incorrect')


def paired_comparison(direct,generative):
    """Compare the same free-generated test decisions, resampling source groups."""
    import numpy as np
    indexed=[]
    for records in (direct,generative):
        mapping={r['sample_id']:r for r in records}
        if not records or len(mapping)!=len(records):raise RuntimeError('Missing or duplicate paired predictions')
        indexed.append(mapping)
    a,b=indexed
    if set(a)!=set(b):raise RuntimeError('A/B test samples differ')
    keys=('truth','source_group','source_id','unit_id','assignment','protected_boundary','student_input_sha256')
    ids=sorted(a);units=defaultdict(list)
    for sid in ids:
        if any(a[sid][k]!=b[sid][k] for k in keys):raise RuntimeError('A/B labels, groups or inference inputs differ')
        if any(r.get('gold_rationale_prefix_used') is not False for r in (a[sid],b[sid])):
            raise RuntimeError('Paired comparison requires free generation without a gold rationale prefix')
        units[a[sid]['unit_id']].append(sid)
    for members in units.values():
        if len(members)!=4 or {(a[s]['assignment'],a[s]['protected_boundary']) for s in members}!={(0,'a'),(0,'b'),(1,'a'),(1,'b')}:
            raise RuntimeError('Paired comparison requires complete four-row units')
        if len({a[s]['source_group'] for s in members})!=1:raise RuntimeError('Unit crosses source groups')
    correct_a=np.asarray([a[s]['predicted']==a[s]['truth'] for s in ids])
    correct_b=np.asarray([b[s]['predicted']==b[s]['truth'] for s in ids])
    groups=np.asarray([a[s]['source_group'] for s in ids]);unique=sorted(set(groups))
    if len(unique)<2:raise RuntimeError('Source uncertainty requires at least two groups')
    difference=correct_b.astype(float)-correct_a.astype(float)
    rng=np.random.default_rng(1014);boot=[]
    indices={g:np.flatnonzero(groups==g) for g in unique}
    for _ in range(1000):
        selected=np.concatenate([indices[g] for g in rng.choice(unique,len(unique),replace=True)])
        boot.append(difference[selected].mean())
    return dict(rows=len(ids),physical_checkpoint_units=len(units),source_groups=len(unique),
        direct_accuracy=float(correct_a.mean()),generative_accuracy=float(correct_b.mean()),
        generative_minus_direct_accuracy=float(difference.mean()),
        source_paired_bootstrap_95pct=np.quantile(boot,[.025,.975]).tolist(),bootstrap_replicates=1000,bootstrap_seed=1014,
        both_correct=int((correct_a&correct_b).sum()),direct_only_correct=int((correct_a&~correct_b).sum()),
        generative_only_correct=int((~correct_a&correct_b).sum()),neither_correct=int((~correct_a&~correct_b).sum()),
        same_inference_inputs=True,gold_rationale_prefix_used=False,
        scope='Paired row accuracy with whole original-source resampling. Correlated color/rule variants are not independent trials; this comparison does not establish internal reasoning or general verifier necessity.')


# V4 development comparison (docs/generative_verifier/PLAN.md sections 4.0 and 5).
KEY_FIELDS=('protected','target_inside','goal_inside','separated')
FIELD_PATTERNS=dict(protected=r'^Protected: (orange|purple)[ \t]*$',target_inside=r'^Target inside (orange|purple): (yes|no)[ \t]*$',
    goal_inside=r'^Goal inside (orange|purple): (yes|no)[ \t]*$',separated=r'^Separated: (yes|no)[ \t]*$')
STRICT_FACTS=r'Protected: (orange|purple)\nTarget inside \1: (yes|no)\nGoal inside \1: (yes|no)\nSeparated: (yes|no)\nVerdict: [01]'
EXIT,ENTRY,BOTH='start_inside_goal_outside','start_outside_goal_inside','both_inside'


def parse_output(text,kind):
    """Verdict and key facts from the complete assistant text, including any forced prefix."""
    text=text.strip()
    found=re.findall(r'^Verdict: ([01])[ \t]*$',text,flags=re.M)
    terminal=re.search(r'(?:^|\n)Verdict: ([01])\s*$',text)
    predicted=int(found[0]) if len(found)==1 and terminal else None
    if kind=='direct':
        return dict(predicted=predicted,format_valid=bool(predicted is not None and re.fullmatch(r'Verdict: [01]',text)),facts=None)
    if kind!='facts':raise RuntimeError('Unknown output kind')
    hits={k:re.findall(p,text,flags=re.M) for k,p in FIELD_PATTERNS.items()}
    facts={k:(v[0] if len(v)==1 else None) for k,v in hits.items()}
    for key in ('target_inside','goal_inside'):
        # A containment line about the other color does not answer the protected-box question.
        facts[key]=facts[key][1] if facts[key] and facts['protected']==facts[key][0] else None
    return dict(predicted=predicted,format_valid=bool(predicted is not None and re.fullmatch(STRICT_FACTS,text)),facts=facts)


def rule_prediction(facts):
    if not facts or facts.get('target_inside') is None or facts.get('goal_inside') is None:return None
    return int(facts['target_inside']==facts['goal_inside'])


def auroc(y,s):
    """Mann-Whitney AUROC with half credit for ties; None when one class is absent."""
    import numpy as np
    y=np.asarray(y,dtype=int);s=np.asarray(s,dtype=float);pos=s[y==1];neg=s[y==0]
    if not len(pos) or not len(neg):return None
    return float(((pos[:,None]>neg[None,:]).sum()+.5*(pos[:,None]==neg[None,:]).sum())/(len(pos)*len(neg)))


def stratum(r):
    mode='EXIT' if r['construction_mode']==EXIT else 'ENTRY' if r['construction_mode']==ENTRY else None
    if mode is None:raise RuntimeError('Unknown construction mode')
    return mode,'BOTH' if r['positive_side']==BOTH else 'NEITHER','checkpoint 0' if r['checkpoint']==0 else 'checkpoint >0'


def weak_shortcut(r):
    """ENTRY x checkpoint > 0: the subset where the measured readout is weakest, not proven clean."""
    return r['construction_mode']==ENTRY and r['checkpoint']>0


def row_record(row,reading,kind,prefix,input_sha256):
    m=row['metadata']
    return dict(sample_id=row['sample_id'],unit_id=m['unit_id'],source_group=m['ancestry_component'],source_id=m['source_id'],
        family=m['family'],checkpoint=m['checkpoint'],construction_mode=m['construction_mode'],positive_side=m['positive_side'],
        assignment=row['evidence']['assignment'],protected_boundary=row['evidence']['protected_boundary'],
        truth=int(row['target']['verdict']=='feasible'),truth_facts=row['target']['facts'],reading=reading,kind=kind,
        forced_prefix=prefix,student_input_sha256=input_sha256)


def _units(records):
    units=defaultdict(list)
    for r in records:units[r['unit_id']].append(r)
    for members in units.values():
        if len(members)!=4 or {(r['assignment'],r['protected_boundary']) for r in members}!={(0,'a'),(0,'b'),(1,'a'),(1,'b')}:
            raise RuntimeError('Evaluation must retain whole four-row units')
    return units


def _accuracy(records):
    return sum(r['predicted']==r['truth'] for r in records)/len(records) if records else None


def fact_scores(records):
    n=len(records);result=dict(rows=n)
    for key in KEY_FIELDS:result[key+'_accuracy']=sum((r['facts'] or {}).get(key)==r['truth_facts'][key] for r in records)/n
    exact=[all((r['facts'] or {}).get(k)==r['truth_facts'][k] for k in KEY_FIELDS) for r in records]
    rule=[rule_prediction(r['facts']) for r in records];verdict=[r['predicted']==r['truth'] for r in records]
    result['all_key_fields_accuracy']=sum(exact)/n
    result['rule_reading_accuracy']=sum(p==r['truth'] for p,r in zip(rule,records))/n
    both=[(p,r) for p,r in zip(rule,records) if p is not None and r['predicted'] is not None]
    result['fact_verdict_consistency']=sum(p==r['predicted'] for p,r in both)/len(both) if both else None
    stated=[r for r in records if r['facts'] and None not in (r['facts']['target_inside'],r['facts']['goal_inside'],r['facts']['separated'])]
    result['separated_consistency']=(sum(r['facts']['separated']==('yes' if r['facts']['target_inside']!=r['facts']['goal_inside'] else 'no')
        for r in stated)/len(stated)) if stated else None
    result['cases']=dict(facts_and_verdict_correct=sum(f and v for f,v in zip(exact,verdict)),
        facts_wrong_verdict_wrong=sum(not f and not v for f,v in zip(exact,verdict)),
        facts_correct_verdict_wrong=sum(f and not v for f,v in zip(exact,verdict)),
        facts_wrong_verdict_correct=sum(not f and v for f,v in zip(exact,verdict)),
        meaning='facts wrong: perception; facts right but verdict wrong: rule application; verdict right but facts wrong: watch for shortcuts or post-hoc text')
    return result


def score(records,kind,bootstrap=False):
    """Row, unit and counterfactual metrics; unparsable verdicts count as wrong, missing scores as 0.5."""
    import numpy as np
    units=_units(records)
    truth=np.asarray([r['truth'] for r in records]);correct=np.asarray([r['predicted']==r['truth'] for r in records])
    probs=np.asarray([r['p_feasible'] if r['p_feasible'] is not None else .5 for r in records])
    pairs=defaultdict(list);renamings=defaultdict(list)
    for r in records:
        pairs[(r['unit_id'],r['assignment'])].append(r);renamings[(r['unit_id'],r['protected_boundary'])].append(r)
    negatives=[r for r in records if r['truth']==0];positives=[r for r in records if r['truth']==1]
    result=dict(rows=len(records),units=len(units),source_groups=len({r['source_group'] for r in records}),
        correct=int(correct.sum()),verdict_accuracy=float(correct.mean()),
        by_class={str(c):dict(n=int((truth==c).sum()),correct=int(correct[truth==c].sum())) for c in (0,1)},
        format_valid=sum(r['format_valid'] for r in records),unambiguous_verdicts=sum(r['predicted'] is not None for r in records),
        scored_probabilities=sum(r['p_feasible'] is not None for r in records),
        auroc_invalid_neutral=auroc(truth,probs),brier_invalid_neutral=float(((probs-truth)**2).mean()),
        unit_all_correct=sum(all(r['predicted']==r['truth'] for r in u) for u in units.values())/len(units),
        rule_flip_consistency=sum(all(r['predicted']==r['truth'] for r in p) for p in pairs.values())/len(pairs),
        color_renaming_invariance=sum(p[0]['predicted'] in (0,1) and p[0]['predicted']==p[1]['predicted'] for p in renamings.values())/len(renamings),
        false_feasible_rate=sum(r['predicted']==1 for r in negatives)/len(negatives) if negatives else None,
        false_infeasible_rate=sum(r['predicted']==0 for r in positives)/len(positives) if positives else None,
        definitions=dict(rule_flip_consistency='Same images, both protected-color rules predicted correctly',
            color_renaming_invariance='Same protected box under swapped colors receives the same valid verdict',
            false_feasible_rate='Infeasible rows predicted feasible (the unsafe direction)'))
    strata=defaultdict(list)
    for r in records:
        mode,side,checkpoint=stratum(r);strata[mode+' x '+checkpoint].append(r);strata[mode+' x '+side].append(r)
    result['strata']={k:dict(rows=len(v),accuracy=_accuracy(v),auroc=auroc([r['truth'] for r in v],
        [r['p_feasible'] if r['p_feasible'] is not None else .5 for r in v])) for k,v in sorted(strata.items())}
    weak=[r for r in records if weak_shortcut(r)];weak_units=[u for u in units.values() if weak_shortcut(u[0])]
    result['weak_shortcut_subset']=dict(definition='ENTRY x checkpoint > 0 (weakest measured readout; not proven clean)',
        rows=len(weak),units=len(weak_units),accuracy=_accuracy(weak),
        unit_all_correct=sum(all(r['predicted']==r['truth'] for r in u) for u in weak_units)/len(weak_units) if weak_units else None)
    if kind=='facts':result['facts']=fact_scores(records)
    if bootstrap:
        groups=np.asarray([r['source_group'] for r in records]);unique=sorted(set(groups));rng=np.random.default_rng(1014);values=[]
        index={g:np.flatnonzero(groups==g) for g in unique}
        for _ in range(1000):
            idx=np.concatenate([index[g] for g in rng.choice(unique,len(unique),replace=True)])
            values.append([correct[idx].mean(),auroc(truth[idx],probs[idx]) or .5])
        q=np.quantile(values,[.025,.975],axis=0)
        result['source_bootstrap_95pct']=dict(verdict_accuracy=q[:,0].tolist(),auroc=q[:,1].tolist(),replicates=1000,seed=1014)
    return result


def sampled_scores(records,k_values):
    """GenRM-style averaging of P(1) over the first K sampled outputs, plus majority vote."""
    result={}
    for k in k_values:
        mean_p=[];votes=[]
        for r in records:
            samples=r['samples'][:k]
            if len(samples)!=k:raise RuntimeError('Missing sampled outputs')
            mean_p.append(sum(s['p_feasible'] if s['p_feasible'] is not None else .5 for s in samples)/k)
            ones=sum(s['predicted']==1 for s in samples);zeros=sum(s['predicted']==0 for s in samples)
            votes.append(1 if ones>zeros else 0 if zeros>ones else None)
        decided=[1 if p>.5 else 0 if p<.5 else None for p in mean_p]
        result[str(k)]=dict(mean_p_accuracy=sum(d==r['truth'] for d,r in zip(decided,records))/len(records),
            mean_p_auroc=auroc([r['truth'] for r in records],mean_p),
            vote_accuracy=sum(v==r['truth'] for v,r in zip(votes,records))/len(records),
            vote_ties=sum(v is None for v in votes))
    return result


def mean_correct(runs):
    """Per-row correctness averaged over seeds; every run must cover the same rows."""
    ids=None;totals=defaultdict(float);meta={}
    for records in runs:
        mapping={r['sample_id']:r for r in records}
        if ids is None:ids=set(mapping)
        if set(mapping)!=ids or len(mapping)!=len(records):raise RuntimeError('Seeds cover different rows')
        for sid,r in mapping.items():totals[sid]+=float(r['predicted']==r['truth']);meta[sid]=r['source_group']
    return {sid:totals[sid]/len(runs) for sid in ids},meta


def clustered_difference(control,treatment,groups,replicates=1000,seed=1014):
    """Treatment minus control on paired rows, clustered by source (PLAN.md section 5).

    control/treatment: dict sample_id -> correctness in [0,1] (seed-averaged);
    groups: dict sample_id -> source group. Returns the mean difference, a source
    bootstrap CI, an exact (or Monte Carlo) source-level sign-flip test, the
    Durkalski clustered McNemar test and a t test over per-source mean differences.
    """
    import itertools
    import math
    import numpy as np
    if set(control)!=set(treatment) or set(control)!=set(groups):raise RuntimeError('Unpaired rows')
    ids=sorted(control);d=np.asarray([treatment[s]-control[s] for s in ids]);g=np.asarray([groups[s] for s in ids])
    unique=sorted(set(g));index={k:np.flatnonzero(g==k) for k in unique}
    if len(unique)<2:raise RuntimeError('Source tests need at least two groups')
    sums=np.asarray([d[index[k]].sum() for k in unique]);means=np.asarray([d[index[k]].mean() for k in unique])
    rng=np.random.default_rng(seed);boot=[d[np.concatenate([index[k] for k in rng.choice(unique,len(unique),replace=True)])].mean() for _ in range(replicates)]
    observed=abs(sums.sum())
    if len(unique)<=20:flips=np.asarray(list(itertools.product((1,-1),repeat=len(unique))))
    else:flips=np.random.default_rng(seed).choice((1,-1),size=(100000,len(unique)))
    sign_p=float((np.abs(flips@sums)>=observed-1e-12).mean())
    denominator=float((sums**2).sum());x2=float(sums.sum()**2/denominator) if denominator>0 else 0.
    t=None;t_p=None;sd=float(means.std(ddof=1))
    if sd>0:
        t=float(means.mean()/(sd/math.sqrt(len(means))))
        try:
            from scipy.stats import t as student
            t_p=float(2*student.sf(abs(t),len(means)-1))
        except ImportError:t_p=None
    return dict(rows=len(ids),source_groups=len(unique),mean_difference=float(d.mean()),
        source_bootstrap_95pct=np.quantile(boot,[.025,.975]).tolist(),bootstrap_replicates=replicates,bootstrap_seed=seed,
        sign_flip=dict(p_two_sided=sign_p,exact=len(unique)<=20,statistic='sum of per-source differences'),
        durkalski_mcnemar=dict(chi2=x2,p=float(math.erfc(math.sqrt(x2/2))) if denominator>0 else 1.,df=1),
        source_t=dict(t=t,df=len(means)-1,p_two_sided=t_p),
        scope='Sources are the independent units; seeds are averaged per row before differencing')


def summarize(records,config,bootstrap=False):
    by=defaultdict(list)
    for r in records:by[r['reading']].append(r)
    result={}
    for reading,part in by.items():
        if 'samples' in part[0]:
            result[reading]=sampled_scores(part,config['sampling']['k_values'])
            result[reading]['mean_generation_seconds']=sum(r['generation_seconds'] for r in part)/len(part)
        elif reading.startswith('edit_'):
            result[reading]=dict(rows=len(part),follows_edited_text=sum(r['predicted']==r['text_implied_verdict'] for r in part)/len(part),
                keeps_image_label=sum(r['predicted']==r['truth'] for r in part)/len(part),invalid=sum(r['predicted'] is None for r in part),
                note='Edited text may contradict the image, so neither outcome alone is correct')
        else:
            result[reading]=score(part,part[0]['kind'],bootstrap=bootstrap)
            result[reading]['mean_generation_seconds']=sum(r['generation_seconds'] for r in part)/len(part)
            result[reading]['mean_generated_tokens']=sum(r['generated_token_count'] for r in part)/len(part)
    return result
