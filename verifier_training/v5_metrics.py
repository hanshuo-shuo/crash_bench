"""V5 S3 free-generation grading (docs/generative_verifier/V5_S3_PROTOCOL.md, readouts R1-R5)."""
from collections import Counter,defaultdict
import re

FACT_KEYS=('line','target_robot_side','goal_robot_side','separated')
FIELD_PATTERNS=dict(line=r'^Line: (.+?)[ \t]*$',target_robot_side=r'^Target on robot side: (yes|no)[ \t]*$',
    goal_robot_side=r'^Goal on robot side: (yes|no)[ \t]*$',separated=r'^Separated: (yes|no)[ \t]*$')
STRICT_FACTS=r'Line: (.+)\nTarget on robot side: (yes|no)\nGoal on robot side: (yes|no)\nSeparated: (yes|no)\nVerdict: [01]'
REPLICATES,SEED=1000,1014


def parse_output(text,kind):
    """Verdict and facts from the complete assistant text; a verdict needs exactly one terminal Verdict line."""
    text=text.strip()
    found=re.findall(r'^Verdict: ([01])[ \t]*$',text,flags=re.M)
    terminal=re.search(r'(?:^|\n)Verdict: ([01])\s*$',text)
    predicted=int(found[0]) if len(found)==1 and terminal else None
    if kind=='direct':
        return dict(predicted=predicted,format_valid=bool(predicted is not None and re.fullmatch(r'Verdict: [01]',text)),facts=None)
    if kind!='facts':raise RuntimeError('Unknown output kind')
    hits={k:re.findall(p,text,flags=re.M) for k,p in FIELD_PATTERNS.items()}
    facts={k:(v[0] if len(v)==1 else None) for k,v in hits.items()}
    return dict(predicted=predicted,format_valid=bool(predicted is not None and re.fullmatch(STRICT_FACTS,text)),facts=facts)


def program_verdict(facts):
    """Verdict implied by the model's own two robot-side facts: 0 when exactly one endpoint is on the robot side."""
    if not facts or facts.get('target_robot_side') is None or facts.get('goal_robot_side') is None:return None
    return int(facts['target_robot_side']==facts['goal_robot_side'])


def row_record(row,reading,kind,prefix,input_sha256,lookup=None):
    m=row['metadata']
    return dict(sample_id=row['sample_id'],split=row['split'],image_id=m['image_id'],source_id=m['source_id'],layout=m['layout'],
        layout_kind=m['layout_kind'],rule=m['rule'],time_bin=m['time_bin'],checkpoint=m['checkpoint'],truth=int(row['target']['digit']),
        truth_facts=row['target']['facts'],reading=reading,kind=kind,forced_prefix=prefix,student_input_sha256=input_sha256,
        lookup_prediction=lookup)


def auroc(y,s):
    """Mann-Whitney AUROC with half credit for ties; None when one class is absent."""
    import numpy as np
    y=np.asarray(y,dtype=int);s=np.asarray(s,dtype=float);pos=s[y==1];neg=s[y==0]
    if not len(pos) or not len(neg):return None
    return float(((pos[:,None]>neg[None,:]).sum()+.5*(pos[:,None]==neg[None,:]).sum())/(len(pos)*len(neg)))


def cluster_interval(values,groups,replicates=REPLICATES,seed=SEED):
    """95% percentile interval of the mean, resampling whole clusters (images, or sources for reference)."""
    import numpy as np
    values=np.asarray(values,dtype=float);groups=np.asarray(groups);unique=sorted(set(groups.tolist()))
    if len(unique)<2:return None
    index={g:np.flatnonzero(groups==g) for g in unique};rng=np.random.default_rng(seed)
    boot=[values[np.concatenate([index[g] for g in rng.choice(unique,len(unique),replace=True)])].mean() for _ in range(replicates)]
    return np.quantile(boot,[.025,.975]).tolist()


def _accuracy(records):
    return sum(r['predicted']==r['truth'] for r in records)/len(records) if records else None


def fact_scores(records):
    """R3: each fact, format, and whether the verdict follows the model's own facts."""
    n=len(records);result=dict(rows=n)
    got=lambda r,k:(r['facts'] or {}).get(k)
    for key in FACT_KEYS:result[key+'_accuracy']=sum(got(r,key)==r['truth_facts'][key] for r in records)/n
    sides=[(got(r,'target_robot_side')==r['truth_facts']['target_robot_side'],got(r,'goal_robot_side')==r['truth_facts']['goal_robot_side']) for r in records]
    both=[t and g for t,g in sides];verdict=[r['predicted']==r['truth'] for r in records]
    program=[program_verdict(r['facts']) for r in records]
    result.update(both_sides_accuracy=sum(both)/n,all_key_fields_accuracy=sum(all(got(r,k)==r['truth_facts'][k] for k in FACT_KEYS) for r in records)/n,
        format_valid_rate=sum(r['format_valid'] for r in records)/n,
        program_verdict_accuracy=sum(p==r['truth'] for p,r in zip(program,records))/n)
    stated=[(p,r) for p,r in zip(program,records) if p is not None and r['predicted'] is not None]
    result['fact_verdict_consistency']=sum(p==r['predicted'] for p,r in stated)/len(stated) if stated else None
    result['fact_verdict_consistency_rows']=len(stated)
    own=[r for r in records if None not in (got(r,'target_robot_side'),got(r,'goal_robot_side'),got(r,'separated'))]
    result['separated_consistency']=(sum(got(r,'separated')==('yes' if got(r,'target_robot_side')!=got(r,'goal_robot_side') else 'no')
        for r in own)/len(own)) if own else None
    result['side_errors']=dict(target_only=sum(not t and g for t,g in sides),goal_only=sum(t and not g for t,g in sides),
        both=sum(not t and not g for t,g in sides),none=sum(both))
    result['cases']=dict(sides_and_verdict_correct=sum(f and v for f,v in zip(both,verdict)),
        sides_correct_verdict_wrong=sum(f and not v for f,v in zip(both,verdict)),
        sides_wrong_verdict_correct=sum(not f and v for f,v in zip(both,verdict)),
        sides_wrong_verdict_wrong=sum(not f and not v for f,v in zip(both,verdict)),
        meaning='sides right but verdict wrong: composition; sides wrong: perception (see side_errors); '
            'verdict right with wrong sides: watch for shortcuts')
    return result


def score(records,kind,bootstrap=True):
    """R1/R2 (or R4/R5 on other row sets) for one reading; unparsable verdicts count as wrong, missing scores as 0.5."""
    import numpy as np
    truth=np.asarray([r['truth'] for r in records]);correct=np.asarray([r['predicted']==r['truth'] for r in records],dtype=float)
    probs=np.asarray([r['p_feasible'] if r['p_feasible'] is not None else .5 for r in records])
    images=defaultdict(list)
    for r in records:images[r['image_id']].append(r)
    twins=[rs for rs in images.values() if {r['truth'] for r in rs}=={0,1}]
    negatives=[r for r in records if r['truth']==0];positives=[r for r in records if r['truth']==1]
    result=dict(rows=len(records),images=len(images),sources=len({r['source_id'] for r in records}),correct=int(correct.sum()),
        verdict_accuracy=float(correct.mean()),by_class={str(c):dict(n=int((truth==c).sum()),correct=int(correct[truth==c].sum())) for c in (0,1)},
        format_valid=sum(r['format_valid'] for r in records),unambiguous_verdicts=sum(r['predicted'] is not None for r in records),
        predicted_one_rate=sum(r['predicted']==1 for r in records)/len(records),
        scored_probabilities=sum(r['p_feasible'] is not None for r in records),
        auroc_invalid_neutral=auroc(truth,probs),brier_invalid_neutral=float(((probs-truth)**2).mean()),
        twin_images=len(twins),twin_image_all_correct=sum(all(r['predicted']==r['truth'] for r in rs) for rs in twins)/len(twins) if twins else None,
        image_all_correct=sum(all(r['predicted']==r['truth'] for r in rs) for rs in images.values())/len(images),
        false_feasible_rate=sum(r['predicted']==1 for r in negatives)/len(negatives) if negatives else None,
        false_infeasible_rate=sum(r['predicted']==0 for r in positives)/len(positives) if positives else None,
        definitions=dict(twin_image_all_correct='Images carrying both answers whose every row is correct (R2)',
            false_feasible_rate='Separated rows predicted feasible (the unsafe direction)',
            lookup_agreement='Predictions equal to the training rule x progress lookup (the shortcut)'))
    looked=[r for r in records if r.get('lookup_prediction') is not None]
    if looked:
        result.update(lookup_accuracy=sum(r['lookup_prediction']==r['truth'] for r in looked)/len(looked),
            lookup_agreement=sum(r['predicted']==r['lookup_prediction'] for r in looked)/len(looked))
    strata=defaultdict(list)
    for r in records:strata['time_bin %s'%r['time_bin']].append(r);strata['source '+r['source_id'].rsplit('/',1)[-1]].append(r)
    result['strata']={k:dict(rows=len(v),accuracy=_accuracy(v)) for k,v in sorted(strata.items())}
    if kind=='facts':result['facts']=fact_scores(records)
    if bootstrap:
        result['image_bootstrap_95pct']=dict(verdict_accuracy=cluster_interval(correct,[r['image_id'] for r in records]),
            replicates=REPLICATES,seed=SEED)
        result['source_bootstrap_95pct_reference']=dict(verdict_accuracy=cluster_interval(correct,[r['source_id'] for r in records]),
            sources=result['sources'],note='Few sources; reference only')
    return result


def paired_difference(control,treatment):
    """Treatment minus control (B - A) on the same rows; image-bootstrap 95% interval, source clusters for reference."""
    import numpy as np
    a={r['sample_id']:r for r in control};b={r['sample_id']:r for r in treatment}
    if not a or set(a)!=set(b) or len(a)!=len(control) or len(b)!=len(treatment):raise RuntimeError('Unpaired or duplicate records')
    ids=sorted(a)
    for s in ids:
        if any(a[s][k]!=b[s][k] for k in ('truth','image_id','source_id','student_input_sha256')):raise RuntimeError('Paired rows differ')
        if a[s]['forced_prefix'] or b[s]['forced_prefix']:raise RuntimeError('Paired comparison needs free generation')
    ca=np.asarray([a[s]['predicted']==a[s]['truth'] for s in ids]);cb=np.asarray([b[s]['predicted']==b[s]['truth'] for s in ids])
    d=cb.astype(float)-ca.astype(float)
    return dict(rows=len(ids),images=len({a[s]['image_id'] for s in ids}),control_accuracy=float(ca.mean()),treatment_accuracy=float(cb.mean()),
        difference=float(d.mean()),image_bootstrap_95pct=cluster_interval(d,[a[s]['image_id'] for s in ids]),
        source_bootstrap_95pct_reference=cluster_interval(d,[a[s]['source_id'] for s in ids]),replicates=REPLICATES,seed=SEED,
        both_correct=int((ca&cb).sum()),control_only_correct=int((ca&~cb).sum()),treatment_only_correct=int((~ca&cb).sum()),
        neither_correct=int((~ca&~cb).sum()))


def summarize(records,bootstrap=True):
    by=defaultdict(list)
    for r in records:by[r['reading']].append(r)
    result={}
    for reading,part in by.items():
        result[reading]=score(part,part[0]['kind'],bootstrap=bootstrap)
        result[reading]['mean_generation_seconds']=sum(r['generation_seconds'] for r in part)/len(part)
        result[reading]['mean_generated_tokens']=sum(r['generated_token_count'] for r in part)/len(part)
    return result
