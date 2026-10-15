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


def score(records,arm,bootstrap=False):
    import numpy as np
    from sklearn.metrics import roc_auc_score,brier_score_loss,log_loss
    truth=np.asarray([r['truth'] for r in records]);correct=np.asarray([r['predicted']==r['truth'] for r in records])
    probs=np.asarray([r['p_feasible'] if r['p_feasible'] is not None else .5 for r in records])
    pairs=defaultdict(list);renamings=defaultdict(list)
    for r in records:
        pairs[(r['unit_id'],r['assignment'])].append(r)
        renamings[(r['unit_id'],r['protected_boundary'])].append(r)
    if any(len(v)!=2 for v in list(pairs.values())+list(renamings.values())):
        raise RuntimeError('Evaluation must retain whole counterfactual units')
    result=dict(rows=len(records),source_groups=len({r['source_group'] for r in records}),
        correct=int(correct.sum()),verdict_accuracy=float(correct.mean()),
        by_class={str(c):dict(n=int((truth==c).sum()),correct=int(correct[truth==c].sum())) for c in (0,1)},
        strict_format_valid=sum(r['format_valid'] for r in records),
        unambiguous_verdicts=sum(r['predicted'] is not None for r in records),
        native_binary_scores_available=sum(r['p_feasible'] is not None for r in records),
        counterfactual_both_correct=sum(all(r['predicted']==r['truth'] for r in p) for p in pairs.values())/len(pairs),
        counterfactual_flip_rate=sum(all(r['predicted'] in (0,1) for r in p) and p[0]['predicted']!=p[1]['predicted'] for p in pairs.values())/len(pairs),
        color_renaming_invariance=sum(all(r['predicted'] in (0,1) for r in p) and p[0]['predicted']==p[1]['predicted'] for p in renamings.values())/len(renamings),
        auroc_invalid_scores_neutral=float(roc_auc_score(truth,probs)),brier_invalid_scores_neutral=float(brier_score_loss(truth,probs)),
        log_loss_invalid_scores_neutral=float(log_loss(truth,probs,labels=[0,1])),
        probability_scope='Binary-normalized logits at the actual free-generated verdict token. Invalid or unscorable decisions use 0.5 for all-row diagnostics; these are not calibrated feasibility probabilities.',
        gold_rationale_prefix_used=False)
    if arm=='B':
        result['rationale_grades']=dict(Counter(r['rationale_assessment']['status'] for r in records))
        result['verified_rationale_and_correct']=sum(r['predicted']==r['truth'] and r['rationale_assessment']['status']=='verified_template' for r in records)
    if bootstrap:
        groups=np.asarray([r['source_group'] for r in records]);unique=sorted(set(groups));rng=np.random.default_rng(1014);metrics=[]
        for _ in range(1000):
            idx=np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(unique,len(unique),replace=True)])
            metrics.append([correct[idx].mean(),roc_auc_score(truth[idx],probs[idx])])
        q=np.quantile(metrics,[.025,.975],axis=0)
        result['source_bootstrap_95pct']=dict(verdict_accuracy=q[:,0].tolist(),auroc=q[:,1].tolist(),replicates=1000)
    return result


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
