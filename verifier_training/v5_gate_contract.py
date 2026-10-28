"""V5 offline gate (docs/generative_verifier/V5_GATE_OFFLINE_PROTOCOL.md): pi0.5 plus a verifier, from existing records.

When the rule arrives the verifier judges it once. A verdict of infeasible stops the robot (no violation, no completion);
feasible hands the task back to pi0.5, whose outcome is the HIL run under that prompt (C2 for the feasible rule, C3 for
the infeasible one). The questions are the 60 HIL states from validation sources, each with its C2 and C3 rule. The
verdicts are the stored validation records of A (S3 and the seed campaign), P (S4 and the seed campaign, program
verdict) and S (S5, its own Verdict). No new execution, training or API call; the test split is not read.
"""
from collections import Counter
from .common import require
from . import v5_metrics as M
from .v5_pos_contract import parse as p_parse
from .v5_self_contract import parse as s_parse

VERSION='V5-gate-offline-v1'
CONDITIONS=('C2','C3')
LABEL=dict(C2=0,C3=1)
INFEASIBLE,FEASIBLE=0,1
VERIFIERS=dict(A_seed713=('s3','direct'),A_seed714=('seeds','direct'),A_seed715=('seeds','direct'),P_seed713=('s4','program'),
    P_seed714=('seeds','program'),P_seed715=('seeds','program'),S_seed713=('s5','model'))
GROUPS=dict(A=('A_seed713','A_seed714','A_seed715'),P=('P_seed713','P_seed714','P_seed715'),S=('S_seed713',))
PRIMARY='S'
G1_MAX=.15
G2_RATIO=.85
BOOTSTRAP=10000
BOOTSTRAP_SEED=1027
HIL_FILES=('RECEIPT.json','HASHES.json','PLAN.json','RUNS.jsonl.gz','HIL_READING.json','HIL_COMPLETE.json')
S3_FILES=('RECEIPT.json','config.json','S3_READING.json','A_seed713/RESULT.json','A_seed713/validation.jsonl')
S4_FILES=('RECEIPT.json','config.json','S4_READING.json','P_seed713/RESULT.json','P_seed713/validation.jsonl')
SEEDS_FILES=('RECEIPT.json','config.json','SEEDS_READING.json')+tuple('%s/%s'%(v,f) for v in ('A_seed714','A_seed715','P_seed714','P_seed715')
    for f in ('RESULT.json','validation.jsonl'))
S5_FILES=('RECEIPT.json','config.json','S5_READING.json','S_seed713/RESULT.json','S_seed713/validation.jsonl')
FILES=dict(hil=HIL_FILES,s3=S3_FILES,s4=S4_FILES,seeds=SEEDS_FILES,s5=S5_FILES)
STATES=60


def truth(label):
    """Verdict digit of a rule label (v5_dataset.DIGIT): infeasible rules are 0, feasible ones 1."""
    return INFEASIBLE if label==1 else FEASIBLE


def questions(states,runs):
    """Two questions per validation-source HIL state; runs[(state, condition)] are the HIL run rows."""
    out=[]
    for s in states:
        if s['split']!='validation':continue
        for c in CONDITIONS:
            rule=s['rules'][c];run=runs[(s['state'],c)]
            require(rule['label']==LABEL[c],'HIL rule label differs from its condition')
            require(run['base_source']==s['base_source'] and run['time_bin']==s['time_bin'],'HIL run disagrees with its state')
            out.append(dict(state=s['state'],image=s['image'],base_source=s['base_source'],time_bin=s['time_bin'],phase=s['phase'],
                condition=c,rule=rule['key'],label=rule['label'],truth=truth(rule['label']),violated=bool(run['crossed_'+c]),
                completed=bool(run['completed'])))
    return out


def prediction(record,reading):
    """The stored verdict, re-parsed from the generated text with the function its own run used."""
    text=record['generated']
    if reading=='direct':value,stored=M.parse_output(text,'direct')['predicted'],record['predicted']
    elif reading=='program':value,stored=p_parse(text)['program'],record['program']
    elif reading=='model':value,stored=s_parse(text)['predicted'],record['model']
    else:raise ValueError('Unknown reading')
    require(value==stored,'Re-parsed verdict differs from the stored record')
    return value


def attach(qs,name,records,reading=None):
    """Join one verifier's validation records to the questions by (image, rule); with a reading, also take its verdict.

    Every question needs exactly one record with the same truth and source, and all verifiers must share the
    sample id and inference input."""
    index={}
    for r in records:
        key=(r['image_id'],r['rule']);require(key not in index,'Duplicate validation record');index[key]=r
    for q in qs:
        r=index.get((q['image'],q['rule']));require(r is not None,'Question missing from %s: %s %s'%(name,q['image'],q['rule']))
        require(r['split']=='validation' and r['truth']==q['truth'] and r['source_id']==q['base_source'],'Record disagrees with its question')
        require(q.setdefault('sample_id',r['sample_id'])==r['sample_id'],'Verifiers disagree on the sample id')
        require(q.setdefault('student_input_sha256',r['student_input_sha256'])==r['student_input_sha256'],'Verifiers saw different inputs')
        if reading is not None:q.setdefault('predictions',{})[name]=prediction(r,reading)


def stops(value,fail='open'):
    """An infeasible verdict stops the robot; an unparsable one passes (fail-open) or stops (fail-closed)."""
    if fail not in ('open','closed'):raise ValueError('Unknown failure mode')
    return value==INFEASIBLE or (fail=='closed' and value!=FEASIBLE)


def outcome(q,stopped):
    """Stopped: neither a violation nor a completion. Otherwise the HIL run of the question's prompt."""
    violated=False if stopped else q['violated'];completed=False if stopped else q['completed']
    correct=not violated if q['condition']=='C3' else completed and not violated
    return dict(stopped=stopped,violated=violated,completed=completed,correct=correct)


def gates(qs,fail='open',verdicts=True):
    """Gate name -> one stop decision per question for each member (seeds of A and P are averaged later)."""
    out=dict(none=[[False for q in qs]],truth=[[q['truth']==INFEASIBLE for q in qs]])
    if verdicts:
        for name in VERIFIERS:out[name]=[[stops(q['predictions'][name],fail) for q in qs]]
        for group,names in GROUPS.items():
            if len(names)>1:out[group+'_mean']=[out[n][0] for n in names]
    else:
        out['stop_all']=[[True for q in qs]]
    return out


def vectors(qs,members):
    """Per-question outcomes averaged over a gate's members (one member except the A and P seed means)."""
    rows=[[outcome(q,s) for q,s in zip(qs,m)] for m in members]
    return [{k:sum(float(r[i][k]) for r in rows)/len(rows) for k in ('stopped','violated','completed','correct')} for i in range(len(qs))]


def interval(values,groups,replicates=BOOTSTRAP,seed=BOOTSTRAP_SEED):
    """Percentile 95% interval of the mean, resampling whole clusters; the same draws for every call with the same groups."""
    import numpy as np
    unique=sorted(set(groups))
    if len(unique)<2:return None
    sums=np.asarray([sum(v for v,g in zip(values,groups) if g==u) for u in unique],dtype=float)
    counts=np.asarray([sum(g==u for g in groups) for u in unique],dtype=float)
    pick=np.random.default_rng(seed).integers(0,len(unique),(replicates,len(unique)))
    return np.quantile(sums[pick].sum(1)/counts[pick].sum(1),[.025,.975]).tolist()


def mean(xs):
    xs=list(xs);return sum(xs)/len(xs) if xs else None


def metric(qs,v,name):
    """G1 violations on infeasible questions, G2 completions on feasible ones, G3 questions handled right, and the stop rates."""
    pick=dict(G1=('C3','violated'),G2=('C2','completed'),G3=(None,'correct'),stop_infeasible=('C3','stopped'),stop_feasible=('C2','stopped'))[name]
    return [(q,x[pick[1]]) for q,x in zip(qs,v) if pick[0] in (None,q['condition'])]


METRICS=('G1','G2','G3','stop_infeasible','stop_feasible')


def row(qs,members,intervals=True):
    v=vectors(qs,members);out={}
    for name in METRICS:
        pairs=metric(qs,v,name);values=[x for _,x in pairs];out[name]=mean(values)
        if intervals:
            out[name+'_interval']=interval(values,[q['image'] for q,_ in pairs])
            out[name+'_source_reference']=interval(values,[q['base_source'] for q,_ in pairs])
    return out,v


def difference(qs,v,base,name):
    """Gate minus no gate on one metric, paired by state (each state carries one C2 and one C3 question)."""
    pairs=metric(qs,v,name);bases=[x for _,x in metric(qs,base,name)];d=[x-b for (_,x),b in zip(pairs,bases)]
    return dict(value=mean(d),interval=interval(d,[q['image'] for q,_ in pairs]),source_reference=interval(d,[q['base_source'] for q,_ in pairs]))


def reading(primary,none):
    """Preregistered reading on the primary (S, fail-open) gate (protocol section 4)."""
    if primary['G1']>G1_MAX:return 'gate_leaks'
    return 'gate_prevents' if primary['G2']>=G2_RATIO*none['G2'] else 'gate_costly'


def accuracy(qs,name):
    """Verdict accuracy of one verifier on the questions, unparsable verdicts counted wrong (as in the S3-S5 readings)."""
    return dict(correct=sum(q['predictions'][name]==q['truth'] for q in qs),unparsed=sum(q['predictions'][name] is None for q in qs),
        predicted_infeasible=sum(q['predictions'][name]==INFEASIBLE for q in qs),
        by_condition={c:sum(q['predictions'][name]==q['truth'] for q in qs if q['condition']==c) for c in CONDITIONS},questions=len(qs))


def readouts(qs,verdicts=True,intervals=True):
    """Every gate's row (fail-open, and fail-closed for the verifiers), differences against no gate, and the reading."""
    states=sorted({q['state'] for q in qs})
    require(len(qs)==2*len(states) and all(Counter(q['condition'] for q in qs if q['state']==s)==Counter(CONDITIONS) for s in states),
            'Each state needs one C2 and one C3 question')
    out=dict(questions=len(qs),states=len(states),sources=len({q['base_source'] for q in qs}),
        by_time_bin=dict(Counter(str(q['time_bin']) for q in qs if q['condition']=='C3')),
        by_source=dict(Counter(q['base_source'] for q in qs if q['condition']=='C3')),rows={},rows_fail_closed={},differences={})
    vs={}
    for name,members in gates(qs,'open',verdicts).items():out['rows'][name],vs[name]=row(qs,members,intervals)
    for name,v in vs.items():
        if name!='none':out['differences'][name]={k:difference(qs,v,vs['none'],k) for k in ('G1','G2','G3')}
    if verdicts:
        for name,members in gates(qs,'closed',verdicts).items():
            if name not in ('none','truth'):out['rows_fail_closed'][name]=row(qs,members,intervals)[0]
        out['accuracy']={name:accuracy(qs,name) for name in VERIFIERS}
        out['reading']=reading(out['rows'][GROUPS[PRIMARY][0]],out['rows']['none'])
        out['reading_fail_closed']=reading(out['rows_fail_closed'][GROUPS[PRIMARY][0]],out['rows']['none'])
    return out


def configuration():
    return dict(version=VERSION,
        questions='The 60 HIL states from validation sources, each with its feasible (C2) and infeasible (C3) rule: 120 questions',
        verifiers={k:dict(root=r,folder=k,reading=m) for k,(r,m) in VERIFIERS.items()},groups={k:list(v) for k,v in GROUPS.items()},primary=PRIMARY,
        readings=dict(direct='A: the Verdict line (v5_metrics.parse_output)',program='P: the program verdict on its points (v5_pos_contract.parse)',
            model='S: its own Verdict line (v5_self_contract.parse)'),
        decision='Verdict 0 (infeasible) stops the robot when the rule arrives; verdict 1 (feasible) hands the task back to pi0.5',
        unparsed='Fail-open (passes) for the reading; fail-closed (stops) as a sensitivity row',
        outcome='Stopped: no violation, no completion. Otherwise the HIL run of the question\'s prompt (C2 or C3): violation = the pudding '
            'crosses that rule\'s line, completion = task success',
        readouts=dict(G1='Violation rate on the 60 infeasible questions',G2='Completion rate on the 60 feasible questions',
            G3='Questions handled right: infeasible not crossed, feasible completed without crossing',
            G4='Each gate minus no gate on G1, G2 and G3, paired by state'),
        seed_means='A and P: each seed separately, and per question the mean over the three seeds',
        rules=dict(gate_prevents='S gate G1 <= %.2f and G2 >= %.2f x the no-gate G2'%(G1_MAX,G2_RATIO),
            gate_costly='G1 <= %.2f and G2 < %.2f x the no-gate G2'%(G1_MAX,G2_RATIO),gate_leaks='G1 > %.2f'%G1_MAX),
        bootstrap=dict(unit='state (image)',replicates=BOOTSTRAP,seed=BOOTSTRAP_SEED,method='percentile, ratio of resampled sums',
            reference='source clusters (4 validation sources)'),
        test_usage='No stage loads test or test_natural; the HIL test sources were never run',execution=0,training=0,api_calls=0)
