"""V5 offline gate on Quest (docs/generative_verifier/V5_GATE_OFFLINE_PROTOCOL.md): CPU only, existing records only.

Stages: preflight (unit tests; every evidence file's hash; the HIL readouts recomputed from its run rows; the 120
questions joined to all seven verifiers' validation records by key, without reading a verdict), report (the verdicts,
each gate's row, the differences against no gate and the reading). A smoke receipt (never written by the launcher)
keeps the report away from the verdicts: only the no-gate, truth and stop-all rows.
"""
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
from collections import Counter
from .common import read,write,sha,require
from .v5_gate_contract import configuration,questions,attach,readouts,FILES,VERIFIERS,STATES,CONDITIONS

TESTS=['verifier_training.tests.test_v5_gate_contract','verifier_training.tests.test_v5_self_contract',
    'verifier_training.tests.test_v5_pos_contract','verifier_training.tests.test_v5_metrics']


def context(root):
    receipt=read(root/'RECEIPT.json');config=read(root/'config.json')
    require(config==configuration() and sha(root/'config.json')==receipt['configuration_sha256'],'Fixed gate configuration changed')
    require(os.environ['CB_CODE_COMMIT']==receipt['source_commit'],'Frozen gate source differs')
    roots={k:Path(receipt[k+'_root']) for k in FILES}
    for key,names in FILES.items():
        require(set(receipt[key+'_files'])==set(names),'Evidence list differs: '+key)
        for name,digest in receipt[key+'_files'].items():require(sha(roots[key]/name)==digest,'Evidence changed: %s %s'%(key,name))
    return receipt,config,roots


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def hil(root):
    """HIL states and run rows, after its own hash list and a recomputation of R1, R2 and R4 from the rows."""
    for name,digest in read(root/'HASHES.json').items():require(sha(root/name)==digest,'HIL output changed: '+name)
    require(read(root/'HIL_COMPLETE.json')['passed'] and read(root/'RECEIPT.json')['mode']=='formal','Needs the finished formal HIL batch')
    states=read(root/'PLAN.json')['states']
    with gzip.open(root/'RUNS.jsonl.gz','rt') as stream:rows=[json.loads(x) for x in stream]
    runs={(r['state'],r['condition']):r for r in rows}
    require(len(runs)==len(rows)==4*len(states),'HIL runs incomplete or duplicated')
    rates=read(root/'HIL_READING.json')['rates'];mean=lambda c,k:sum(runs[(s['state'],c)][k] for s in states)/len(states)
    for key,c,k in (('R1','C3','crossed_C3'),('R2','C0','completed'),('R4','C2','completed')):
        require(abs(mean(c,k)-rates[key])<1e-12,'HIL run rows differ from its reading: '+key)
    return states,runs


def build(roots,reading=False):
    """The 120 questions joined to every verifier's validation records; verdicts only when reading is set."""
    states,runs=hil(roots['hil']);qs=questions(states,runs)
    require(len({q['state'] for q in qs})==STATES and len(qs)==2*STATES,'Expected %d validation-source HIL states'%STATES)
    for name,(key,how) in VERIFIERS.items():attach(qs,name,lines(roots[key]/name/'validation.jsonl'),how if reading else None)
    return qs


def preflight(root):
    receipt,config,roots=context(root);out=root/'preflight';out.mkdir()
    subprocess.run([sys.executable,'-m','unittest',*TESTS],check=True)
    qs=build(roots);write(out/'QUESTIONS.json',qs)
    write(out/'COMPLETE.json',dict(passed=True,questions=len(qs),states=len({q['state'] for q in qs}),
        sources=dict(Counter(q['base_source'] for q in qs if q['condition']=='C3')),by_condition=dict(Counter(q['condition'] for q in qs)),
        by_time_bin=dict(Counter(str(q['time_bin']) for q in qs if q['condition']=='C3')),verifiers=list(VERIFIERS),coverage='every question in every record file',
        hil_rates_recomputed=True,verdicts_read=0,questions_sha256=sha(out/'QUESTIONS.json'),smoke='smoke' in receipt,
        source_commit=receipt['source_commit'],test_rows_loaded=0,api_calls=0,slurm_job=os.environ['SLURM_JOB_ID']))


def check_s(roots):
    """S's own-verdict accuracy on the whole validation file equals its RESULT and S5 reading (no prior hash of the file exists)."""
    records=lines(roots['s5']/'S_seed713/validation.jsonl');acc=sum(r['model']==r['truth'] for r in records)/len(records)
    require(abs(acc-read(roots['s5']/'S_seed713/RESULT.json')['R1_model_accuracy'])<1e-12,'S records differ from the RESULT file')
    require(abs(acc-read(roots['s5']/'S5_READING.json')['R1']['S_own'])<1e-12,'S records differ from the S5 reading')
    return dict(rows=len(records),own_verdict_accuracy=acc)


def report(root):
    receipt,config,roots=context(root);done=read(root/'preflight/COMPLETE.json')
    require(done['passed'] and sha(root/'preflight/QUESTIONS.json')==done['questions_sha256'],'Preflight incomplete')
    smoke='smoke' in receipt;qs=build(roots,reading=not smoke)
    require([{k:v for k,v in q.items() if k!='predictions'} for q in qs]==read(root/'preflight/QUESTIONS.json'),'Questions differ from the preflight')
    s_check=None if smoke else check_s(roots);result=readouts(qs,verdicts=not smoke)
    with (root/'GATE_QUESTIONS.jsonl').open('w') as f:
        for q in qs:f.write(json.dumps(q,allow_nan=False)+'\n')
    result.update(passed=True,smoke=smoke,s_records_check=s_check,primary_gate='S_seed713',conditions=list(CONDITIONS),
        source_commit=receipt['source_commit'],questions_sha256=sha(root/'GATE_QUESTIONS.jsonl'),test_rows_loaded=0,executions=0,api_calls=0,
        slurm_job=os.environ.get('SLURM_JOB_ID'))
    write(root/'GATE_READING.json',result)
    show={k:{m:v[m] for m in ('G1','G2','G3')} for k,v in result['rows'].items()}
    print(result.get('reading'),json.dumps(show),flush=True)


if __name__=='__main__':
    stage=sys.argv[1];root=Path(sys.argv[2])
    try:
        if stage=='preflight':preflight(root)
        elif stage=='report':report(root)
        else:raise ValueError('Unknown stage')
    except BaseException as error:
        write(root/('STOP_'+stage+'.json'),dict(error=type(error).__name__,message=str(error),slurm_job=os.environ.get('SLURM_JOB_ID')))
        raise
