"""V5 S5 on Quest (docs/generative_verifier/V5_S5_PROTOCOL.md): arm S writes positions, the computation and its own verdict.

Stages: preflight (CPU; inputs equal to S3's, true points equal to S4's, every target chain checked, the S4 P order),
train S SEED (GPU; v5_pos_run.train_p with S's item, evaluation, summary and own-verdict selection), report (CPU; S
against the S3 A and S4 P records, step checks and the reading). No stage loads test or test_natural.
"""
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from collections import Counter
import numpy as np
from .common import read,write,sha,require,MODEL_ID
from .fd1000_run import vision_inputs,encode,input_digest,processor_for,run_generation
from .v5_contract import validate
from .v5_pos_contract import image_points,facts as p_facts,FORCED
from .v5_pos_run import gate,views,point_errors,train_p,by_image,split
from .v5_self_contract import configuration,facts,parse,s5_reading,STEPS,S3_FILES,S4_FILES,ARM
from . import v5_metrics as M

TESTS=['verifier_training.tests.test_v5_self_contract','verifier_training.tests.test_v5_pos_contract','verifier_training.tests.test_v5_metrics']
SPLITS=('train','validation','validation_natural')
SCOPE='Validation-only pilot of arm S (positions, computation, own verdict); one seed; the test split is not read'


def context(root):
    receipt=read(root/'RECEIPT.json');config=read(root/'config.json')
    require(config==configuration() and sha(root/'config.json')==receipt['configuration_sha256'],'Fixed S5 configuration changed')
    require(os.environ['CB_CODE_COMMIT']==receipt['source_commit'],'Frozen S5 source differs')
    s3=Path(receipt['s3_root']);s4=Path(receipt['s4_root'])
    for base,key,names in ((s3,'s3_files',S3_FILES),(s4,'s4_files',S4_FILES)):
        require(set(receipt[key])==set(names),'Evidence list differs: '+key)
        for name,digest in receipt[key].items():require(sha(base/name)==digest,'Evidence changed: %s %s'%(key,name))
    require(read(s3/'config.json')==config['s3_configuration'] and read(s4/'config.json')==config['s4_configuration'],'S3 or S4 configuration differs')
    dataset=Path(receipt['dataset']);assets=Path(receipt['assets'])
    rows=validate(dataset,include_test=False)
    # A smoke receipt (never written by the launcher) names a few development images; the real run reads every one.
    if 'smoke' in receipt:keep=set(receipt['smoke']['images']);rows=[r for r in rows if r['metadata']['image_id'] in keep]
    require({r['split'] for r in rows}==set(SPLITS),'Unexpected splits loaded')
    require(sha(assets/'STAGED.json')==receipt['assets_stage_sha256'],'Model staging manifest changed')
    staged=read(assets/'STAGED.json');require(staged['model_id']==MODEL_ID,'Unexpected base model')
    for name,digest in staged['files'].items():require(sha(assets/'model'/name)==digest,'Staged base model changed')
    return receipt,config,dataset,assets,s3,s4,rows,staged


def s_item(processor,row,dataset,config,vision,points,verify=False):
    _,_,text=facts(row,points)
    return encode(processor,row,dataset,'facts',config['s3_configuration']['max_length'],vision,verify=verify,answer=text,keys=('separated',),forced=FORCED)


def smoke_orders(order,train_ids,size):
    """The S4 P order restricted to the smoke training rows, in the same sequence, regrouped into batches of size."""
    keep=[pair for batch in order for pair in batch if pair[0] in train_ids];return [keep[i:i+size] for i in range(0,len(keep),size)]


def smoke_configuration(config,receipt,rows,subsets):
    """The in-memory settings of a smoke: its own number of updates and evaluation points, and the subsets it holds."""
    base=dict(config['s3_configuration']);order=read(Path(receipt['run_root'])/'ORDERS.json')[str(config['seeds'][0])][ARM]
    ids={r['sample_id'] for r in rows};base.update(optimizer_updates=len(order),evaluation_updates=list(receipt['smoke']['evaluation_updates']),
        selection=dict(base['selection'],rows=len(ids&set(subsets['selection']))),training_fit=dict(base['training_fit'],rows=len(ids&set(subsets['training_fit']))))
    return dict(config,s3_configuration=base)


def preflight(root):
    receipt,config,dataset,assets,s3,s4,rows,staged=context(root);out=root/'preflight';out.mkdir()
    subprocess.run([sys.executable,'-m','unittest',*TESTS],check=True)
    s3_hashes={r['sample_id']:r['input_sha256'] for r in read(s3/'preflight/ENCODINGS.json')}
    processor=processor_for(assets);records=[];points={}
    for image,members in sorted(by_image(rows).items()):
        points[image]=image_points(dataset,image);vision=vision_inputs(processor,members[0],dataset)
        for row in members:
            it=s_item(processor,row,dataset,config,vision,points[image]['points'],verify=True)
            require(input_digest(it)==s3_hashes[row['sample_id']],'S input differs from the S3 inputs')
            require(it['labels'][:,:it['n']].eq(-100).all() and int(it['vm'].sum())==1,'Prompt supervised or decision mask invalid')
            p=parse(it['answer']);digit=int(row['target']['digit'])
            require(p['format_valid'] and p['all_steps'] and p['predicted']==p['program']==digit,'Target chain fails its own check')
            records.append(dict(sample_id=row['sample_id'],split=row['split'],image_id=image,input_sha256=s3_hashes[row['sample_id']],
                prompt_tokens=it['n'],tokens=it['fields']['input_ids'].shape[1],answer_tokens=it['fields']['input_ids'].shape[1]-it['n'],answer=it['answer']))
    s4_points=read(s4/'POINTS.json')
    if 'smoke' in receipt:s4_points={k:v for k,v in s4_points.items() if k in points}
    require(json.loads(json.dumps(points))==s4_points,'Projected points differ from the S4 points')
    seed=str(config['seeds'][0]);order=read(s4/'ORDERS.json')[seed]['P']
    if 'smoke' in receipt:
        order=smoke_orders(order,{r['sample_id'] for r in rows if r['split']=='train'},config['s3_configuration']['effective_batch_size'])
    write(root/'POINTS.json',points);write(root/'ORDERS.json',{seed:{ARM:order}});write(out/'ENCODINGS.json',records)
    longest=max(r['tokens'] for r in records);answers=[r['answer_tokens'] for r in records]
    write(out/'COMPLETE.json',dict(passed=True,rows=len(rows),by_split=dict(Counter(r['split'] for r in rows)),images=len(points),
        inputs_equal_s3=True,native_processor_equal=True,points_equal_s4=True,every_chain_checked=True,longest_sequence=longest,
        no_truncation=longest<=config['s3_configuration']['max_length'],answer_tokens=dict(mean=float(np.mean(answers)),max=max(answers)),
        order_is_s4_p_order='smoke' not in receipt,updates=len(order),source_commit=receipt['source_commit'],
        points_sha256=sha(root/'POINTS.json'),orders_sha256=sha(root/'ORDERS.json'),encodings_sha256=sha(out/'ENCODINGS.json'),
        config_sha256=sha(root/'config.json'),smoke='smoke' in receipt,test_rows_loaded=0,optimizer_steps=0,model_predictions=0,
        packages={k:metadata.version(k) for k in ('torch','transformers','peft','accelerate','numpy','pillow')},slurm_job=os.environ['SLURM_JOB_ID']))


def evaluate_s(model,processor,rows,items,config,hashes,lookup,points,stream):
    model.eval();records=[];stream.touch(exist_ok=False)
    for row in rows:
        (output,),seconds=run_generation(model,processor,items[row['sample_id']],'facts',None,config['maximum_new_tokens'])
        p=parse(output['generated']);_,truth=p_facts(row,points[row['metadata']['image_id']]['points'])
        record=M.row_record(row,'s',None,None,hashes[row['sample_id']],lookup.get(row['sample_id']))
        record.update(generated=output['generated'],generated_token_count=output['generated_token_count'],stopped=output['stopped'],p_feasible=output['p_feasible'],
            model=p['predicted'],program=p['program'],points=p['points'],separated=p['separated'],format_valid=p['format_valid'],written=p['written'],
            steps=p['steps'],all_steps=p['all_steps'],execution_agreement=p['execution_agreement'],truth_points=truth,generation_seconds=seconds,
            gold_rationale_prefix_used=False,kind='self')
        records.append(record)
        with stream.open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
    return records


def steps_summary(records):
    n=len(records)
    return dict({s:sum(r['steps'][s] for r in records)/n for s in STEPS},all_steps=sum(r['all_steps'] for r in records)/n,
        execution_agreement=sum(r['execution_agreement'] for r in records)/n,format_valid=sum(r['format_valid'] for r in records)/n)


def summary_s(records):
    program,model=views(records)
    return dict(program=M.score(program,'direct'),model=M.score(model,'direct'),point_errors=point_errors(records),steps=steps_summary(records),
        mean_generation_seconds=sum(r['generation_seconds'] for r in records)/len(records),
        mean_generated_tokens=sum(r['generated_token_count'] for r in records)/len(records))


def train(root,arm,seed):
    receipt,config,dataset,assets,s3,s4,rows,staged=context(root);hashes,points,subsets,lookup=gate(root,receipt,s3)
    require(arm==ARM and seed in config['seeds'],'Unplanned run')
    settings=smoke_configuration(config,receipt,rows,subsets) if 'smoke' in receipt else config
    train_p(root,arm,seed,receipt,settings,dataset,assets,rows,staged,hashes,points,subsets,lookup,
        provenance=dict(campaign=config['version'],smoke='smoke' in receipt),scope=SCOPE,
        hooks=dict(arm=ARM,item=s_item,evaluate=evaluate_s,summary=summary_s,selection='model'))


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def decompose(records):
    """R6: wrong own verdicts split by whether the program on the same points is right (computation) or wrong (points)."""
    own=lambda r:r['model']==r['truth'];prog=lambda r:r['program']==r['truth']
    return dict(rows=len(records),own_wrong=sum(not own(r) for r in records),points_wrong=sum(not own(r) and not prog(r) for r in records),
        computation_wrong=sum(not own(r) and prog(r) for r in records),points_wrong_verdict_right=sum(own(r) and not prog(r) for r in records))


def report(root):
    """CPU: S's own verdict against the S3 A and S4 P records on the validation split, step checks and the reading."""
    receipt,config,dataset,assets,s3,s4,rows,staged=context(root);gate(root,receipt,s3)
    seed=config['seeds'][0];out=root/f'{ARM}_seed{seed}';done=read(out/'RESULT.json')
    require(done['passed'] and done['arm']==ARM and done['test_rows_loaded']==0,'S run incomplete')
    records=lines(out/'validation.jsonl');program,model=views(records);ids={r['sample_id'] for r in records}
    a=[r for r in lines(s3/'A_seed713/validation.jsonl') if r['sample_id'] in ids]
    p_records=[r for r in lines(s4/'P_seed713/validation.jsonl') if r['sample_id'] in ids];p_program,p_model=views(p_records)
    s_minus_a=M.paired_difference(a,model);s=s_minus_a['treatment_accuracy']
    require(abs(s-done['R1_model_accuracy'])<1e-12,'Records differ from the RESULT file')
    reading=s5_reading(s,s_minus_a['image_bootstrap_95pct'])
    write(root/'S5_READING.json',dict(passed=True,reading=reading,R1=dict(S_own=s,A=s_minus_a['control_accuracy'],
            P_program=M.paired_difference(p_program,model)['control_accuracy'],P_own=M.paired_difference(p_model,model)['control_accuracy'],
            S_program_on_own_points=M.score(program,'direct',bootstrap=False)['verdict_accuracy']),
        delta_S_minus_A=s_minus_a,delta_S_minus_P_program=M.paired_difference(p_program,model),delta_S_minus_P_own=M.paired_difference(p_model,model),
        R2_twin_image_all_correct=M.score(model,'direct',bootstrap=False)['twin_image_all_correct'],
        R3=dict(steps=steps_summary(records),point_errors=point_errors(records)),
        R4_validation_natural=done['scores']['validation_natural']['model']['verdict_accuracy'],
        R5_training_fit=done['scores']['training_fit']['model']['verdict_accuracy'],
        R6=decompose(records),selected_step=done['selected_step'],checkpoint_values=done['checkpoint_values'],scores=done['scores'],
        adapter_sha256=done['adapter_sha256'],rules=config['rules'],smoke='smoke' in receipt,source_commit=receipt['source_commit'],
        test_rows_loaded=0,api_calls=0,slurm_job=os.environ.get('SLURM_JOB_ID')))
    print(reading,json.dumps(dict(S=s,A=s_minus_a['control_accuracy'],delta=s_minus_a['difference'],interval=s_minus_a['image_bootstrap_95pct'])),flush=True)


if __name__=='__main__':
    stage=sys.argv[1];root=Path(sys.argv[2]);label=stage+('_'+'_'.join(sys.argv[3:]) if len(sys.argv)>3 else '')
    try:
        if stage=='preflight':preflight(root)
        elif stage=='train':train(root,sys.argv[3],int(sys.argv[4]))
        elif stage=='report':report(root)
        else:raise ValueError('Unknown stage')
    except BaseException as error:
        write(root/('STOP_'+label+'.json'),dict(error=type(error).__name__,message=str(error),slurm_job=os.environ.get('SLURM_JOB_ID')))
        raise
