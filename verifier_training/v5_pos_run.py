"""V5 S4 on Quest (docs/generative_verifier/V5_S4_PROTOCOL.md): arm P writes positions, a program decides.

Stages: preflight (CPU; projected points, target encoding, inputs equal to S3's, S3 orders and subsets), train P SEED
(GPU; S3 schedule and selection with the program verdict), report (CPU; P against the S3 A records and the reading).
No stage loads test or test_natural.
"""
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from collections import Counter
import numpy as np
import torch
from peft import get_peft_model_state_dict,set_peft_model_state_dict
from .common import read,write,sha,require,MODEL_ID,UPSTREAM
from .loss import gradient_probe
from .verdict_diagnostic import tensor_digest
from .fd1000_run import vision_inputs,encode,input_digest,processor_for,adapter_model,update,run_generation
from .v5_contract import validate
from .v5_pos_contract import configuration,image_points,facts,answer,parse,s4_reading,FORCED,S3_FILES
from . import v5_metrics as M

TESTS=['verifier_training.tests.test_v5_pos_contract','verifier_training.tests.test_v5_contract','verifier_training.tests.test_v5_metrics']
SPLITS=('train','validation','validation_natural')


def context(root):
    receipt=read(root/'RECEIPT.json');config=read(root/'config.json')
    require(config==configuration() and sha(root/'config.json')==receipt['configuration_sha256'],'Fixed S4 configuration changed')
    require(os.environ['CB_CODE_COMMIT']==receipt['source_commit'],'Frozen S4 source differs')
    s3=Path(receipt['s3_root']);require(set(receipt['s3_files'])==set(S3_FILES),'S3 evidence list differs')
    for name,digest in receipt['s3_files'].items():require(sha(s3/name)==digest,'S3 evidence changed: '+name)
    require(read(s3/'config.json')==config['s3_configuration'],'S3 configuration differs')
    dataset=Path(receipt['dataset']);assets=Path(receipt['assets'])
    rows=validate(dataset,include_test=False);require({r['split'] for r in rows}==set(SPLITS),'Unexpected splits loaded')
    require(sha(assets/'STAGED.json')==receipt['assets_stage_sha256'],'Model staging manifest changed')
    staged=read(assets/'STAGED.json');require(staged['model_id']==MODEL_ID,'Unexpected base model')
    for name,digest in staged['files'].items():require(sha(assets/'model'/name)==digest,'Staged base model changed')
    return receipt,config,dataset,assets,s3,rows,staged


def split(rows,name):
    return [r for r in rows if r['split']==name]


def by_image(rows):
    groups={}
    for r in rows:groups.setdefault(r['metadata']['image_id'],[]).append(r)
    return groups


def p_item(processor,row,dataset,config,vision,points,verify=False):
    f,_=facts(row,points)
    return encode(processor,row,dataset,'facts',config['s3_configuration']['max_length'],vision,verify=verify,
        answer=answer(f,row['target']['digit']),keys=('separated',),forced=FORCED)


def preflight(root):
    receipt,config,dataset,assets,s3,rows,staged=context(root);out=root/'preflight';out.mkdir()
    subprocess.run([sys.executable,'-m','unittest',*TESTS],check=True)
    s3_hashes={r['sample_id']:r['input_sha256'] for r in read(s3/'preflight/ENCODINGS.json')}
    processor=processor_for(assets);records=[];points={}
    for image,members in sorted(by_image(rows).items()):
        points[image]=image_points(dataset,image);vision=vision_inputs(processor,members[0],dataset)
        for row in members:
            it=p_item(processor,row,dataset,config,vision,points[image]['points'],verify=True)
            require(input_digest(it)==s3_hashes[row['sample_id']],'P input differs from the S3 inputs')
            require(it['labels'][:,:it['n']].eq(-100).all() and int(it['vm'].sum())==1,'Prompt supervised or decision mask invalid')
            records.append(dict(sample_id=row['sample_id'],split=row['split'],image_id=image,input_sha256=s3_hashes[row['sample_id']],
                prompt_tokens=it['n'],tokens=it['fields']['input_ids'].shape[1],answer=it['answer']))
    s3_orders=read(s3/'ORDERS.json');orders={seed:{'P':arms['B']} for seed,arms in s3_orders.items()}
    require(all({k for b in o['P'] for s,k in b}=={'facts'} for o in orders.values()),'S3 B order has another target kind')
    write(root/'POINTS.json',points);write(root/'ORDERS.json',orders);write(out/'ENCODINGS.json',records)
    longest=max(r['tokens'] for r in records)
    write(out/'COMPLETE.json',dict(passed=True,rows=len(rows),by_split=dict(Counter(r['split'] for r in rows)),images=len(points),
        program_agrees_with_every_label=True,inputs_equal_s3=True,native_processor_equal=True,longest_sequence=longest,
        no_truncation=longest<=config['s3_configuration']['max_length'],source_commit=receipt['source_commit'],
        points_sha256=sha(root/'POINTS.json'),orders_sha256=sha(root/'ORDERS.json'),encodings_sha256=sha(out/'ENCODINGS.json'),
        config_sha256=sha(root/'config.json'),test_rows_loaded=0,optimizer_steps=0,model_predictions=0,
        packages={k:metadata.version(k) for k in ('torch','transformers','peft','accelerate','numpy','pillow')},slurm_job=os.environ['SLURM_JOB_ID']))


def gate(root,receipt,s3):
    done=read(root/'preflight/COMPLETE.json');require(done['passed'] and done['source_commit']==receipt['source_commit'],'S4 preflight gate failed')
    for name,key in (('POINTS.json','points_sha256'),('ORDERS.json','orders_sha256'),('preflight/ENCODINGS.json','encodings_sha256'),('config.json','config_sha256')):
        require(sha(root/name)==done[key],'Preflight evidence changed: '+name)
    hashes={r['sample_id']:r['input_sha256'] for r in read(root/'preflight/ENCODINGS.json')}
    return hashes,read(root/'POINTS.json'),read(s3/'SUBSETS.json'),read(s3/'LOOKUP.json')['predictions']


def views(records):
    """The same generations read two ways: by the program on the points (primary) and by the written verdict."""
    program=[dict(r,predicted=r['program'],p_feasible=None,reading='program') for r in records]
    model=[dict(r,predicted=r['model'],reading='model') for r in records]
    return program,model


def point_errors(records):
    out={}
    for field in ('a','b','target','goal'):
        d=[float(np.hypot(*(np.asarray(r['points'][field])-np.asarray(r['truth_points'][field])))) for r in records if r['points']]
        out[field]=dict(parsed=len(d),median_px=float(np.median(d)) if d else None,within_15px=sum(x<=15 for x in d)/len(records),
            within_30px=sum(x<=30 for x in d)/len(records))
    return out


def summary(records):
    program,model=views(records)
    result=dict(program=M.score(program,'direct'),model=M.score(model,'direct'),point_errors=point_errors(records),
        program_model_agreement=sum(r['program'] is not None and r['program']==r['model'] for r in records)/len(records),
        written_separated_matches_program=sum(r['separated'] is not None and r['program'] is not None and (r['separated']=='yes')==(r['program']==0)
            for r in records)/len(records),
        mean_generation_seconds=sum(r['generation_seconds'] for r in records)/len(records),
        mean_generated_tokens=sum(r['generated_token_count'] for r in records)/len(records))
    return result


def evaluate(model,processor,rows,items,config,hashes,lookup,points,stream):
    model.eval();records=[];stream.touch(exist_ok=False)
    for row in rows:
        (output,),seconds=run_generation(model,processor,items[row['sample_id']],'facts',None,config['maximum_new_tokens'])
        p=parse(output['generated']);_,truth=facts(row,points[row['metadata']['image_id']]['points'])
        record=M.row_record(row,'p',None,None,hashes[row['sample_id']],lookup.get(row['sample_id']))
        record.update(generated=output['generated'],generated_token_count=output['generated_token_count'],stopped=output['stopped'],p_feasible=output['p_feasible'],
            model=p['predicted'],program=p['program'],points=p['points'],separated=p['separated'],format_valid=p['format_valid'],
            truth_points=truth,generation_seconds=seconds,gold_rationale_prefix_used=False,kind='points')
        records.append(record)
        with stream.open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
    return records


def train(root,arm,seed):
    receipt,config,dataset,assets,s3,rows,staged=context(root);hashes,points,subsets,lookup=gate(root,receipt,s3)
    require(arm=='P' and seed in config['seeds'],'Unplanned run')
    train_p(root,arm,seed,receipt,config,dataset,assets,rows,staged,hashes,points,subsets,lookup)


def train_p(root,arm,seed,receipt,config,dataset,assets,rows,staged,hashes,points,subsets,lookup,provenance=None,
            scope='Validation-only pilot of arm P; one seed; the test split is not read'):
    """Arm P for one seed with the S4 settings in config (S4 above; the V5 seed campaign, v5_seeds_run).

    The caller has checked its campaign, the planned seed and the preflight gate; root holds ORDERS.json with a P order
    for the seed. provenance adds campaign fields to PROVENANCE.json."""
    require(arm=='P','Unplanned run')
    out=root/f'{arm}_seed{seed}';out.mkdir();require(torch.cuda.is_available(),'Quest GPU required');base=config['s3_configuration']
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);torch.backends.cuda.matmul.allow_tf32=True
    processor=processor_for(assets);model,targets=adapter_model(assets,base,seed)
    params=[(n,v) for n,v in model.named_parameters() if v.requires_grad]
    require(params and all('lora_' in n and 'visual' not in n and 'lm_head' not in n for n,v in params),'Unexpected trainable weights')
    visual=[(n,v) for n,v in model.named_parameters() if 'visual' in n];visual_hash=tensor_digest(visual)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.config.use_cache=False
    train_rows=split(rows,'train');validation=split(rows,'validation');natural=split(rows,'validation_natural')
    chosen=set(subsets['selection']);selection=[r for r in validation if r['sample_id'] in chosen]
    fitted=set(subsets['training_fit']);fit_rows=[r for r in train_rows if r['sample_id'] in fitted]
    require(len(selection)==base['selection']['rows'] and len(fit_rows)==base['training_fit']['rows'],'Subset rows missing')
    items={}
    for image,members in by_image(train_rows+validation+natural).items():
        vision=vision_inputs(processor,members[0],dataset)
        for row in members:items[row['sample_id']]=p_item(processor,row,dataset,config,vision,points[image]['points'])
    require(all(input_digest(v)==hashes[s] for s,v in items.items()),'Real training input differs from preflight')
    keyed={(s,'facts'):v for s,v in items.items()}
    order=read(root/'ORDERS.json')[str(seed)][arm];require(len(order)==base['optimizer_updates'],'Incomplete update schedule')
    require(Counter(s for b in order for s,k in b)==Counter({r['sample_id']:1 for r in train_rows}),'The epoch repeats or omits training rows')
    optimizer=torch.optim.AdamW([v for _,v in params],lr=base['learning_rate'],weight_decay=base['weight_decay'])
    lookups={**lookup['validation'],**lookup['validation_natural'],**lookup['train']}
    write(out/'PROVENANCE.json',dict(arm=arm,seed=seed,source_commit=receipt['source_commit'],model_id=MODEL_ID,model_revision=staged['model_revision'],
        upstream_commit=UPSTREAM,config=config,lora_targets=targets,trainable_parameters=sum(v.numel() for n,v in params),frozen_visual_sha256=visual_hash,
        preflight_sha256=sha(root/'preflight/COMPLETE.json'),splits_loaded=sorted({r['split'] for r in rows}),
        slurm={k:v for k,v in os.environ.items() if k in ('SLURM_JOB_ID','SLURM_JOB_NODELIST','SLURM_JOB_PARTITION','SLURM_JOB_ACCOUNT')},
        gpu=torch.cuda.get_device_name(),same_inference_inputs_as_s3=True,gold_rationale_prefix_at_selection=False,**(provenance or {})))
    started=time.monotonic();checkpoints=[];states={};probes={}

    def probe(kind,values):
        if probes:return
        probes[kind]=gradient_probe(model,values,'B');write(out/'GRADIENT_PROBE.json',probes)

    for step,batch in enumerate(order,1):
        totals=update(model,optimizer,keyed,batch,params,base,probe)
        with (out/'curve.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,seconds=time.monotonic()-started,**totals))+'\n')
        if step%10==0:print(arm,seed,step,json.dumps(totals),flush=True)
        if step in base['evaluation_updates']:
            measured=summary(evaluate(model,processor,selection,items,config,hashes,lookups,points,out/f'selection_step_{step}.jsonl'))
            value=measured['program']['verdict_accuracy'];checkpoints.append(dict(step=step,selection_value=value,scores=measured))
            states[step]={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
            model.save_pretrained(out/f'checkpoint_{step}');write(out/'SELECTION_SCORES.json',checkpoints)
            print(arm,seed,'step',step,'program',value,'model',measured['model']['verdict_accuracy'],flush=True)
    best=max(checkpoints,key=lambda c:(c['selection_value'],-c['step']));selected=best['step']
    set_peft_model_state_dict(model,states[selected]);restored=get_peft_model_state_dict(model)
    require(set(restored)==set(states[selected]) and all(torch.equal(v.detach().cpu(),states[selected][k]) for k,v in restored.items()),
            'Selected adapter was not restored exactly')
    require(tensor_digest(visual)==visual_hash,'Frozen vision/projector changed')
    model.save_pretrained(out/'selected_adapter');processor.save_pretrained(out/'selected_adapter')
    scores={}
    for name,part in (('validation',validation),('validation_natural',natural),('training_fit',fit_rows)):
        scores[name]=summary(evaluate(model,processor,part,items,config,hashes,lookups,points,out/f'{name}.jsonl'))
        print(arm,seed,name,'program',scores[name]['program']['verdict_accuracy'],'model',scores[name]['model']['verdict_accuracy'],flush=True)
    write(out/'RESULT.json',dict(passed=True,arm=arm,seed=seed,steps=len(order),selected_step=selected,selection_value=best['selection_value'],
        checkpoint_values={c['step']:c['selection_value'] for c in checkpoints},
        R1_program_accuracy=scores['validation']['program']['verdict_accuracy'],R1_model_accuracy=scores['validation']['model']['verdict_accuracy'],
        R2_twin_image_all_correct=scores['validation']['program']['twin_image_all_correct'],
        R4_natural_program_accuracy=scores['validation_natural']['program']['verdict_accuracy'],
        R5_training_fit_program_accuracy=scores['training_fit']['program']['verdict_accuracy'],scores=scores,
        adapter_sha256=sha(out/'selected_adapter/adapter_model.safetensors'),elapsed_seconds=time.monotonic()-started,
        peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,test_rows_loaded=0,test_evaluations=0,api_calls=0,scope=scope))


def report(root):
    """CPU: P (program reading) against the S3 A records on the full validation split, and the reading."""
    receipt,config,dataset,assets,s3,rows,staged=context(root);gate(root,receipt,s3)
    done=read(root/'P_seed713/RESULT.json');require(done['passed'] and done['test_rows_loaded']==0,'P run incomplete')
    records=[json.loads(x) for x in (root/'P_seed713/validation.jsonl').read_text().splitlines()];program,model=views(records)
    a=[json.loads(x) for x in (s3/'A_seed713/validation.jsonl').read_text().splitlines()]
    paired=M.paired_difference(a,program);paired_model=M.paired_difference(a,model)
    a_acc=read(s3/'A_seed713/RESULT.json')['R1_validation_accuracy'];p_acc=done['R1_program_accuracy']
    require(abs(paired['control_accuracy']-a_acc)<1e-12 and abs(paired['treatment_accuracy']-p_acc)<1e-12,'Records differ from RESULT files')
    reading=s4_reading(p_acc,a_acc,paired['image_bootstrap_95pct'])
    zero=read(root/'zero_shot/SUMMARY.json') if (root/'zero_shot/SUMMARY.json').exists() else None
    write(root/'S4_READING.json',dict(passed=True,reading=reading,P_program=p_acc,P_model=done['R1_model_accuracy'],A=a_acc,
        B_s3=read(s3/'B_seed713/RESULT.json')['R1_validation_accuracy'],delta_P_program_minus_A=paired,delta_P_model_minus_A=paired_model,
        P=dict(selected_step=done['selected_step'],checkpoint_values=done['checkpoint_values'],R2=done['R2_twin_image_all_correct'],
            R4=done['R4_natural_program_accuracy'],R5=done['R5_training_fit_program_accuracy'],scores=done['scores'],adapter_sha256=done['adapter_sha256']),
        zero_shot=zero,rules=config['rules'],source_commit=receipt['source_commit'],test_rows_loaded=0,api_calls=0,
        slurm_job=os.environ.get('SLURM_JOB_ID')))
    print(reading,json.dumps(dict(P=p_acc,A=a_acc,delta=paired['difference'],interval=paired['image_bootstrap_95pct'])),flush=True)


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
