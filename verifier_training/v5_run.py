"""V5 S3 A-versus-B pilot on Quest (docs/generative_verifier/V5_S3_PROTOCOL.md).

Stages: preflight (CPU token audit, fixed order, subsets and lookup), train ARM SEED (train + validation
splits only), zero-shot-3b (base model on the selection subset), report (CPU, the preregistered reading).
No stage here loads test or test_natural.
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
from .common import read,write,sha,require,MODEL_ID,UPSTREAM,student_messages
from .loss import gradient_probe
from .verdict_diagnostic import tensor_digest
from .fd1000_run import vision_inputs,prompt_item,encode,input_digest,processor_for,base_model,adapter_model,update,run_generation
from .v5_contract import (campaign,preflight_gate,assistant_target,readings,schedule,check_schedule,selection_subset,fit_subset,
    lookup_predictions,zero_shot_messages,s3_reading,FACT_KEYS,KINDS,FORCED_PREFIX,TRAIN_LOOKUP)
from . import v5_metrics as M

TESTS=['verifier_training.tests.test_v5_contract','verifier_training.tests.test_v5_metrics','verifier_training.tests.test_v5_encoding']
EVALUATION_SPLITS=('validation','validation_natural')


def context(root):
    receipt,config,dataset,rows=campaign(root);assets=Path(receipt['assets'])
    require(sha(assets/'STAGED.json')==receipt['assets_stage_sha256'],'Model staging manifest changed')
    staged=read(assets/'STAGED.json');require(staged['model_id']==MODEL_ID,'Unexpected base model')
    for name,digest in staged['files'].items():require(sha(assets/'model'/name)==digest,'Staged base model changed')
    return receipt,config,dataset,assets,rows,staged


def split(rows,name):
    return [r for r in rows if r['split']==name]


def by_image(rows):
    groups={}
    for r in rows:groups.setdefault(r['metadata']['image_id'],[]).append(r)
    return groups


def item(processor,row,dataset,kind,config,vision,verify=False):
    return encode(processor,row,dataset,kind,config['max_length'],vision,verify=verify,answer=assistant_target(row,kind),
        keys=FACT_KEYS if kind=='facts' else (),forced=FORCED_PREFIX[kind])


def lookup_check(measured,dataset):
    """The rule x progress lookup must reproduce the S2 readout (train 0.760, validation 0.500 by construction)."""
    expected=read(dataset/'S2_SUMMARY.json')
    require(all(abs(measured[s]-expected[s]['lookup_from_train']['label'])<1e-9 for s in measured),'Lookup differs from the S2 readout')
    require(abs(measured['train']-TRAIN_LOOKUP)<5e-4,'Training lookup bound differs from the protocol')


def preflight(root):
    receipt,config,dataset,assets,rows,staged=context(root);out=root/'preflight';out.mkdir()
    subprocess.run([sys.executable,'-m','unittest',*TESTS],check=True)
    processor=processor_for(assets);records=[]
    for word in ('0','1'):require(len(processor.tokenizer.encode(word,add_special_tokens=False))==1,'Verdict digit is not one token: '+word)
    for image,members in sorted(by_image(rows).items()):
        vision=vision_inputs(processor,members[0],dataset)
        for row in members:
            items={k:item(processor,row,dataset,k,config,vision,verify=True) for k in KINDS}
            digests={k:input_digest(v) for k,v in items.items()};require(len(set(digests.values()))==1,'Target kinds change the inference input')
            for it in items.values():
                require(it['labels'][:,:it['n']].eq(-100).all() and int(it['vm'].sum())==1,'Prompt supervised or decision mask invalid')
            f=items['facts'];zero={m:prompt_item(processor,zero_shot_messages(row,m),vision)['n'] for m in ('direct','facts')}
            require(max(zero.values())+config['maximum_new_tokens']['facts']<=config['max_length'],'Zero-shot prompt too long')
            records.append(dict(sample_id=row['sample_id'],split=row['split'],image_id=image,input_sha256=digests['direct'],prompt_tokens=f['n'],
                tokens_direct=items['direct']['fields']['input_ids'].shape[1],tokens_facts=f['fields']['input_ids'].shape[1],
                key_value_tokens={k:int(f['detail']['key_'+k].sum()) for k in FACT_KEYS},template_tokens=int(f['detail']['template'].sum()),
                zero_shot_prompt_tokens=zero,image_grid=vision['image_grid_thw'].tolist()))
    train=split(rows,'train');validation=split(rows,'validation');train_ids=[r['sample_id'] for r in train]
    orders=schedule(train_ids,config)
    for seed,arms in orders.items():
        for arm,order in arms.items():check_schedule(order,train_ids,arm,config)
        require(len({tuple(s for b in o for s,k in b) for o in arms.values()})==1,'Arms of one seed differ in sample order')
    n=config['selection']['images'];selection=selection_subset(validation,n);fit=fit_subset(train,config['training_fit']['rows'])
    chosen={r['sample_id']:r for r in validation if r['sample_id'] in set(selection)}
    require(len(chosen)==config['selection']['rows'] and Counter(r['target']['digit'] for r in chosen.values())=={'0':n,'1':n},'Selection subset unbalanced')
    require(all(sorted(r['target']['digit'] for r in rs)==['0','1'] for rs in by_image(chosen.values()).values()),'Selection images need one row of each answer')
    lookup={s:lookup_predictions(train,split(rows,s)) for s in ('train',*EVALUATION_SPLITS)}
    measured={s:sum(lookup[s][r['sample_id']]==int(r['target']['digit']) for r in split(rows,s))/len(split(rows,s)) for s in lookup}
    lookup_check(measured,dataset)
    write(root/'ORDERS.json',orders);write(root/'SUBSETS.json',dict(selection=selection,training_fit=fit))
    write(root/'LOOKUP.json',dict(predictions=lookup,accuracy=measured));write(out/'ENCODINGS.json',records)
    longest=max(max(r['tokens_direct'],r['tokens_facts']) for r in records)
    write(out/'COMPLETE.json',dict(passed=True,rows=len(rows),by_split=dict(Counter(r['split'] for r in rows)),images=len(by_image(rows)),
        identical_inference_inputs_across_kinds=True,native_processor_equal=True,longest_sequence=longest,max_length=config['max_length'],
        no_truncation=longest<=config['max_length'],single_binary_verdict_token=True,forced_prefix_boundaries_checked=True,
        key_value_tokens_per_row=dict(Counter(sum(r['key_value_tokens'].values()) for r in records)),lookup_accuracy=measured,
        source_commit=receipt['source_commit'],dataset_hashes=config['dataset_hashes'],
        encodings_sha256=sha(out/'ENCODINGS.json'),orders_sha256=sha(root/'ORDERS.json'),subsets_sha256=sha(root/'SUBSETS.json'),
        lookup_sha256=sha(root/'LOOKUP.json'),config_sha256=sha(root/'config.json'),test_rows_loaded=0,
        optimizer_steps=0,model_predictions=0,packages={k:metadata.version(k) for k in ('torch','transformers','peft','accelerate','numpy','pillow')},
        slurm_job=os.environ['SLURM_JOB_ID']))


def generate(model,processor,it,kind,prefix,config):
    """One greedy continuation, regraded with the V5 parser."""
    (output,),seconds=run_generation(model,processor,it,kind,prefix,config['maximum_new_tokens'][kind])
    parsed=M.parse_output(output['generated'],kind);require(parsed['predicted']==output['predicted'],'Verdict parsers disagree')
    output.update(parsed);return output,seconds


def evaluate(model,processor,rows,items,reading,kind,config,hashes,lookup,stream):
    model.eval();records=[];stream.touch(exist_ok=False)
    for row in rows:
        output,seconds=generate(model,processor,items[(row['sample_id'],kind)],kind,None,config)
        record=M.row_record(row,reading,kind,None,hashes[row['sample_id']],lookup.get(row['sample_id']))
        record.update(output,generation_seconds=seconds,gold_rationale_prefix_used=False);records.append(record)
        with stream.open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
    return records


def train(root,arm,seed):
    receipt,config,dataset,assets,rows,staged=context(root);hashes,subsets,lookup=preflight_gate(root,receipt)
    require(arm in config['arms'] and seed in config['seeds'],'Unplanned run')
    out=root/f'{arm}_seed{seed}';out.mkdir();require(torch.cuda.is_available(),'Quest GPU required')
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);torch.backends.cuda.matmul.allow_tf32=True
    processor=processor_for(assets);model,targets=adapter_model(assets,config,seed)
    params=[(n,v) for n,v in model.named_parameters() if v.requires_grad]
    require(params and all('lora_' in n and 'visual' not in n and 'lm_head' not in n for n,v in params),'Unexpected trainable weights')
    visual=[(n,v) for n,v in model.named_parameters() if 'visual' in n]
    require(visual and any('merger' in n for n,v in visual),'Frozen visual projector missing');visual_hash=tensor_digest(visual)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.config.use_cache=False
    (reading,(kind,prefix)),=readings(arm).items();require(prefix is None,'Selection reading must be free generation')
    train_rows=split(rows,'train');validation=split(rows,'validation');natural=split(rows,'validation_natural')
    chosen=set(subsets['selection']);selection=[r for r in validation if r['sample_id'] in chosen]
    fitted=set(subsets['training_fit']);fit_rows=[r for r in train_rows if r['sample_id'] in fitted]
    require(len(selection)==config['selection']['rows'] and len(fit_rows)==config['training_fit']['rows'],'Subset rows missing')
    items={}
    for image,members in by_image(train_rows+validation+natural).items():
        vision=vision_inputs(processor,members[0],dataset)
        for row in members:items[(row['sample_id'],kind)]=item(processor,row,dataset,kind,config,vision)
    require(all(input_digest(v)==hashes[s] for (s,k),v in items.items()),'Real training input differs from preflight')
    order=read(root/'ORDERS.json')[str(seed)][arm];check_schedule(order,[r['sample_id'] for r in train_rows],arm,config)
    optimizer=torch.optim.AdamW([v for _,v in params],lr=config['learning_rate'],weight_decay=config['weight_decay'])
    lookups={**lookup['validation'],**lookup['validation_natural'],**lookup['train']}
    write(out/'PROVENANCE.json',dict(arm=arm,seed=seed,source_commit=receipt['source_commit'],dataset_hashes=config['dataset_hashes'],
        model_id=MODEL_ID,model_revision=staged['model_revision'],upstream_commit=UPSTREAM,config=config,lora_targets=targets,
        trainable_parameters=sum(v.numel() for n,v in params),frozen_visual_sha256=visual_hash,preflight_sha256=sha(root/'preflight/COMPLETE.json'),
        orders_sha256=sha(root/'ORDERS.json'),subsets_sha256=sha(root/'SUBSETS.json'),splits_loaded=sorted({r['split'] for r in rows}),
        slurm={k:v for k,v in os.environ.items() if k in ('SLURM_JOB_ID','SLURM_JOB_NODELIST','SLURM_JOB_PARTITION','SLURM_JOB_ACCOUNT')},
        gpu=torch.cuda.get_device_name(),same_inference_inputs=True,gold_rationale_prefix_at_selection=False))
    started=time.monotonic();checkpoints=[];states={};probes={}

    def probe(kind_,values):
        """Gradient probe on the first item; for B also the key facts alone."""
        if kind_ in probes:return
        probes[kind_]=gradient_probe(model,values,'B')
        if kind_=='facts':
            grads=torch.autograd.grad(values['key_values'],[v for n,v in params],retain_graph=True,allow_unused=True)
            norm=sum(float(g.detach().float().square().sum()) for g in grads if g is not None)**.5
            require(np.isfinite(norm) and norm>0,'Key facts do not train language adapters')
            probes[kind_]['key_values']=dict(tokens=values['key_values_tokens'],gradient_norm=norm)
        write(out/'GRADIENT_PROBE.json',probes)

    for step,batch in enumerate(order,1):
        totals=update(model,optimizer,items,batch,params,config,probe)
        record=dict(step=step,seconds=time.monotonic()-started,**totals)
        with (out/'curve.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if step%10==0:print(arm,seed,json.dumps(record),flush=True)
        if step in config['evaluation_updates']:
            measured=M.summarize(evaluate(model,processor,selection,items,reading,kind,config,hashes,lookups,out/f'selection_step_{step}.jsonl'))
            value=measured[reading]['verdict_accuracy'];checkpoints.append(dict(step=step,selection_value=value,scores=measured))
            states[step]={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
            model.save_pretrained(out/f'checkpoint_{step}');write(out/'SELECTION_SCORES.json',checkpoints)
            print(arm,seed,'step',step,'selection',value,flush=True)
    best=max(checkpoints,key=lambda c:(c['selection_value'],-c['step']));selected=best['step']
    set_peft_model_state_dict(model,states[selected]);restored=get_peft_model_state_dict(model)
    require(set(restored)==set(states[selected]) and all(torch.equal(v.detach().cpu(),states[selected][k]) for k,v in restored.items()),
            'Selected adapter was not restored exactly')
    require(tensor_digest(visual)==visual_hash and all(not v.requires_grad and v.grad is None for n,v in visual),'Frozen vision/projector changed')
    model.save_pretrained(out/'selected_adapter');processor.save_pretrained(out/'selected_adapter')
    scores={}
    for name,part in (('validation',validation),('validation_natural',natural),('training_fit',fit_rows)):
        scores[name]=M.summarize(evaluate(model,processor,part,items,reading,kind,config,hashes,lookups,out/f'{name}.jsonl'))[reading]
        print(arm,seed,name,scores[name]['verdict_accuracy'],flush=True)
    write(out/'RESULT.json',dict(passed=True,arm=arm,seed=seed,reading=reading,steps=len(order),selected_step=selected,
        selection_value=best['selection_value'],checkpoint_values={c['step']:c['selection_value'] for c in checkpoints},
        R1_validation_accuracy=scores['validation']['verdict_accuracy'],R2_twin_image_all_correct=scores['validation']['twin_image_all_correct'],
        R4_validation_natural_accuracy=scores['validation_natural']['verdict_accuracy'],R5_training_fit_accuracy=scores['training_fit']['verdict_accuracy'],
        scores=scores,adapter_sha256=sha(out/'selected_adapter/adapter_model.safetensors'),elapsed_seconds=time.monotonic()-started,
        frozen_vision_and_projector_unchanged=True,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
        test_rows_loaded=0,test_evaluations=0,api_calls=0,scope='Validation-only pilot; one seed; the test split is not read'))


def zero_shot_3b(root):
    """R6 (a): the staged 3B base model on the selection subset, direct and facts prompts."""
    receipt,config,dataset,assets,rows,staged=context(root);hashes,subsets,lookup=preflight_gate(root,receipt)
    out=root/'zero_shot_3b';out.mkdir();require(torch.cuda.is_available(),'Quest GPU required')
    chosen=set(subsets['selection']);selection=[r for r in split(rows,'validation') if r['sample_id'] in chosen]
    torch.manual_seed(1014);processor=processor_for(assets);model=base_model(assets);model.eval();started=time.monotonic()
    records=[];stream=out/'records.jsonl';stream.touch(exist_ok=False)
    for image,members in sorted(by_image(selection).items()):
        vision=vision_inputs(processor,members[0],dataset)
        for row in members:
            for mode,kind,prefix in (('direct','direct','Verdict: '),('facts','facts',None)):
                output,seconds=generate(model,processor,prompt_item(processor,zero_shot_messages(row,mode),vision),kind,prefix,config)
                record=M.row_record(row,'zero_shot_'+mode,kind,prefix,hashes[row['sample_id']],lookup['validation'][row['sample_id']])
                record.update(output,generation_seconds=seconds,gold_rationale_prefix_used=False);records.append(record)
                with stream.open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
    write(out/'SUMMARY.json',dict(passed=True,scores=M.summarize(records),model_id=MODEL_ID,model_revision=staged['model_revision'],adapter=None,
        rows=len(selection),split='validation selection subset',test_rows_loaded=0,elapsed_seconds=time.monotonic()-started,
        peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,api_calls=0,source_commit=receipt['source_commit'],slurm_job=os.environ['SLURM_JOB_ID']))


def report(root):
    """CPU: R1-R6 and the preregistered reading from the finished runs."""
    receipt,config,dataset,assets,rows,staged=context(root);preflight_gate(root,receipt);result={};records={}
    for arm in config['arms']:
        label=f'{arm}_seed{config["seeds"][0]}';done=read(root/label/'RESULT.json');require(done['passed'] and done['test_rows_loaded']==0,'Run incomplete')
        records[arm]=[json.loads(x) for x in (root/label/'validation.jsonl').read_text().splitlines()]
        result[arm]={k:done[k] for k in ('selected_step','checkpoint_values','R1_validation_accuracy','R2_twin_image_all_correct',
            'R4_validation_natural_accuracy','R5_training_fit_accuracy')}
        result[arm].update(validation=done['scores']['validation'],validation_natural=done['scores']['validation_natural'],
            training_fit=done['scores']['training_fit'],result_sha256=sha(root/label/'RESULT.json'),adapter_sha256=done['adapter_sha256'])
    require(M.summarize(records['A'])['direct_free']['verdict_accuracy']==result['A']['R1_validation_accuracy'],'A records differ from RESULT')
    paired=M.paired_difference(records['A'],records['B'])
    a,b=result['A']['R1_validation_accuracy'],result['B']['R1_validation_accuracy']
    reading=s3_reading(a,b,paired['image_bootstrap_95pct'],{arm:result[arm]['R5_training_fit_accuracy'] for arm in config['arms']})
    zero={}
    for name,path in (('qwen2.5-vl-3b','zero_shot_3b/SUMMARY.json'),('qwen3-vl-32b','zero_shot_32b/SUMMARY.json')):
        if (root/path).exists():zero[name]=dict(read(root/path)['scores'],summary_sha256=sha(root/path))
    write(root/'S3_READING.json',dict(passed=True,reading=reading,R1=dict(A=a,B=b),delta_B_minus_A=paired,runs=result,
        R6_zero_shot=zero,zero_shot_included=sorted(zero),lookup_accuracy=read(root/'preflight/COMPLETE.json')['lookup_accuracy'],
        rules='b_grounds: B>=0.65, delta>=0.10, image interval low>0; both_ground: A>=0.65, delta<0.10; '
            'both_shortcut: A<=0.60, B<=0.60, max R5>=0.70; otherwise unclear',
        source_commit=receipt['source_commit'],test_rows_loaded=0,api_calls=0,slurm_job=os.environ.get('SLURM_JOB_ID')))
    print(reading,json.dumps(dict(A=a,B=b,delta=paired['difference'],interval=paired['image_bootstrap_95pct'])),flush=True)


if __name__=='__main__':
    stage=sys.argv[1];root=Path(sys.argv[2]);label=stage+('_'+'_'.join(sys.argv[3:]) if len(sys.argv)>3 else '')
    try:
        if stage=='preflight':preflight(root)
        elif stage=='train':train(root,sys.argv[3],int(sys.argv[4]))
        elif stage=='zero-shot-3b':zero_shot_3b(root)
        elif stage=='report':report(root)
        else:raise ValueError('Unknown stage')
    except BaseException as error:
        write(root/('STOP_'+label+'.json'),dict(error=type(error).__name__,message=str(error),slurm_job=os.environ.get('SLURM_JOB_ID')))
        raise
