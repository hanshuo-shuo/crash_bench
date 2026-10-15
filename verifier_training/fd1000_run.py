"""Accepted-data-only native Qwen LoRA comparison with free generation."""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
from collections import Counter
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
from .common import read,write,sha,require,MODEL_ID,UPSTREAM,student_messages
from .binary_verifier import encode as encode_b,losses,LABEL_TO_DIGIT
from .loss import gradient_probe
from .verdict_diagnostic import tensor_digest
from .engineering_contract import epoch_orders
from .fd1000_contract import configuration,validate
from .fd1000_metrics import parse,rationale_grade,score,paired_comparison


def encode(processor,row,dataset,arm,maximum):
    if arm=='B':return encode_b(processor,row,dataset,max_length=maximum)
    images=[]
    for spec in row['student']['images']:
        path=dataset/spec['path'];require(sha(path)==spec['sha256'],'Image changed')
        with Image.open(path) as im:images.append(im.convert('RGB'))
    messages=student_messages(row,'A');answer='Verdict: '+LABEL_TO_DIGIT[row['target']['verdict']]
    prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    full=processor.apply_chat_template(messages+[dict(role='assistant',content=answer)],tokenize=False,add_generation_prompt=False)
    require(full.startswith(prompt+answer),'Direct target template changed')
    fields=processor(text=[full],images=images,return_tensors='pt',padding=False)
    prefix=processor(text=[prompt],images=images,return_tensors='pt',padding=False)['input_ids'][0]
    ids=fields['input_ids'][0];n=len(prefix)
    require(len(ids)<=maximum and torch.equal(ids[:n],prefix),'Direct prompt boundary or length invalid')
    eos=processor.tokenizer.convert_tokens_to_ids('<|im_end|>');ends=(ids[n:]==eos).nonzero()
    require(len(ends)==1,'Direct assistant terminator ambiguous');end=n+int(ends[0]);digit=end-1
    digit_ids=processor.tokenizer.encode(answer[-1],add_special_tokens=False)
    require(len(digit_ids)==1 and int(ids[digit])==digit_ids[0],'Direct verdict is not a single binary token')
    before=processor(text=[prompt+answer[:-1]],images=images,return_tensors='pt',padding=False)['input_ids'][0]
    require(torch.equal(ids[:digit],before),'Direct binary decision boundary differs')
    labels=fields['input_ids'].clone();labels[:,:n]=-100;labels[:,end+1:]=-100;labels[~fields['attention_mask'].bool()]=-100
    vm=torch.zeros_like(labels,dtype=torch.bool);vm[:,digit]=True;rm=labels.ne(-100)&~vm
    detail=dict(rationale_text=torch.zeros_like(vm),format=rm&~labels.eq(eos),eos=labels.eq(eos))
    return fields,labels,rm,vm,n,detail


def input_digest(item):
    h=hashlib.sha256();fields=item[0];n=item[4]
    for key in ('input_ids','attention_mask','image_grid_thw','pixel_values'):
        tensor=fields[key][:,:n] if key in ('input_ids','attention_mask') else fields[key]
        h.update(key.encode());h.update(str(tuple(tensor.shape)).encode());h.update(str(tensor.dtype).encode())
        h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def context(root):
    receipt=read(root/'RECEIPT.json');config=read(root/'config.json')
    require(config==configuration() and sha(root/'config.json')==receipt['configuration_sha256'],'Fixed training configuration changed')
    require(os.environ['CB_CODE_COMMIT']==receipt['source_commit'],'Frozen training source differs')
    dataset=Path(receipt['dataset']);assets=Path(receipt['assets'])
    require(sha(dataset/'ACCEPTANCE.json')==receipt['dataset_acceptance_sha256'],'Dataset acceptance changed')
    rows,acceptance=validate(dataset)
    require(sha(assets/'STAGED.json')==receipt['assets_stage_sha256'],'Model staging manifest changed')
    staged=read(assets/'STAGED.json');require(staged['model_id']==MODEL_ID,'Unexpected base model')
    for name,digest in staged['files'].items():require(sha(assets/'model'/name)==digest,'Staged base model changed')
    return receipt,config,dataset,assets,rows,staged


def processor_for(assets):
    pixels=configuration()['image_pixels']
    return AutoProcessor.from_pretrained(assets/'model',local_files_only=True,min_pixels=pixels,max_pixels=pixels)


def preflight(root):
    receipt,config,dataset,assets,rows,staged=context(root);out=root/'preflight';out.mkdir()
    subprocess.run([sys.executable,'-m','unittest','verifier_training.tests.test_binary_verifier',
        'verifier_training.tests.test_fd1000_contract','verifier_training.tests.test_fd1000_metrics'],check=True)
    processor=processor_for(assets);records=[]
    for row in rows:
        a=encode(processor,row,dataset,'A',config['max_length']);b=encode(processor,row,dataset,'B',config['max_length'])
        digest=input_digest(a);require(digest==input_digest(b),'A/B full inference inputs differ')
        for item in (a,b):
            require(item[1][:,:item[4]].eq(-100).all(),'Prompt tokens supervised')
            require(int(item[3].sum())==1,'Binary decision mask invalid')
            require(torch.equal(item[2]|item[3],item[1].ne(-100)) and not (item[2]&item[3]).any(),'Assistant token partition invalid')
        require(b[5]['rationale_text'].any(),'No narrative supervision')
        records.append(dict(sample_id=row['sample_id'],split=row['split'],input_sha256=digest,
            prompt_tokens=a[4],tokens_A=a[0]['input_ids'].shape[1],tokens_B=b[0]['input_ids'].shape[1],
            decision_tokens=1,rationale_tokens=int(b[5]['rationale_text'].sum())))
    train=[r for r in rows if r['split']=='train'];orders=epoch_orders(len(train),config['seed'],config['epochs'],config['effective_batch_size'])
    write(root/'TRAINING_ORDER.json',[[train[i]['sample_id'] for i in order] for order in orders])
    write(out/'ENCODINGS.json',records)
    write(out/'COMPLETE.json',dict(passed=True,rows=len(rows),identical_inference_inputs=True,
        all_real_prompt_image_padding_tokens_masked=True,one_binary_token_each=True,
        source_commit=receipt['source_commit'],dataset_sha256=acceptance_digest(dataset),
        encodings_sha256=sha(out/'ENCODINGS.json'),order_sha256=sha(root/'TRAINING_ORDER.json'),
        config_sha256=sha(root/'config.json'),optimizer_steps=0,model_predictions=0,
        packages={k:metadata.version(k) for k in ('torch','transformers','peft','accelerate','numpy','pillow')},slurm_job=os.environ['SLURM_JOB_ID']))


def acceptance_digest(dataset):return read(dataset/'ACCEPTANCE.json')['dataset_sha256']


def compare(root):
    receipt,config,dataset,assets,rows,staged=context(root)
    path=root/'COMPARISON.json';require(not path.exists(),'No silent replacement of the paired comparison')
    expected={r['sample_id']:r for r in rows if r['split']=='test'}
    hashes={r['sample_id']:r['input_sha256'] for r in read(root/'preflight/ENCODINGS.json')}
    predictions={};results={};artifacts={}
    for arm in ('A','B'):
        out=root/arm;result=read(out/'RESULT.json');provenance=read(out/'PROVENANCE.json')
        require(result['passed'] and result['test_evaluations']==1,'Both selected models need one complete test evaluation')
        require(provenance['source_commit']==receipt['source_commit'] and provenance['config']==config and
            provenance['dataset_sha256']==acceptance_digest(dataset),'A/B source, data or configuration differs')
        records=read(out/'TEST_GENERATIONS.json')
        stream=[json.loads(line) for line in (out/'TEST_GENERATIONS.jsonl').read_text().splitlines()]
        require(records==stream and len(records)==len(expected) and {r['sample_id'] for r in records}==set(expected),
            'Final predictions do not match the complete raw test-generation stream')
        for r in records:
            row=expected[r['sample_id']];m=row['metadata']
            require(r['truth']==int(row['target']['verdict']=='feasible') and
                r['student_input_sha256']==hashes[r['sample_id']] and r['unit_id']==m['unit_id'] and
                r['source_group']==m['ancestry_component'] and r['source_id']==m['source_id'] and
                r['assignment']==row['evidence']['assignment'] and r['protected_boundary']==row['evidence']['protected_boundary'],
                'Test predictions are not anchored to the accepted source-split records')
        predictions[arm]=records;results[arm]=result
        artifacts[arm]={name:sha(out/name) for name in ('RESULT.json','PROVENANCE.json','TEST_GENERATIONS.json','TEST_GENERATIONS.jsonl')}
    write(path,dict(passed=True,source_commit=receipt['source_commit'],dataset_sha256=acceptance_digest(dataset),
        comparison=paired_comparison(predictions['A'],predictions['B']),arms=results,artifacts=artifacts,
        new_model_predictions=0,api_calls=0,slurm_job=os.environ['SLURM_JOB_ID']))


def generate(model,processor,items,rows,arm,config,hashes,audit_stream):
    model.eval();records=[];digit_ids=[processor.tokenizer.encode(s,add_special_tokens=False)[0] for s in ('0','1')]
    # Preserve every completed prediction if a later example or job fails.
    # Exclusive creation prevents a retry from silently replacing test evidence.
    audit_stream.touch(exist_ok=False)
    with torch.no_grad():
        for row,item in zip(rows,items):
            n=item[4];require(input_digest(item)==hashes[row['sample_id']],'Evaluation input changed')
            prompt={k:(v[:,:n] if k in ('input_ids','attention_mask') else v).to('cuda') for k,v in item[0].items()}
            torch.cuda.synchronize();generation_start=time.monotonic()
            result=model.generate(**prompt,max_new_tokens=config['maximum_new_tokens'][arm],do_sample=False,use_cache=True,
                repetition_penalty=config['repetition_penalty'],pad_token_id=processor.tokenizer.pad_token_id,
                return_dict_in_generate=True,output_scores=True,output_logits=True)
            torch.cuda.synchronize();generation_seconds=time.monotonic()-generation_start
            tokens=result.sequences[0,n:];text=processor.tokenizer.decode(tokens,skip_special_tokens=True).strip()
            pred,formatted=parse(text,arm);probability=None;decision=None
            if pred is not None:
                prefix=text.rsplit('Verdict: ',1)[0]+'Verdict: '
                positions=[i for i in range(len(tokens)) if processor.tokenizer.decode(tokens[:i],skip_special_tokens=True).lstrip()==prefix]
                if len(positions)==1 and int(tokens[positions[0]]) in digit_ids:
                    at=positions[0];raw=result.logits[at][0].float();probability=float(raw[digit_ids].softmax(0)[1])
                    decision=dict(generated_token_index=at,generated_prefix_token_ids=tokens[:at].tolist(),raw_binary_logits=raw[digit_ids].tolist())
            record=dict(sample_id=row['sample_id'],unit_id=row['metadata']['unit_id'],source_group=row['metadata']['ancestry_component'],
                source_id=row['metadata']['source_id'],family=row['metadata']['family'],assignment=row['evidence']['assignment'],
                protected_boundary=row['evidence']['protected_boundary'],truth=int(row['target']['verdict']=='feasible'),
                generated=text,predicted=pred,format_valid=formatted,p_feasible=probability,native_decision=decision,
                generated_token_ids=tokens.tolist(),generated_token_count=len(tokens),generation_seconds=generation_seconds,
                student_input_sha256=hashes[row['sample_id']],gold_rationale_prefix_used=False)
            if arm=='B':record['rationale_assessment']=rationale_grade(text,row)
            records.append(record)
            with audit_stream.open('a') as stream:
                stream.write(json.dumps(record,allow_nan=False)+'\n')
    return records


def train(root,arm):
    receipt,config,dataset,assets,rows,staged=context(root);gate=read(root/'preflight/COMPLETE.json')
    require(gate['passed'] and gate['source_commit']==receipt['source_commit'] and gate['dataset_sha256']==acceptance_digest(dataset),'Token preflight gate failed')
    require(sha(root/'preflight/ENCODINGS.json')==gate['encodings_sha256'] and sha(root/'TRAINING_ORDER.json')==gate['order_sha256'],'Preflight evidence changed')
    require(sha(root/'config.json')==gate['config_sha256'],'Training settings changed')
    out=root/arm;out.mkdir();require(torch.cuda.is_available(),'Quest GPU required')
    torch.manual_seed(config['seed']);np.random.seed(config['seed']);random.seed(config['seed']);torch.backends.cuda.matmul.allow_tf32=True
    processor=processor_for(assets)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(assets/'model',local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    targets=[n for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and 'visual' not in n and
        re.search(r'(?:language_model|model)\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)$',n)]
    require(len(targets)==252,'Unexpected language adapter coverage')
    model.requires_grad_(False)
    model=get_peft_model(model,LoraConfig(r=config['lora_rank'],lora_alpha=config['lora_alpha'],lora_dropout=config['lora_dropout'],
        target_modules=targets,bias='none',task_type='CAUSAL_LM'))
    params=[(n,v) for n,v in model.named_parameters() if v.requires_grad]
    require(params and all('lora_' in n and 'visual' not in n and 'lm_head' not in n for n,v in params),'Unexpected trainable weights')
    visual=[(n,v) for n,v in model.named_parameters() if 'visual' in n]
    require(visual and any('merger' in n for n,v in visual),'Frozen visual projector missing');visual_hash=tensor_digest(visual)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.config.use_cache=False
    items={r['sample_id']:encode(processor,r,dataset,arm,config['max_length']) for r in rows}
    hashes={r['sample_id']:r['input_sha256'] for r in read(root/'preflight/ENCODINGS.json')}
    require(all(input_digest(v)==hashes[k] for k,v in items.items()),'Real training input differs from preflight')
    validation=[r for r in rows if r['split']=='validation'];test=[r for r in rows if r['split']=='test']
    orders=read(root/'TRAINING_ORDER.json');train_ids={r['sample_id'] for r in rows if r['split']=='train'}
    require(all(len(batch)==4 and set(batch)<=train_ids for batch in orders),'Non-training row in optimizer order')
    per_epoch=len(train_ids)//4;require(len(orders)==per_epoch*config['epochs'],'Incomplete epoch schedule')
    for epoch in range(config['epochs']):
        sampled=[sid for batch in orders[epoch*per_epoch:(epoch+1)*per_epoch] for sid in batch]
        require(Counter(sampled)==Counter({sid:1 for sid in train_ids}),'An epoch repeats or omits training examples')
    optimizer=torch.optim.AdamW([v for _,v in params],lr=config['learning_rate'],weight_decay=config['weight_decay'])
    write(out/'PROVENANCE.json',dict(source_commit=receipt['source_commit'],dataset_sha256=acceptance_digest(dataset),
        acceptance_sha256=receipt['dataset_acceptance_sha256'],model_id=MODEL_ID,model_revision=staged['model_revision'],upstream_commit=UPSTREAM,
        config=config,lora_targets=targets,trainable_names=[n for n,v in params],trainable_parameters=sum(v.numel() for n,v in params),
        frozen_visual_sha256=visual_hash,preflight_sha256=sha(root/'preflight/COMPLETE.json'),
        slurm={k:v for k,v in os.environ.items() if k in ('SLURM_JOB_ID','SLURM_JOB_NODELIST','SLURM_JOB_PARTITION','SLURM_JOB_ACCOUNT')},
        gpu=torch.cuda.get_device_name(),same_inference_inputs=True,gold_rationale_prefix_at_evaluation=False))
    best_score=-1.;best_state=None;best_epoch=None;started=time.monotonic();epoch_scores=[]
    for step,batch in enumerate(orders,1):
        model.train();optimizer.zero_grad(set_to_none=True);totals=dict(total=0.,rationale=0.,verdict=0.)
        for j,sample in enumerate(batch):
            item=items[sample];fields={k:v.to('cuda') for k,v in item[0].items()}
            logits=model(**fields,use_cache=False).logits
            values=losses(logits,*[v.to('cuda') for v in item[1:4]],{k:v.to('cuda') for k,v in item[5].items()})
            require(torch.isfinite(values['total']),'Nonfinite training loss')
            if step==1 and j==0:
                probe=gradient_probe(model,values,arm)
                if arm=='B':
                    grads=torch.autograd.grad(values['rationale_text'],[v for n,v in params],retain_graph=True,allow_unused=True)
                    norm=sum(float(g.detach().float().square().sum()) for g in grads if g is not None)**.5
                    require(np.isfinite(norm) and norm>0,'Narrative text does not train language adapters')
                    probe['rationale_text']=dict(tokens=values['rationale_text_tokens'],gradient_norm=norm)
                write(out/'GRADIENT_PROBE.json',probe)
            (values['total']/4).backward()
            for k in totals:totals[k]+=float(values[k].detach())/4
        norm=torch.nn.utils.clip_grad_norm_([v for n,v in params],config['gradient_clip']);require(torch.isfinite(norm),'Nonfinite optimizer gradient')
        optimizer.step()
        record=dict(step=step,epoch=(step-1)//per_epoch+1,seconds=time.monotonic()-started,gradient_norm=float(norm),**totals)
        with (out/'curve.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if step%10==0:print(arm,json.dumps(record),flush=True)
        if step%per_epoch==0:
            epoch=step//per_epoch
            predictions=generate(model,processor,[items[r['sample_id']] for r in validation],validation,arm,config,hashes,
                out/('validation_epoch_%d.jsonl'%epoch))
            measured=score(predictions,arm);epoch_scores.append(dict(epoch=epoch,**measured))
            write(out/('validation_epoch_%d.json'%epoch),predictions);write(out/'VALIDATION_SCORES.json',epoch_scores)
            if measured['verdict_accuracy']>best_score:
                best_score=measured['verdict_accuracy'];best_epoch=epoch
                best_state={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
                model.save_pretrained(out/('selected_epoch_%d'%epoch))
            print(arm,'epoch',epoch,'validation',json.dumps(measured),flush=True)
    require(best_state is not None,'No validation-selected checkpoint')
    set_peft_model_state_dict(model,best_state)
    restored=get_peft_model_state_dict(model)
    require(set(restored)==set(best_state) and all(torch.equal(v.detach().cpu(),best_state[k]) for k,v in restored.items()),
            'Validation-selected adapter was not restored exactly')
    require(tensor_digest(visual)==visual_hash and all(not v.requires_grad and v.grad is None for n,v in visual),'Frozen vision/projector changed')
    model.save_pretrained(out/'adapter');processor.save_pretrained(out/'adapter')
    predictions=generate(model,processor,[items[r['sample_id']] for r in test],test,arm,config,hashes,
        out/'TEST_GENERATIONS.jsonl')
    write(out/'TEST_GENERATIONS.json',predictions);measured=score(predictions,arm,bootstrap=True)
    write(out/'RESULT.json',dict(passed=True,arm=arm,steps=len(orders),selected_epoch=best_epoch,
        selected_validation_accuracy=best_score,test=measured,test_evaluations=1,
        adapter_sha256=sha(out/'adapter/adapter_model.safetensors'),elapsed_seconds=time.monotonic()-started,
        frozen_vision_and_projector_unchanged=True,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
        scope='Source-held-out controlled conditional feasibility; no general necessity or broad reachability claim',api_calls=0))


if __name__=='__main__':
    stage=sys.argv[1];root=Path(sys.argv[2])
    try:
        if stage=='preflight':preflight(root)
        elif stage=='compare':compare(root)
        else:require(stage in ('A','B'),'Unknown arm');train(root,stage)
    except BaseException as error:
        write(root/('STOP_'+stage+'.json'),dict(error=type(error).__name__,message=str(error),slurm_job=os.environ.get('SLURM_JOB_ID')))
        raise
