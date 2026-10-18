"""V4 development A/B/C LoRA comparison on Quest (docs/generative_verifier/PLAN.md).

Stages: preflight (CPU token audit and fixed orders), zero-shot (base model on
validation), train ARM SEED (train + validation only), evaluate-test (once, after
TEST_FREEZE.json). train() never loads test rows.
"""
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
from collections import Counter,defaultdict
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig,PeftModel,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
from .common import read,write,sha,require,MODEL_ID,UPSTREAM,student_messages
from .binary_verifier import losses
from .loss import gradient_probe
from .verdict_diagnostic import tensor_digest
from .fd1000_contract import (configuration,validate,assistant_target,arm_kinds,readings,facts_text,rule_verdict,
    zero_shot_messages,probe_messages,probe_truth,schedules,check_schedule,edit_prefix,diagnostic_plan,FORCED_PREFIX,FACT_KEYS,KINDS)
from .fd1000_metrics import parse_output,auroc,row_record,summarize

TESTS=['verifier_training.tests.test_fd1000_contract','verifier_training.tests.test_fd1000_metrics',
       'verifier_training.tests.test_v4_export','verifier_training.tests.test_fd1000_comparison','verifier_training.tests.test_v4_report']
PROBE_WORDS={'yes':('yes','Yes'),'no':('no','No')}


def load_images(row,dataset):
    images=[]
    for spec in row['student']['images']:
        path=Path(dataset)/spec['path'];require(sha(path)==spec['sha256'],'Image changed')
        with Image.open(path) as im:images.append(im.convert('RGB'))
    return images


def vision_inputs(processor,row,dataset):
    return dict(processor.image_processor(images=load_images(row,dataset),return_tensors='pt'))


def text_ids(processor,text,grid):
    """Tokenize exactly as Qwen2.5-VL's processor does, reusing precomputed image grids."""
    token=processor.image_token;merge=processor.image_processor.merge_size**2;parts=text.split(token)
    require(len(parts)==len(grid)+1,'Image placeholder count differs')
    expanded=parts[0]+''.join(token*int(grid[i].prod()//merge)+part for i,part in enumerate(parts[1:]))
    return torch.tensor(processor.tokenizer(expanded)['input_ids'])


def prompt_item(processor,messages,vision):
    prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    return dict(prompt=prompt,grid=vision['image_grid_thw'],vision=vision,n=len(text_ids(processor,prompt,vision['image_grid_thw'])))


def encode(processor,row,dataset,kind,maximum,vision=None,verify=False,answer=None,keys=None):
    """Supervised item; verify=True also checks the fast path against the native processor.

    answer/keys replace the arm's target text and its key-value fields (fit-diagnostic controls);
    by default they are the Phase 2 target and, for facts, all four FACT_KEYS."""
    images=load_images(row,dataset) if verify else None
    vision=vision or vision_inputs(processor,row,dataset);grid=vision['image_grid_thw']
    messages=student_messages(row,'A');answer=answer or assistant_target(row,kind)
    keys=(FACT_KEYS if kind=='facts' else ()) if keys is None else tuple(keys)
    item=prompt_item(processor,messages,vision);prompt=item['prompt']
    full=processor.apply_chat_template(messages+[dict(role='assistant',content=answer)],tokenize=False,add_generation_prompt=False)
    require(full.startswith(prompt+answer),'Assistant target template changed')
    ids=text_ids(processor,full,grid);n=item['n']
    require(len(ids)<=maximum and torch.equal(ids[:n],text_ids(processor,prompt,grid)),'Prompt boundary or length invalid')
    if verify:
        native=processor(text=[full],images=images,return_tensors='pt',padding=False)
        require(torch.equal(native['input_ids'][0],ids) and torch.equal(native['pixel_values'],vision['pixel_values']) and
                torch.equal(native['image_grid_thw'],grid),'Fast tokenization differs from the native processor')
    eos=processor.tokenizer.convert_tokens_to_ids('<|im_end|>');ends=(ids[n:]==eos).nonzero()
    require(len(ends)==1,'Assistant terminator ambiguous');end=n+int(ends[0]);digit=end-1
    digit_ids=processor.tokenizer.encode(answer[-1],add_special_tokens=False)
    require(len(digit_ids)==1 and int(ids[digit])==digit_ids[0],'Verdict is not a single binary token')

    def boundary(text):
        t=text_ids(processor,prompt+text,grid);require(torch.equal(ids[:len(t)],t),'Not a token boundary: '+repr(text[-24:]));return len(t)

    require(boundary(answer[:-1])==digit,'Binary decision boundary differs');boundary(FORCED_PREFIX[kind])
    labels=ids.clone()[None];labels[:,:n]=-100;labels[:,end+1:]=-100
    vm=torch.zeros_like(labels,dtype=torch.bool);vm[:,digit]=True;rm=labels.ne(-100)&~vm
    eos_mask=labels.eq(eos);key=torch.zeros_like(vm);detail={}
    if keys:
        cursor=0
        for name in keys:
            value=' '+row['target']['facts'][name];start=answer.index(value+'\n',cursor)
            a=boundary(answer[:start]);b=boundary(answer[:start+len(value)]);require(b>a,'Empty key-field span')
            mask=torch.zeros_like(vm);mask[:,a:b]=True;detail['key_'+name]=mask;key|=mask;cursor=start+len(value)
    detail.update(key_values=key,template=rm&~key&~eos_mask,eos=eos_mask)
    require(torch.equal(rm|vm,labels.ne(-100)) and not (rm&vm).any() and not (key&~rm).any(),'Assistant token partition invalid')
    item.update(fields=dict(input_ids=ids[None],attention_mask=torch.ones_like(ids)[None],**vision),
        labels=labels,rm=rm,vm=vm,detail=detail,kind=kind,answer=answer)
    return item


def input_digest(item):
    """Prompt tokens plus pixels; identical across arms and target kinds of one row."""
    h=hashlib.sha256()
    for key,tensor in (('input_ids',text_ids_cache(item)),('image_grid_thw',item['vision']['image_grid_thw']),('pixel_values',item['vision']['pixel_values'])):
        h.update(key.encode());h.update(str(tuple(tensor.shape)).encode());h.update(str(tensor.dtype).encode())
        h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def text_ids_cache(item):
    return item['fields']['input_ids'][0,:item['n']]


def context(root,include_test=False):
    receipt=read(root/'RECEIPT.json');config=read(root/'config.json')
    require(config==configuration(config['schedule']) and sha(root/'config.json')==receipt['configuration_sha256'],'Fixed training configuration changed')
    require(os.environ['CB_CODE_COMMIT']==receipt['source_commit'],'Frozen training source differs')
    dataset=Path(receipt['dataset']);assets=Path(receipt['assets'])
    require(sha(dataset/'DEV_ACCEPTANCE.json')==receipt['dataset_acceptance_sha256'],'Dataset acceptance changed')
    rows,acceptance=validate(dataset,mode='development',include_test=include_test)
    require(include_test or all(r['split']!='test' for r in rows),'Locked test rows reached a development stage')
    require(sha(assets/'STAGED.json')==receipt['assets_stage_sha256'],'Model staging manifest changed')
    staged=read(assets/'STAGED.json');require(staged['model_id']==MODEL_ID,'Unexpected base model')
    for name,digest in staged['files'].items():require(sha(assets/'model'/name)==digest,'Staged base model changed')
    return receipt,config,dataset,assets,rows,staged


def processor_for(assets):
    pixels=configuration()['image_pixels']
    return AutoProcessor.from_pretrained(assets/'model',local_files_only=True,min_pixels=pixels,max_pixels=pixels)


def preflight(root):
    receipt,config,dataset,assets,rows,staged=context(root,include_test=True);out=root/'preflight';out.mkdir()
    subprocess.run([sys.executable,'-m','unittest',*TESTS],check=True)
    processor=processor_for(assets);records=[];tokens=processor.tokenizer
    for word in [w for words in PROBE_WORDS.values() for w in words]+['0','1']:
        require(len(tokens.encode(word,add_special_tokens=False))==1,'Probe or verdict word is not one token: '+word)
    for row in rows:
        vision=vision_inputs(processor,row,dataset);items={k:encode(processor,row,dataset,k,config['max_length'],vision,verify=True) for k in KINDS}
        digests={k:input_digest(v) for k,v in items.items()};require(len(set(digests.values()))==1,'Target kinds change the inference input')
        for item in items.values():
            require(item['labels'][:,:item['n']].eq(-100).all() and int(item['vm'].sum())==1,'Prompt supervised or decision mask invalid')
        f=items['facts'];zero={m:prompt_item(processor,zero_shot_messages(row,m),vision)['n'] for m in ('direct','facts','oracle')}
        require(max(zero.values())+64<=config['max_length'],'Zero-shot prompt too long')
        records.append(dict(sample_id=row['sample_id'],split=row['split'],input_sha256=digests['direct'],prompt_tokens=f['n'],
            tokens_direct=items['direct']['fields']['input_ids'].shape[1],tokens_facts=f['fields']['input_ids'].shape[1],
            key_value_tokens={k:int(f['detail']['key_'+k].sum()) for k in FACT_KEYS},template_tokens=int(f['detail']['template'].sum()),
            zero_shot_prompt_tokens=zero,image_grid=vision['image_grid_thw'].tolist()))
    train_ids=[r['sample_id'] for r in rows if r['split']=='train'];orders=schedules(train_ids,config)
    for seed in orders:
        for arm,order in orders[seed].items():check_schedule(order,train_ids,arm,config)
        first={arm:[s for b in order for s,k in b] for arm,order in orders[seed].items()}
        require(len({tuple(v) for v in first.values()})==1,'Arms of one seed differ in sample order')
    write(root/'ORDERS.json',orders);write(out/'ENCODINGS.json',records)
    longest=max(max(r['tokens_direct'],r['tokens_facts']) for r in records)
    write(out/'COMPLETE.json',dict(passed=True,rows=len(rows),by_split=dict(Counter(r['split'] for r in rows)),
        identical_inference_inputs_across_kinds=True,native_processor_equal=True,longest_sequence=longest,max_length=config['max_length'],
        no_truncation=longest<=config['max_length'],single_binary_verdict_token=True,forced_prefix_boundaries_checked=True,
        key_value_tokens_per_row=dict(Counter(sum(r['key_value_tokens'].values()) for r in records)),
        source_commit=receipt['source_commit'],dataset_acceptance_sha256=receipt['dataset_acceptance_sha256'],
        encodings_sha256=sha(out/'ENCODINGS.json'),orders_sha256=sha(root/'ORDERS.json'),config_sha256=sha(root/'config.json'),
        optimizer_steps=0,model_predictions=0,packages={k:metadata.version(k) for k in ('torch','transformers','peft','accelerate','numpy','pillow')},
        slurm_job=os.environ['SLURM_JOB_ID']))


def gate(root,receipt):
    done=read(root/'preflight/COMPLETE.json')
    require(done['passed'] and done['source_commit']==receipt['source_commit'] and done['dataset_acceptance_sha256']==receipt['dataset_acceptance_sha256'],'Token preflight gate failed')
    require(sha(root/'preflight/ENCODINGS.json')==done['encodings_sha256'] and sha(root/'ORDERS.json')==done['orders_sha256'] and
            sha(root/'config.json')==done['config_sha256'],'Preflight evidence changed')
    return {r['sample_id']:r['input_sha256'] for r in read(root/'preflight/ENCODINGS.json')}


def base_model(assets):
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(assets/'model',local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')


def verdict_position(processor,prefix,tokens,text):
    before=text.strip().rsplit('Verdict: ',1)[0]+'Verdict: '
    hits=[i for i in range(len(tokens)) if (prefix+processor.tokenizer.decode(tokens[:i],skip_special_tokens=True)).lstrip()==before]
    return hits[0] if len(hits)==1 else None


def run_generation(model,processor,item,kind,prefix,max_new,sampling=None,seed=None):
    """Greedy (or sampled) continuation after the chat prompt plus an optional forced prefix."""
    prefix=prefix or '';ids=text_ids(processor,item['prompt']+prefix,item['grid'])
    inputs=dict(input_ids=ids[None].cuda(),attention_mask=torch.ones_like(ids)[None].cuda(),
        pixel_values=item['vision']['pixel_values'].cuda(),image_grid_thw=item['grid'].cuda())
    options=dict(max_new_tokens=max_new,use_cache=True,repetition_penalty=1.,pad_token_id=processor.tokenizer.pad_token_id,
        return_dict_in_generate=True,output_logits=True)
    if sampling:
        torch.manual_seed(seed)
        options.update(do_sample=True,temperature=sampling['temperature'],top_p=sampling['top_p'],top_k=sampling['top_k'],
                       num_return_sequences=sampling['samples'])
    else:options.update(do_sample=False,temperature=None,top_p=None,top_k=None)
    digits=[processor.tokenizer.encode(d,add_special_tokens=False)[0] for d in ('0','1')];stops=set(processor.tokenizer.convert_tokens_to_ids(['<|im_end|>','<|endoftext|>']))
    torch.cuda.synchronize();start=time.monotonic()
    with torch.no_grad():result=model.generate(**inputs,**options)
    torch.cuda.synchronize();seconds=time.monotonic()-start;outputs=[]
    for j in range(result.sequences.shape[0]):
        tokens=result.sequences[j,len(ids):].tolist();count=next((i+1 for i,t in enumerate(tokens) if t in stops),len(tokens))
        text=prefix+processor.tokenizer.decode(tokens[:count],skip_special_tokens=True);parsed=parse_output(text,kind);p=None
        if parsed['predicted'] is not None:
            at=verdict_position(processor,prefix,tokens[:count],text)
            if at is not None and tokens[at] in digits:p=float(result.logits[at][j].float()[digits].softmax(0)[1])
        outputs.append(dict(generated=text,**parsed,p_feasible=p,generated_token_count=count,stopped=count<len(tokens) or tokens[count-1] in stops))
    return outputs,seconds


def evaluate(model,processor,rows,items,plan,config,hashes,stream):
    """plan: list of (reading, kind, prefix, mode) with mode greedy, sampled, oracle or edit:<field>."""
    model.eval();records=[];stream.touch(exist_ok=False)
    for index,row in enumerate(rows):
        for reading,kind,prefix,mode in plan:
            item=items[(row['sample_id'],kind)];record=row_record(row,reading,kind,prefix,hashes[row['sample_id']])
            if mode=='sampled':
                samples,seconds=run_generation(model,processor,item,kind,prefix,config['maximum_new_tokens'][kind],config['sampling'],
                    seed=config['sampling']['seed']*100003+index)
                record.update(samples=samples,generation_seconds=seconds,gold_rationale_prefix_used=False)
            else:
                if mode=='oracle':prefix=facts_text(row['target']['facts'])+'\nVerdict: '
                elif mode.startswith('edit:'):prefix=edit_prefix(row['target']['facts'],mode[5:])
                (output,),seconds=run_generation(model,processor,item,kind,prefix,config['maximum_new_tokens'][kind])
                record.update(output,forced_prefix=prefix,generation_seconds=seconds,gold_rationale_prefix_used=mode=='oracle')
                if mode.startswith('edit:'):
                    edited=dict(row['target']['facts'],**{mode[5:]:'no' if row['target']['facts'][mode[5:]]=='yes' else 'yes'})
                    record.update(edited_field=mode[5:],text_implied_verdict=rule_verdict(edited['target_inside'],edited['goal_inside']))
            records.append(record)
            with stream.open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
    return records


def selection_value(scores,arm):
    names=list(readings(arm));return sum(scores[n]['verdict_accuracy'] for n in names)/len(names)


def adapter_model(assets,config,seed):
    model=base_model(assets)
    targets=[n for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and 'visual' not in n and
        re.search(r'(?:language_model|model)\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)$',n)]
    require(len(targets)==252,'Unexpected language adapter coverage')
    model.requires_grad_(False)
    model=get_peft_model(model,LoraConfig(r=config['lora_rank'],lora_alpha=config['lora_alpha'],lora_dropout=config['lora_dropout'],
        target_modules=targets,bias='none',task_type='CAUSAL_LM'))
    return model,targets


def update(model,optimizer,items,batch,params,config,inspect=None):
    """One AdamW update over an effective batch; inspect(kind,values) sees each item's losses before backward."""
    model.train();optimizer.zero_grad(set_to_none=True);totals=defaultdict(float);counts=Counter()
    for sample,kind in batch:
        item=items[(sample,kind)];fields={k:v.to('cuda') for k,v in item['fields'].items()}
        logits=model(**fields,use_cache=False).logits
        values=losses(logits,item['labels'].cuda(),item['rm'].cuda(),item['vm'].cuda(),{k:v.cuda() for k,v in item['detail'].items()})
        require(torch.isfinite(values['total']),'Nonfinite training loss')
        if inspect:inspect(kind,values)
        (values['total']/len(batch)).backward();counts[kind]+=1
        for name in ('total','rationale','verdict'):totals[name]+=float(values[name].detach())/len(batch)
        totals[kind+'_verdict']+=float(values['verdict'].detach())
        if kind=='facts':totals['facts_key_values']+=float(values['key_values'].detach());totals['facts_template']+=float(values['template'].detach())
    for name in list(totals):
        for kind in KINDS:
            if name.startswith(kind+'_'):totals[name]/=counts[kind]
    norm=torch.nn.utils.clip_grad_norm_([v for n,v in params],config['gradient_clip']);require(torch.isfinite(norm),'Nonfinite optimizer gradient')
    optimizer.step()
    return dict(gradient_norm=float(norm),kinds=dict(counts),**totals)


def train(root,arm,seed):
    receipt,config,dataset,assets,rows,staged=context(root,include_test=False);hashes=gate(root,receipt)
    require(arm in config['arms'] and seed in config['seeds'],'Unplanned run')
    out=root/f'{arm}_seed{seed}';out.mkdir();require(torch.cuda.is_available(),'Quest GPU required')
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);torch.backends.cuda.matmul.allow_tf32=True
    processor=processor_for(assets);model,targets=adapter_model(assets,config,seed)
    params=[(n,v) for n,v in model.named_parameters() if v.requires_grad]
    require(params and all('lora_' in n and 'visual' not in n and 'lm_head' not in n for n,v in params),'Unexpected trainable weights')
    visual=[(n,v) for n,v in model.named_parameters() if 'visual' in n]
    require(visual and any('merger' in n for n,v in visual),'Frozen visual projector missing');visual_hash=tensor_digest(visual)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.config.use_cache=False
    train_rows=[r for r in rows if r['split']=='train'];validation=[r for r in rows if r['split']=='validation']
    plan=[(name,kind,prefix,'greedy') for name,(kind,prefix) in readings(arm).items()]
    eval_kinds=sorted({k for _,k,_,_ in plan+diagnostic_plan(arm)});items={}
    for row in train_rows+validation:
        vision=vision_inputs(processor,row,dataset)
        for kind in (arm_kinds(arm) if row['split']=='train' else eval_kinds):
            items[(row['sample_id'],kind)]=encode(processor,row,dataset,kind,config['max_length'],vision)
    require(all(input_digest(v)==hashes[s] for (s,k),v in items.items()),'Real training input differs from preflight')
    order=read(root/'ORDERS.json')[str(seed)][arm];check_schedule(order,[r['sample_id'] for r in train_rows],arm,config)
    optimizer=torch.optim.AdamW([v for _,v in params],lr=config['learning_rate'],weight_decay=config['weight_decay'])
    write(out/'PROVENANCE.json',dict(arm=arm,seed=seed,source_commit=receipt['source_commit'],dataset_acceptance_sha256=receipt['dataset_acceptance_sha256'],
        model_id=MODEL_ID,model_revision=staged['model_revision'],upstream_commit=UPSTREAM,config=config,lora_targets=targets,
        trainable_parameters=sum(v.numel() for n,v in params),frozen_visual_sha256=visual_hash,preflight_sha256=sha(root/'preflight/COMPLETE.json'),
        orders_sha256=sha(root/'ORDERS.json'),splits_loaded=sorted({r['split'] for r in rows}),
        slurm={k:v for k,v in os.environ.items() if k in ('SLURM_JOB_ID','SLURM_JOB_NODELIST','SLURM_JOB_PARTITION','SLURM_JOB_ACCOUNT')},
        gpu=torch.cuda.get_device_name(),same_inference_inputs=True,gold_rationale_prefix_at_selection=False))
    started=time.monotonic();checkpoints=[];states={};probes={}

    def probe(kind,values):
        """Gradient probe on the first item of each target kind."""
        if kind in probes:return
        probes[kind]=gradient_probe(model,values,'B')
        if kind=='facts':
            grads=torch.autograd.grad(values['key_values'],[v for n,v in params],retain_graph=True,allow_unused=True)
            norm=sum(float(g.detach().float().square().sum()) for g in grads if g is not None)**.5
            require(np.isfinite(norm) and norm>0,'Key facts do not train language adapters')
            probes[kind]['key_values']=dict(tokens=values['key_values_tokens'],gradient_norm=norm)
        write(out/'GRADIENT_PROBE.json',probes)

    for step,batch in enumerate(order,1):
        totals=update(model,optimizer,items,batch,params,config,probe)
        record=dict(step=step,seconds=time.monotonic()-started,**totals)
        with (out/'curve.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if step%10==0:print(arm,seed,json.dumps(record),flush=True)
        if step in config['evaluation_updates']:
            measured=summarize(evaluate(model,processor,validation,items,plan,config,hashes,out/f'validation_step_{step}.jsonl'),config)
            value=selection_value(measured,arm);checkpoints.append(dict(step=step,selection_value=value,scores=measured))
            states[step]={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
            model.save_pretrained(out/f'checkpoint_{step}');write(out/'VALIDATION_SCORES.json',checkpoints)
            print(arm,seed,'step',step,'validation',json.dumps({k:v.get('verdict_accuracy') for k,v in measured.items()}),flush=True)
    best=max(checkpoints,key=lambda c:(c['selection_value'],-c['step']));selected=best['step']
    set_peft_model_state_dict(model,states[selected]);restored=get_peft_model_state_dict(model)
    require(set(restored)==set(states[selected]) and all(torch.equal(v.detach().cpu(),states[selected][k]) for k,v in restored.items()),
            'Validation-selected adapter was not restored exactly')
    require(tensor_digest(visual)==visual_hash and all(not v.requires_grad and v.grad is None for n,v in visual),'Frozen vision/projector changed')
    model.save_pretrained(out/'selected_adapter');processor.save_pretrained(out/'selected_adapter')
    diagnostics=summarize(evaluate(model,processor,validation,items,diagnostic_plan(arm),config,hashes,out/'validation_diagnostics.jsonl'),config) if arm!='A' else {}
    write(out/'RESULT.json',dict(passed=True,arm=arm,seed=seed,steps=len(order),selected_step=selected,
        selection_value=best['selection_value'],validation=best['scores'],validation_diagnostics=diagnostics,
        checkpoint_values={c['step']:c['selection_value'] for c in checkpoints},
        adapter_sha256=sha(out/'selected_adapter/adapter_model.safetensors'),elapsed_seconds=time.monotonic()-started,
        frozen_vision_and_projector_unchanged=True,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
        test_rows_loaded=0,test_evaluations=0,api_calls=0,
        scope='Validation-only development measurement; the locked test is scored separately by evaluate-test'))


def zero_shot(root):
    """Phase 1.1/1.2: perception probe and three zero-shot verdict prompts on validation, base model only."""
    receipt,config,dataset,assets,rows,staged=context(root,include_test=False);hashes=gate(root,receipt)
    validation=[r for r in rows if r['split']=='validation'];out=root/'zero_shot';out.mkdir();require(torch.cuda.is_available(),'Quest GPU required')
    torch.manual_seed(1014);processor=processor_for(assets);model=base_model(assets);model.eval();tokens=processor.tokenizer
    words={k:[tokens.encode(w,add_special_tokens=False)[0] for w in v] for k,v in PROBE_WORDS.items()};started=time.monotonic()
    pairs=defaultdict(list)
    for r in validation:pairs[(r['metadata']['unit_id'],r['evidence']['assignment'])].append(r)
    probe=[];stream=out/'probe.jsonl';stream.touch(exist_ok=False)
    with torch.no_grad():
        for key,members in sorted(pairs.items()):
            vision=vision_inputs(processor,members[0],dataset);truth=probe_truth(members)
            for (entity,color),inside in sorted(truth.items()):
                item=prompt_item(processor,probe_messages(entity,color),vision);ids=text_ids(processor,item['prompt'],item['grid'])
                logits=model(input_ids=ids[None].cuda(),attention_mask=torch.ones_like(ids)[None].cuda(),
                    pixel_values=vision['pixel_values'].cuda(),image_grid_thw=item['grid'].cuda()).logits[0,-1].float()
                prob=logits.softmax(0);yes=float(prob[words['yes']].sum());no=float(prob[words['no']].sum())
                record=dict(unit_id=key[0],assignment=key[1],source_group=members[0]['metadata']['ancestry_component'],
                    checkpoint=members[0]['metadata']['checkpoint'],construction_mode=members[0]['metadata']['construction_mode'],
                    entity=entity,color=color,truth=int(inside),p_yes=yes/(yes+no),yes_no_mass=yes+no)
                probe.append(record)
                with stream.open('a') as f:f.write(json.dumps(record)+'\n')
    items={}
    for row in validation:
        vision=vision_inputs(processor,row,dataset)
        for mode in ('direct','facts','oracle'):items[(row['sample_id'],'zs_'+mode)]=prompt_item(processor,zero_shot_messages(row,mode),vision)
    records=[];stream=out/'verdicts.jsonl';stream.touch(exist_ok=False)
    for row in validation:
        for mode,kind,prefix in (('direct','direct','Verdict: '),('facts','facts',None),('oracle','direct','Verdict: ')):
            (output,),seconds=run_generation(model,processor,items[(row['sample_id'],'zs_'+mode)],kind,prefix,
                config['maximum_new_tokens']['direct'] if kind=='direct' else 96)
            record=row_record(row,'zero_shot_'+mode,kind,prefix,hashes[row['sample_id']])
            record.update(output,generation_seconds=seconds,gold_rationale_prefix_used=False,oracle_facts_in_prompt=mode=='oracle')
            records.append(record)
            with stream.open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
    probe_summary={}
    for entity in ('target','goal'):
        part=[p for p in probe if p['entity']==entity]
        probe_summary[entity]=dict(bits=len(part),accuracy=sum((p['p_yes']>.5)==bool(p['truth']) for p in part)/len(part),
            auroc=auroc([p['truth'] for p in part],[p['p_yes'] for p in part]),mean_yes_no_mass=sum(p['yes_no_mass'] for p in part)/len(part),
            predicted_yes_rate=sum(p['p_yes']>.5 for p in part)/len(part))
    pairs_correct=defaultdict(list)
    for p in probe:pairs_correct[(p['unit_id'],p['assignment'])].append((p['p_yes']>.5)==bool(p['truth']))
    probe_summary['all_four_bits_correct']=sum(all(v) for v in pairs_correct.values())/len(pairs_correct)
    write(out/'SUMMARY.json',dict(passed=True,image_pairs=len(pairs),probe=probe_summary,verdicts=summarize(records,config),
        model_id=MODEL_ID,model_revision=staged['model_revision'],adapter=None,split='validation',test_rows_loaded=0,
        elapsed_seconds=time.monotonic()-started,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,api_calls=0,
        source_commit=receipt['source_commit'],slurm_job=os.environ['SLURM_JOB_ID']))


def evaluate_test(root,name):
    """Score one frozen target (ARM_seedN or zero_shot) exactly once on the locked test split."""
    freeze=read(root/'TEST_FREEZE.json');receipt,config,dataset,assets,rows,staged=context(root,include_test=True);hashes=gate(root,receipt)
    require(freeze['source_commit']==receipt['source_commit'] and freeze['config_sha256']==sha(root/'config.json'),'Freeze belongs to another campaign')
    require(sha(root/'source'/'verifier_training/configs/v4_dev_prereg.json')==freeze['prereg_sha256'],'Preregistration changed after freeze')
    require({(r['arm'],r['seed']) for r in freeze['runs']}=={(a,s) for a in config['arms'] for s in config['seeds']},'Freeze does not list every planned run')
    runs={f'{r["arm"]}_seed{r["seed"]}':r for r in freeze['runs']};require(name in runs or name=='zero_shot','Target was not frozen')
    (root/'test').mkdir(exist_ok=True);out=root/'test'/name;out.mkdir();require(torch.cuda.is_available(),'Quest GPU required')
    test=[r for r in rows if r['split']=='test'];started=time.monotonic();processor=processor_for(assets)
    vision={r['sample_id']:vision_inputs(processor,r,dataset) for r in test}
    if name=='zero_shot':
        torch.manual_seed(1014);model=base_model(assets);model.eval();records=[];stream=out/'records.jsonl';stream.touch(exist_ok=False)
        for row in test:
            for mode,kind,prefix in (('direct','direct','Verdict: '),('facts','facts',None)):
                item=prompt_item(processor,zero_shot_messages(row,mode),vision[row['sample_id']])
                (output,),seconds=run_generation(model,processor,item,kind,prefix,config['maximum_new_tokens']['direct'] if kind=='direct' else 96)
                record=row_record(row,'zero_shot_'+mode,kind,prefix,hashes[row['sample_id']])
                record.update(output,generation_seconds=seconds,gold_rationale_prefix_used=False);records.append(record)
                with stream.open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        adapter=None
    else:
        run=runs[name];folder=root/name;result=read(folder/'RESULT.json')
        require(sha(folder/'RESULT.json')==run['result_sha256'] and sha(folder/'selected_adapter/adapter_model.safetensors')==run['adapter_sha256']==result['adapter_sha256'],
                'Selected adapter changed after freeze')
        items={(r['sample_id'],k):encode(processor,r,dataset,k,config['max_length'],vision[r['sample_id']]) for r in test for k in KINDS}
        require(all(input_digest(v)==hashes[s] for (s,k),v in items.items()),'Test input differs from preflight')
        torch.manual_seed(run['seed']);model=PeftModel.from_pretrained(base_model(assets),folder/'selected_adapter',is_trainable=False);model.eval()
        plan=[(n,k,p,'greedy') for n,(k,p) in readings(run['arm']).items()]+diagnostic_plan(run['arm'])
        records=evaluate(model,processor,test,items,plan,config,hashes,out/'records.jsonl');adapter=run['adapter_sha256']
    write(out/'SUMMARY.json',summarize(records,config,bootstrap=True))
    write(out/'COMPLETE.json',dict(passed=True,target=name,test_rows=len(test),records=len(records),adapter_sha256=adapter,
        freeze_sha256=sha(root/'TEST_FREEZE.json'),evaluations=1,elapsed_seconds=time.monotonic()-started,
        peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,source_commit=receipt['source_commit'],
        slurm_job=os.environ['SLURM_JOB_ID'],api_calls=0))


if __name__=='__main__':
    stage=sys.argv[1];root=Path(sys.argv[2]);label=stage+('_'+'_'.join(sys.argv[3:]) if len(sys.argv)>3 else '')
    try:
        if stage=='preflight':preflight(root)
        elif stage=='zero-shot':zero_shot(root)
        elif stage=='train':train(root,sys.argv[3],int(sys.argv[4]))
        elif stage=='evaluate-test':evaluate_test(root,sys.argv[3])
        else:raise ValueError('Unknown stage')
    except BaseException as error:
        write(root/('STOP_'+label+'.json'),dict(error=type(error).__name__,message=str(error),slurm_job=os.environ.get('SLURM_JOB_ID')))
        raise
