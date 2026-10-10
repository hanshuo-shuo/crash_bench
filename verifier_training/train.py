"""Bounded A/B LoRA overfit smoke. Full training intentionally not launched here."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import re
import time
import importlib.metadata as metadata
import numpy as np
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
from .common import *
from .data import encode, collate
from .loss import masked_losses, gradient_probe


def generate(model,processor,items,rows,variant):
    model.eval();outputs=[]
    with torch.no_grad():
        for item,row in zip(items,rows):
            fields=item[0]; n=item[4]
            prompt={k:(v[:,:n] if k in ('input_ids','attention_mask') else v).to('cuda') for k,v in fields.items()}
            result=model.generate(**prompt,max_new_tokens=160 if variant=='B' else 12,do_sample=False,use_cache=True,
                                  pad_token_id=processor.tokenizer.pad_token_id)
            text=processor.tokenizer.decode(result[0,n:],skip_special_tokens=True).strip()
            match=re.fullmatch(r'(?:Rationale: .+\n)?Verdict: (feasible|infeasible)',text,flags=re.S)
            pred=match.group(1) if match else None
            outputs.append({'sample_id':row['sample_id'],'target':target_text(row,variant),'generated':text,
                            'verdict_correct':pred==row['target']['verdict'],'exact_match':text==target_text(row,variant),
                            'format_valid':bool(match) and (variant=='A' or text.startswith('Rationale: '))})
    return outputs


def evaluate_loss(model,items):
    model.eval(); accum={'total':0.,'rationale':0.,'verdict':0.}
    with torch.no_grad():
        for item in items:
            batch={k:v.to('cuda') for k,v in item[0].items()}
            losses=masked_losses(model(**batch,use_cache=False).logits,*[x.to('cuda') for x in item[1:4]])
            for k in accum:accum[k]+=losses[k].item()/len(items)
    return accum


def train(a):
    out=a.out;out.mkdir(parents=True,exist_ok=False)
    config=read(a.config);require(config['mode']=='overfit_smoke','Full training requires a separately authorized launcher')
    require(torch.cuda.is_available(),'Quest GPU required')
    audit=read(a.dataset/'AUDIT.json');require(audit['passed'] and audit['engineering_supervision_only'],'Dataset audit gate failed')
    token_audit=read(a.dataset/'TOKEN_AUDIT.json')
    require(token_audit.get('input_contract')==INPUT_CONTRACT and token_audit.get('identical_inference_inputs'), 'A/B input-equality gate failed')
    require(sha(a.dataset/'dataset.jsonl')==audit['dataset_sha256'],'Dataset changed')
    ids=read(a.dataset/'smoke_ids.json');require(8<=len(ids)<=16,'Smoke scope exceeded')
    rows=[json.loads(l) for l in (a.dataset/'dataset.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['sample_id'] in ids]
    require(len(rows)==len(ids),'Missing samples')
    stage=read(a.assets/'STAGED.json')
    require(stage['model_id']==MODEL_ID,'Unexpected model')
    for name,expected in stage['files'].items():require(sha(a.assets/'model'/name)==expected,'Model asset changed')
    torch.manual_seed(config['seed']);np.random.seed(config['seed']);random.seed(config['seed'])
    torch.backends.cuda.matmul.allow_tf32=True
    processor=AutoProcessor.from_pretrained(a.assets/'model',local_files_only=True,min_pixels=224*224,max_pixels=224*224)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(a.assets/'model',local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    # Match concrete language-layer names, never generic q_proj or all-linear.
    targets=[n for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and
             re.search(r'(?:language_model|model)\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)$',n) and 'visual' not in n]
    require(len(targets)>0,'No LM targets matched')
    model.requires_grad_(False)
    model=get_peft_model(model,LoraConfig(r=config['lora_rank'],lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],target_modules=targets,bias='none',task_type='CAUSAL_LM'))
    trainable=[n for n,p in model.named_parameters() if p.requires_grad]
    require(all('lora_' in n and 'visual' not in n and 'lm_head' not in n for n in trainable),'Unexpected trainable parameters')
    visual=[(n,p) for n,p in model.named_parameters() if 'visual' in n]
    require(visual and any('merger' in n for n,p in visual),'Cannot identify frozen vision projector')
    visual_anchor={n:p.detach().clone() for n,p in visual if 'merger' in n}
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    model.config.use_cache=False
    items=[encode(processor,r,a.dataset,a.variant,config['max_length']) for r in rows]
    provenance={'variant':a.variant,'config':config,'model_id':MODEL_ID,'model_revision':stage['model_revision'],
        'input_contract':INPUT_CONTRACT,
        'dataset_sha256':audit['dataset_sha256'],'sample_ids':ids,'code_commit':os.environ['CB_CODE_COMMIT'],
        'upstream_commit':UPSTREAM,'slurm':{k:v for k,v in os.environ.items() if k in ('SLURM_JOB_ID','SLURM_JOB_NODELIST','SLURM_JOB_PARTITION','SLURM_JOB_ACCOUNT','SLURM_CPUS_PER_TASK')},
        'packages':{k:metadata.version(k) for k in ('torch','transformers','peft','accelerate','numpy','pillow')},
        'gpu':torch.cuda.get_device_name(),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'trainable_names':trainable,'lora_targets':targets,'vision_frozen':True,'projector_frozen':True,
        'token_masks':[{'id':r['sample_id'],'prompt_tokens':i[4],'assistant_tokens':int(i[1].ne(-100).sum()),
                        'rationale_tokens':int(i[2].sum()),'verdict_tokens':int(i[3].sum())} for r,i in zip(rows,items)],
        'metric_scope':'Exact same eight training examples; memorization/engineering only, no generalization'}
    write(out/'PROVENANCE.json',provenance)
    write(out/'generations_before.json',generate(model,processor,items,rows,a.variant))
    baseline=evaluate_loss(model,items);write(out/'loss_before.json',baseline)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['learning_rate'],weight_decay=0.)
    log=out/'curve.jsonl';start=time.monotonic();passed=False;probes={}
    for step in range(1,config['max_steps']+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        order=np.random.permutation(len(items))[:config['effective_batch_size']]
        metrics={'total':0.,'rationale':0.,'verdict':0.}
        for j,index in enumerate(order):
            batch,labels,rm,vm=collate([items[index]],processor.tokenizer.pad_token_id)
            batch={k:v.to('cuda') for k,v in batch.items()}
            losses=masked_losses(model(**batch,use_cache=False).logits,labels.to('cuda'),rm.to('cuda'),vm.to('cuda'))
            if step==1 and j==0:
                probes=gradient_probe(model,losses,a.variant);write(out/'GRADIENT_PROBE.json',probes)
            require(torch.isfinite(losses['total']).item(),'Nonfinite loss')
            (losses['total']/len(order)).backward()
            for k in metrics:metrics[k]+=losses[k].item()/len(order)
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        require(torch.isfinite(norm).item(),'Nonfinite gradient')
        optimizer.step()
        entry={'step':step,'elapsed_seconds':time.monotonic()-start,**metrics,'gradient_norm':norm.item()}
        with log.open('a') as f:f.write(json.dumps(entry)+'\n')
        if step%5==0:print(json.dumps(entry),flush=True)
        if step%config['evaluate_every']==0 or step==config['max_steps']:
            generated=generate(model,processor,items,rows,a.variant)
            measured=evaluate_loss(model,items)
            write(out/f'generations_step_{step:04d}.json',generated)
            write(out/f'loss_step_{step:04d}.json',measured)
            # Deliberately strict: a rationale hallucination/paraphrase cannot silently pass.
            passed=all(g['verdict_correct'] and g['format_valid'] and g['exact_match'] for g in generated)
            passed &= measured['total'] < baseline['total']*.25
            passed &= a.variant=='A' or measured['rationale'] < baseline['rationale']*.25
            if passed:break
    require(all(torch.equal(p,visual_anchor[n]) for n,p in visual if n in visual_anchor),'Projector changed')
    require(all(not p.requires_grad and p.grad is None for n,p in visual),'Visual parameters received gradients')
    model.save_pretrained(out/'adapter');processor.save_pretrained(out/'adapter')
    write(out/'generations_final.json',generated)
    result={'passed':bool(passed),'steps':step,'train_examples':len(rows),'before':baseline,'after':measured,
        'correct_verdicts':sum(g['verdict_correct'] for g in generated),'exact_outputs':sum(g['exact_match'] for g in generated),
        'gradient_probe':probes,'vision_and_projector_frozen':True,'projector_bitwise_unchanged':True,
        'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'elapsed_seconds':time.monotonic()-start,
        'scientific_generalization_claim':False,'full_training_launched':False}
    write(out/'RESULT.json',result);print(json.dumps(result),flush=True)
    if not passed:raise RuntimeError('Bounded overfit criteria not met; full training remains gated')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--variant',choices=['A','B'],required=True)
    a=p.parse_args()
    try:train(a)
    except BaseException as e:
        if a.out.exists():write(a.out/'STOP.json',{'error':type(e).__name__,'reason':str(e),'job':os.environ.get('SLURM_JOB_ID')})
        raise
