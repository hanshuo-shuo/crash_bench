"""Gated, single-configuration binary B experiment; old runs stay immutable."""
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
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from .common import read, write, require, sha, MODEL_ID, UPSTREAM, INPUT_CONTRACT
from .binary_verifier import encode, target, parse, losses, LABEL_TO_DIGIT
from .data import encode as old_encode
from .loss import gradient_probe
from .engineering_contract import DATA_SHA, epoch_orders, diagnostic_row, score_rationale
from .verdict_diagnostic import tensor_digest

METRICS=('total','rationale','verdict','rationale_text','format','eos')


def configuration(stage):
    source=Path(__file__).parent/'configs'/('smoke.json' if stage=='smoke' else 'proposed_B.json')
    c=read(source)
    c.update(mode='binary_equal_component_'+stage,variant='B_binary',digit_mapping=LABEL_TO_DIGIT,
        loss='0.5 * mean(non-decision assistant CE, including rationale/format/EOS) + 0.5 * single binary decision CE',
        component_weights={'narrative_and_format_and_eos':.5,'binary_decision':.5},
        approved_for_launch=True,evaluate_every=20 if stage=='smoke' else 30,
        generate_every=20 if stage=='smoke' else 150,
        stopping='Strict 8/8 exact and >=75% component loss reductions by step240' if stage=='smoke' else 'Exactly300 updates /10 epochs; no early stopping',
        initialization='Fresh base and fresh LoRA, never the previous word-verdict or smoke adapter',
        generation={'do_sample':False,'max_new_tokens':160,'repetition_penalty':1.05,
                    'reason':'Preserve original native decoding default; raw and processed decision losses both reported'},
        training_samples=8 if stage=='smoke' else 120,
        scientific_claim='Engineering training-pool fit only; one original source scene')
    c.pop('stop_rule',None)
    return c


def context(root):
    receipt=read(root/'RECEIPT.json')
    campaign=Path(receipt['campaign']);assets=Path(receipt['assets']);dataset=campaign/'dataset'
    require(sha(dataset/'dataset.jsonl')==DATA_SHA,'Dataset changed')
    require(read(campaign/'audit/AUDIT.json')['passed_for_engineering_only'],'Observability isolation missing')
    diag=Path(receipt['diagnostic'])/'diagnostic'
    done=read(diag/'COMPLETE.json')
    require(done['passed'] and done['weights_bitwise_unchanged'] and done['optimizer_steps']==0,'Read-only diagnostic incomplete')
    require(sha(diag/'COMPLETE.json')==receipt['diagnostic_complete_sha256'],'Diagnostic gate changed')
    prefix=Path(receipt['prefix_diagnostic'])/'prefix'
    require(read(prefix/'COMPLETE.json')['passed'] and read(prefix/'COMPLETE.json')['rows']==120,'Native prefix diagnosis incomplete')
    require(sha(prefix/'COMPLETE.json')==receipt['prefix_complete_sha256'],'Prefix gate changed')
    rows=[json.loads(x) for x in (dataset/'dataset.jsonl').read_text().splitlines()]
    return receipt,campaign,assets,dataset,rows


def processor_for(assets):
    return AutoProcessor.from_pretrained(assets/'model',local_files_only=True,min_pixels=224*224,max_pixels=224*224)


def input_digest(item):
    return tensor_digest((k,v[:,:item[4]] if k in ('input_ids','attention_mask') else v) for k,v in sorted(item[0].items()))


def preflight(root):
    receipt,campaign,assets,dataset,rows=context(root)
    out=root/'preflight';out.mkdir(exist_ok=False)
    from .binary_report import summarize_diagnostics
    summarize_diagnostics(root)
    with (out/'unit_tests.txt').open('w') as log:
        subprocess.run([sys.executable,'-m','unittest','discover','-s','verifier_training/tests','-v'],stdout=log,stderr=subprocess.STDOUT,check=True)
    p=processor_for(assets);audit=[]
    for row in rows:
        new=encode(p,row,dataset);old=old_encode(p,row,dataset,'B')
        require(input_digest(new)==input_digest(old),'Student inference input changed')
        require(int(new[3].sum())==1,'Not one classification token')
        audit.append({'sample_id':row['sample_id'],'input_sha256':input_digest(new),
            'original_word_input_sha256':input_digest(old),'prompt_tokens':new[4],
            'assistant_tokens':int(new[1].ne(-100).sum()),'narrative_group_tokens':int(new[2].sum()),
            'binary_tokens':int(new[3].sum()),'digit_token_id':int(new[1][new[3]][0]),
                'detail_counts':{k:int(v.sum()) for k,v in new[5].items()},
                'digits_seen_in_student_prompt':{v:p.tokenizer.encode(v,add_special_tokens=False)[0] in new[0]['input_ids'][0,:new[4]].tolist() for v in ('0','1')}})
    ids=read(dataset/'smoke_ids.json');subset=[r for r in rows if r['sample_id'] in ids]
    require(len(subset)==8 and sum(r['target']['verdict']=='feasible' for r in subset)==4,'Smoke must be original balanced eight')
    write(out/'ENCODINGS.json',audit)
    write(root/'smoke_config.json',configuration('smoke'));write(root/'full_config.json',configuration('full'))
    write(out/'COMPLETE.json',{'passed':True,'code_commit':receipt['source_commit'],'slurm_job':os.environ['SLURM_JOB_ID'],
        'examples':120,'paired_old_new_encoding_checks':240,'identical_inputs':True,'dataset_sha256':DATA_SHA,
        'binary_tokens_per_example':1,'smoke_ids':ids,'digit_mapping':LABEL_TO_DIGIT,
        'config_sha256':{s:sha(root/(s+'_config.json')) for s in ('smoke','full')}})


def gpu_losses(model,item):
    fields={k:v.to('cuda') for k,v in item[0].items()}
    logits=model(**fields,use_cache=False).logits
    return losses(logits,*[x.to('cuda') for x in item[1:4]],{k:v.to('cuda') for k,v in item[5].items()})


def evaluate(model,items,rows):
    model.eval(); values=[]
    with torch.no_grad():
        for row,item in zip(rows,items):
            vals=gpu_losses(model,item)
            values.append({'sample_id':row['sample_id'],'truth':row['target']['verdict'],**{k:float(vals[k]) for k in METRICS}})
    return {'means':{k:sum(x[k] for x in values)/len(values) for k in METRICS},
        'by_class':{c:{k:sum(x[k] for x in values if x['truth']==c)/sum(x['truth']==c for x in values) for k in METRICS}
                    for c in ('feasible','infeasible')},'examples':values}


def generate(model,p,items,rows,condition='original'):
    model.eval();outputs=[]
    with torch.no_grad():
        for row,item in zip(rows,items):
            n=item[4];fields={k:(v[:,:n] if k in ('input_ids','attention_mask') else v).to('cuda') for k,v in item[0].items()}
            generated=model.generate(**fields,max_new_tokens=160,do_sample=False,use_cache=True,
                repetition_penalty=1.05,pad_token_id=p.tokenizer.pad_token_id,
                return_dict_in_generate=True,output_scores=True,output_logits=True)
            tokens=generated.sequences[0,n:]
            text=p.tokenizer.decode(tokens,skip_special_tokens=True).strip();pred=parse(text)
            expected='infeasible' if condition=='horizon_0' else row['target']['verdict']
            grade=score_rationale(text,row,condition)
            expected_text=target(row) if condition=='original' else target(dict(row,target=dict(row['target'],verdict='infeasible')))
            decision=None
            if pred is not None:
                prefix_text=text.rsplit('Verdict: ',1)[0]+'Verdict: '
                positions=[i for i in range(len(tokens)) if p.tokenizer.decode(tokens[:i],skip_special_tokens=True).lstrip()==prefix_text]
                require(len(positions)==1,'Cannot identify generated binary decision')
                at=positions[0];logp=generated.scores[at][0].float().log_softmax(-1)
                raw_logp=generated.logits[at][0].float().log_softmax(-1)
                ids={k:p.tokenizer.encode(v,add_special_tokens=False)[0] for k,v in LABEL_TO_DIGIT.items()}
                norm=torch.tensor([float(logp[ids[c]]) for c in ('infeasible','feasible')]).softmax(0).tolist()
                decision={'true_class_nll':-float(logp[ids[expected]]),'true_digit':LABEL_TO_DIGIT[expected],
                    'true_raw_class_nll':-float(raw_logp[ids[expected]]),
                    'raw_probabilities':{c:float(logp[i].exp()) for c,i in ids.items()},
                    'binary_normalized':dict(zip(('infeasible','feasible'),norm)),
                    'prefix_token_ids':tokens[:at].tolist()}
            eos_id=p.tokenizer.convert_tokens_to_ids('<|im_end|>')
            eos_losses=[-float(generated.scores[i][0].float().log_softmax(-1)[eos_id]) for i,v in enumerate(tokens) if int(v)==eos_id]
            outputs.append({'sample_id':row['sample_id'],'condition':condition,'generated':text,
                'predicted_verdict':pred,'expected_verdict':expected,'verdict_correct':pred==expected,
                'format_valid':pred is not None,'exact_match':text==expected_text,
                'rationale_assessment':grade,'input_sha256':input_digest(item),
                'native_decision':decision,'native_generated_eos_nll':eos_losses})
    return outputs


def score(outputs):
    def native(c):
        values=[x['native_decision']['true_class_nll'] for x in outputs if x['expected_verdict']==c and x['native_decision'] is not None]
        raw=[x['native_decision']['true_raw_class_nll'] for x in outputs if x['expected_verdict']==c and x['native_decision'] is not None]
        return {'n_scored':len(values),'mean_nll':sum(values)/len(values) if values else None,
                'mean_raw_nll':sum(raw)/len(raw) if raw else None}
    return {'n':len(outputs),'verdict_correct':sum(x['verdict_correct'] for x in outputs),
        'by_class':{c:{'n':sum(x['expected_verdict']==c for x in outputs),
                      'correct':sum(x['expected_verdict']==c and x['verdict_correct'] for x in outputs),
                      'native_decisive_token_loss':native(c)} for c in ('feasible','infeasible')},
        'format_valid':sum(x['format_valid'] for x in outputs),'exact_full_output':sum(x['exact_match'] for x in outputs),
        'rationale_supported_complete':sum(x['rationale_assessment']['status']=='supported_complete' for x in outputs),
        'full_output_supported_and_verdict_correct':sum(x['rationale_assessment']['status']=='supported_complete' and x['verdict_correct'] for x in outputs)}


def train(root,stage):
    receipt,campaign,assets,dataset,rows=context(root)
    gate=read(root/'preflight/COMPLETE.json');require(gate['passed'],'Corrected-loss tests not passed')
    require(gate['code_commit']==receipt['source_commit'],'Preflight source changed')
    if stage=='full':require(read(root/'smoke/RESULT.json')['passed'],'Balanced smoke did not pass')
    config=read(root/(stage+'_config.json'));require(config==configuration(stage),'Fixed config changed')
    require(sha(root/(stage+'_config.json'))==gate['config_sha256'][stage],'Config gate changed')
    if stage=='smoke':rows=[r for r in rows if r['sample_id'] in gate['smoke_ids']]
    out=root/stage;out.mkdir(exist_ok=False)
    staged=read(assets/'STAGED.json')
    for name,h in staged['files'].items():require(sha(assets/'model'/name)==h,'Model changed')
    require(torch.cuda.is_available(),'Quest GPU required')
    torch.manual_seed(713);np.random.seed(713);random.seed(713);torch.backends.cuda.matmul.allow_tf32=True
    p=processor_for(assets)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(assets/'model',local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    targets=[n for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and
        re.search(r'(?:language_model|model)\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)$',n) and 'visual' not in n]
    require(targets==read(campaign/'B/PROVENANCE.json')['lora_targets'],'Architecture/LoRA targets changed')
    model.requires_grad_(False)
    model=get_peft_model(model,LoraConfig(r=config['lora_rank'],lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],target_modules=targets,bias='none',task_type='CAUSAL_LM'))
    trainable=[(n,v) for n,v in model.named_parameters() if v.requires_grad]
    require(all('lora_' in n and 'visual' not in n and 'lm_head' not in n for n,v in trainable),'Unexpected trainable parameters')
    frozen=[(n,v) for n,v in model.named_parameters() if 'visual' in n]
    require(frozen and any('merger' in n for n,v in frozen),'Projector not found')
    visual_hash=tensor_digest(frozen)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads();model.config.use_cache=False
    items=[encode(p,r,dataset) for r in rows]
    write(out/'PROVENANCE.json',{'code_commit':receipt['source_commit'],'upstream_commit':UPSTREAM,
        'model_id':MODEL_ID,'model_revision':staged['model_revision'],'input_contract':INPUT_CONTRACT,
        'config':config,'dataset_sha256':DATA_SHA,'sample_ids':[r['sample_id'] for r in rows],
        'lora_targets':targets,'trainable_names':[n for n,v in trainable],'trainable_parameters':sum(v.numel() for n,v in trainable),
        'packages':{k:metadata.version(k) for k in ('torch','transformers','peft','accelerate','numpy','pillow')},
        'slurm':{k:v for k,v in os.environ.items() if k in ('SLURM_JOB_ID','SLURM_JOB_NODELIST','SLURM_JOB_PARTITION','SLURM_JOB_ACCOUNT')},
        'gpu':torch.cuda.get_device_name(),'preflight_sha256':sha(root/'preflight/COMPLETE.json'),
        'all_visual_parameter_sha256_before':visual_hash,'vision_frozen':True,'projector_frozen':True})
    initial=generate(model,p,items,rows);write(out/'generations_before.json',initial)
    baseline=evaluate(model,items,rows);write(out/'loss_before.json',baseline)
    optimizer=torch.optim.AdamW([v for n,v in trainable],lr=config['learning_rate'],weight_decay=0.)
    orders=epoch_orders(120,713,10,4) if stage=='full' else None
    if orders:
        named=[[rows[i]['sample_id'] for i in order] for order in orders]
        require(named==read(campaign/'B/TRAINING_ORDER.json'),'Original full training order changed')
        write(out/'TRAINING_ORDER.json',named)
    start=time.monotonic();passed=False;probe={}
    for step in range(1,config['max_steps']+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        order=orders[step-1] if orders else np.random.permutation(len(items))[:4]
        metrics={k:0. for k in METRICS}
        for j,index in enumerate(order):
            values=gpu_losses(model,items[index])
            if step==1 and j==0:
                probe=gradient_probe(model,values,'B')
                for v in probe.values():v['objective_weight']=.5;v['weighted_gradient_norm']=.5*v['lora_gradient_norm']
                write(out/'GRADIENT_PROBE.json',probe)
            require(torch.isfinite(values['total']),'Nonfinite training loss')
            (values['total']/4).backward()
            for k in METRICS:metrics[k]+=float(values[k].detach())/4
        norm=torch.nn.utils.clip_grad_norm_([v for n,v in trainable],1.)
        require(torch.isfinite(norm),'Nonfinite gradient');optimizer.step()
        record={'step':step,'elapsed_seconds':time.monotonic()-start,**metrics,'gradient_norm':float(norm)}
        with (out/'curve.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if step%5==0:print(stage,json.dumps(record),flush=True)
        if step%config['evaluate_every']==0 or step==config['max_steps']:
            measured=evaluate(model,items,rows);write(out/f'loss_step_{step:04d}.json',measured)
            if step%config['generate_every']!=0 and step!=config['max_steps']:continue
            outputs=generate(model,p,items,rows);write(out/f'generations_step_{step:04d}.json',outputs)
            scored=score(outputs)
            passed=scored['exact_full_output']==len(rows) and all(measured['means'][k]<.25*baseline['means'][k] for k in ('total','rationale','verdict'))
            print('generation',step,scored,'smoke_gate',passed,flush=True)
            if stage=='smoke' and passed:break
    require(tensor_digest(frozen)==visual_hash,'Frozen vision/projector bytes changed')
    require(all(not v.requires_grad and v.grad is None for n,v in frozen),'Visual gradients detected')
    model.save_pretrained(out/'adapter');p.save_pretrained(out/'adapter')
    write(out/'generations_final.json',outputs)
    zero_rows=[diagnostic_row(r,'horizon_0') for r in rows]
    zero=generate(model,p,[encode(p,r,dataset) for r in zero_rows],zero_rows,'horizon_0')
    write(out/'generations_horizon_0.json',zero)
    write(out/'INVALID_OR_INCONSISTENT.json',[x for x in outputs+zero if not x['verdict_correct'] or not x['format_valid'] or x['rationale_assessment']['status']!='supported_complete'])
    result={'passed':bool(passed) if stage=='smoke' else True,'exact_fit_criterion':bool(passed),'stage':stage,'steps':step,
        'before':baseline['means'],'after':measured['means'],'final_score':score(outputs),'zero_command_score':score(zero),
        'zero_correct_among_original_feasible':sum(r['target']['verdict']=='feasible' and z['predicted_verdict']=='infeasible' for r,z in zip(rows,zero)),
        'zero_flips_among_original_feasible':sum(r['target']['verdict']=='feasible' and g['predicted_verdict']=='feasible' and z['predicted_verdict']=='infeasible' for r,g,z in zip(rows,outputs,zero)),
        'gradient_probe':probe,'vision_and_projector_bitwise_unchanged':True,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,
        'elapsed_seconds':time.monotonic()-start,'scientific_generalization_claim':False}
    write(out/'RESULT.json',result);print(json.dumps(result),flush=True)
    if stage=='smoke':require(passed,'Bounded smoke failed; no full fit permitted')


if __name__=='__main__':
    stage=sys.argv[1];root=Path(sys.argv[2])
    try:
        if stage=='preflight':preflight(root)
        else:
            require(stage in ('smoke','full'),'Unknown stage');train(root,stage)
    except BaseException as exc:
        write(root/('STOP_'+stage+'.json'),{'error':type(exc).__name__,'message':str(exc),'job':os.environ.get('SLURM_JOB_ID')})
        raise
