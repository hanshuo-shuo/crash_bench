"""Step 2 ladder on Quest (docs/generative_verifier/LADDER_PROTOCOL.md).

One GPU stage. Part 1 fits AF on the step 1 16 rows through the step 1 overfit loop. Part 2 trains AF and AP on
the 512 training rows with the Phase 2 schedule and orders and scores the 192 validation rows at updates 128, 256
and 384; A and B come from the Phase 2 validation records. The rules frozen in config.json give the readings.
No test row is loaded.
"""
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
import torch
from .common import read,write,sha,require,MODEL_ID,UPSTREAM
from .fd1000_contract import validate,check_schedule
from .fd1000_metrics import summarize,mean_correct,clustered_difference
from .fd1000_run import TESTS,processor_for,vision_inputs,encode,input_digest,adapter_model,evaluate,update
from .fit_contract import overfit_units,arm_class
from .fit_run import overfit,release
from .ladder_contract import (configuration,answer,fit16_orders,comparison_reading,fit16_reading,RUNGS,PHASE2_FILES,STEP1_FILES,CONTROLS_FILES)

PHASE2_READING={'A':'direct_free','B':'facts_free'}


def context(root):
    receipt=read(root/'RECEIPT.json');config=read(root/'config.json')
    require(config==configuration() and sha(root/'config.json')==receipt['configuration_sha256'],'Frozen ladder configuration changed')
    require(os.environ['CB_CODE_COMMIT']==receipt['source_commit'],'Frozen ladder source differs')
    dataset=Path(receipt['dataset']);assets=Path(receipt['assets'])
    roots={name:Path(receipt[name+'_root']) for name in ('phase2','step1','controls')}
    for name,files in (('phase2',PHASE2_FILES),('step1',STEP1_FILES),('controls',CONTROLS_FILES)):
        require(set(receipt[name+'_files'])==set(files),'Evidence list differs: '+name)
        for f,digest in receipt[name+'_files'].items():require(sha(roots[name]/f)==digest,f'Evidence changed: {name} {f}')
    require(read(roots['phase2']/'config.json')==config['phase2_configuration'],'Phase 2 training configuration differs')
    require(all(read(roots[n]/'COMPLETE.json')['passed'] is True for n in ('step1','controls')),'Earlier diagnostic incomplete')
    require(sha(dataset/'DEV_ACCEPTANCE.json')==receipt['dataset_acceptance_sha256'],'Dataset acceptance changed')
    rows,acceptance=validate(dataset,mode='development',include_test=False)
    require(sorted({r['split'] for r in rows})==['train','validation'],'Unexpected splits loaded')
    require(sha(assets/'STAGED.json')==receipt['assets_stage_sha256'],'Model staging manifest changed')
    staged=read(assets/'STAGED.json');require(staged['model_id']==MODEL_ID,'Unexpected base model')
    for name,digest in staged['files'].items():require(sha(assets/'model'/name)==digest,'Staged base model changed')
    hashes={r['sample_id']:r['input_sha256'] for r in read(roots['phase2']/'preflight/ENCODINGS.json') if r['split'] in ('train','validation')}
    return receipt,config,dataset,assets,roots,rows,staged,hashes


def fit16(root,config,assets,processor,dataset,train_rows,hashes,step1):
    """Part 1: AF on the step 1 subset and sample order, through the step 1 overfit loop."""
    subset=overfit_units(train_rows);require(subset==read(step1/'SUBSET.json'),'The 16-row subset differs from step 1')
    plan=fit16_orders(subset['sample_ids'],config);previous=read(step1/'ORDERS.json')
    for seed,arms in plan.items():
        for arm,batches in arms.items():
            require([[s for s,k in b] for b in batches]==[[s for s,k in b] for b in previous[seed]['A']],'Sample order differs from step 1')
    write(root/'FIT16_ORDERS.json',plan)
    chosen=set(subset['sample_ids']);small=[r for r in train_rows if r['sample_id'] in chosen];runs={}
    for arm in config['fit16']['arms']:
        rung=config['rungs'][arm];items={}
        for r in small:
            items[(r['sample_id'],'facts')]=encode(processor,r,dataset,'facts',config['phase2_configuration']['max_length'],
                vision_inputs(processor,r,dataset),verify=True,answer=answer(r,arm),keys=rung['key_fields'])
        require(all(input_digest(v)==hashes[s] for (s,k),v in items.items()),'Fit input differs from the Phase 2 preflight')
        for seed in config['fit16']['seeds']:
            runs[f'{arm}_seed{seed}']=overfit(root,arm,seed,dict(config,overfit=config['fit16']),assets,processor,small,items,hashes,
                plan[str(seed)][arm],target=dict(reading=rung['reading'],kind='facts',prefix=None,loss=rung['loss']))
    return runs


def train512(root,arm,seed,config,assets,processor,validation,items,hashes,order):
    """Part 2: the Phase 2 schedule on the 512 training rows, greedy validation scoring at each evaluation update."""
    base=config['phase2_configuration'];spec=config['train512'];rung=config['rungs'][arm]
    out=root/'train512'/f'{arm}_seed{seed}';out.mkdir(parents=True)
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);torch.backends.cuda.matmul.allow_tf32=True
    model,targets=adapter_model(assets,base,seed);params=[(n,v) for n,v in model.named_parameters() if v.requires_grad]
    require(params and all('lora_' in n and 'visual' not in n and 'lm_head' not in n for n,v in params),'Unexpected trainable weights')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.config.use_cache=False
    optimizer=torch.optim.AdamW([v for _,v in params],lr=base['learning_rate'],weight_decay=base['weight_decay'])
    plan=[(rung['reading'],'facts',None,'greedy')];started=time.monotonic();checks=[]
    for step,batch in enumerate(order,1):
        totals=update(model,optimizer,items,batch,params,base)
        record=dict(step=step,seconds=time.monotonic()-started,**totals)
        with (out/'curve.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if step%10==0:print(arm,seed,json.dumps(record),flush=True)
        if step in spec['evaluation_updates']:
            scores=summarize(evaluate(model,processor,validation,items,plan,base,hashes,out/f'validation_step_{step}.jsonl'),base)[rung['reading']]
            model.save_pretrained(out/f'checkpoint_{step}');checks.append(dict(step=step,scores=scores));write(out/'VALIDATION_SCORES.json',checks)
            print(arm,seed,'step',step,'validation',scores['verdict_accuracy'],flush=True)
    require([c['step'] for c in checks]==spec['evaluation_updates'],'Missing validation checkpoint')
    write(out/'RESULT.json',dict(passed=True,arm=arm,seed=seed,steps=len(order),validation={c['step']:c['scores']['verdict_accuracy'] for c in checks},
        adapter_sha256={c['step']:sha(out/f'checkpoint_{c["step"]}/adapter_model.safetensors') for c in checks},
        trainable_parameters=sum(v.numel() for n,v in params),lora_targets=len(targets),elapsed_seconds=time.monotonic()-started,
        validation_rows_scored=len(validation),test_rows_loaded=0,api_calls=0))
    del model,optimizer,params;release()


def validation_records(root,roots,config,rung,seed,step,validation):
    if rung in PHASE2_READING:path=roots['phase2']/f'{rung}_seed{seed}/validation_step_{step}.jsonl';reading=PHASE2_READING[rung]
    else:path=root/'train512'/f'{rung}_seed{seed}/validation_step_{step}.jsonl';reading=config['rungs'][rung]['reading']
    records=[r for r in (json.loads(line) for line in path.read_text().splitlines()) if r['reading']==reading]
    require(len(records)==len(validation) and {r['sample_id'] for r in records}=={r['sample_id'] for r in validation},'Validation records incomplete: '+str(path))
    return records


def training_tail(root,roots,rung,seed):
    """Mean training losses over the last 32 updates (fit to the training rows under teacher forcing)."""
    path=(roots['phase2']/f'{rung}_seed{seed}' if rung in PHASE2_READING else root/'train512'/f'{rung}_seed{seed}')/'curve.jsonl'
    tail=[json.loads(line) for line in path.read_text().splitlines()][-32:]
    return {k:sum(r[k] for r in tail)/len(tail) for k in ('direct_verdict','facts_verdict','facts_key_values') if all(k in r for r in tail)}


def report(root,config,runs16,roots,validation):
    seeds=config['train512']['seeds'];updates=config['train512']['evaluation_updates'];base=config['phase2_configuration']
    fit_class=arm_class([runs16[f'AF_seed{s}']['classification'] for s in config['fit16']['seeds']]);fit_read=fit16_reading(fit_class)
    rungs={};correct={};groups=None
    for rung in RUNGS:
        reading=PHASE2_READING.get(rung) or config['rungs'][rung]['reading'];rungs[rung]={}
        for step in updates:
            per_seed=[validation_records(root,roots,config,rung,s,step,validation) for s in seeds]
            scores=[summarize(r,base)[reading] for r in per_seed]
            entry=dict({k:sum(x[k] for x in scores)/len(scores) for k in ('verdict_accuracy','unit_all_correct','rule_flip_consistency','false_feasible_rate')},
                seeds=[x['verdict_accuracy'] for x in scores])
            if 'facts' in scores[0]:entry['protected_accuracy']=sum(x['facts']['protected_accuracy'] for x in scores)/len(scores)
            rungs[rung][str(step)]=entry
            if step==config['primary_update']:correct[rung],groups=mean_correct(per_seed)
        rungs[rung]['training_last32']=[training_tail(root,roots,rung,s) for s in seeds]
    comparisons={}
    for treatment,control in config['comparisons']:
        key=f'{treatment}-{control}';result=clustered_difference(correct[control],correct[treatment],groups)
        reading=comparison_reading(result['source_bootstrap_95pct'])
        comparisons[key]=dict(result,reading=reading,meaning=config['rules']['comparisons'][key][reading])
    out=dict(fit16=dict(AF=fit_class,reading=fit_read,meaning=config['rules']['fit16']['readings'][fit_read],
            runs={k:{m:v[m] for m in ('classification','final_accuracy','last_epoch_fit_loss')}|dict(accuracy_by_update={c['step']:c['verdict_accuracy'] for c in v['checks']})
                for k,v in runs16.items()}),
        validation=rungs,comparisons=comparisons,primary_update=config['primary_update'],
        scope='Exploratory validation on V4 development data (14 source groups); the Phase 2 locked test is never scored again')
    write(root/'REPORT.json',out)
    lines=['# Step 2 ladder','',f'- 16-row AF fit: **{fit_class}**, reading **{fit_read}**. {out["fit16"]["meaning"]}','',
        '| rung | validation accuracy at 128 / 256 / 384 (mean of 3 seeds) | seeds at 384 | unit all-correct at 384 | false-feasible at 384 |','|---|---|---|---:|---:|']
    for rung in RUNGS:
        r=rungs[rung];final=r[str(updates[-1])]
        lines.append(f'| {rung} | '+' / '.join(f'{r[str(u)]["verdict_accuracy"]:.3f}' for u in updates)+' | '+' / '.join(f'{v:.3f}' for v in final['seeds'])+
            f' | {final["unit_all_correct"]:.3f} | {final["false_feasible_rate"]:.3f} |')
    lines+=['','| comparison at update 384 | difference | source-bootstrap 95% CI | sign-flip p | reading |','|---|---:|---|---:|---|']
    for key,c in comparisons.items():
        low,high=c['source_bootstrap_95pct'];lines.append(f'| {key} | {c["mean_difference"]:+.3f} | [{low:+.3f}, {high:+.3f}] | {c["sign_flip"]["p_two_sided"]:.4f} | {c["reading"]}: {c["meaning"]} |')
    lines+=['',out['scope']];(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return out


def run(root):
    receipt,config,dataset,assets,roots,rows,staged,hashes=context(root)
    require(torch.cuda.is_available(),'Quest GPU required');started=time.monotonic()
    subprocess.run([sys.executable,'-m','unittest',*TESTS,'verifier_training.tests.test_fit_contract','verifier_training.tests.test_control_contract',
        'verifier_training.tests.test_ladder_contract'],check=True)
    train_rows=[r for r in rows if r['split']=='train'];validation=[r for r in rows if r['split']=='validation']
    require(len(train_rows)==512 and len(validation)==192,'Expected 512 training and 192 validation rows')
    processor=processor_for(assets);runs16=fit16(root,config,assets,processor,dataset,train_rows,hashes,roots['step1'])
    base=config['phase2_configuration'];orders=read(roots['phase2']/'ORDERS.json');train_ids=[r['sample_id'] for r in train_rows]
    items={arm:{} for arm in config['train512']['arms']}
    for row in train_rows+validation:
        vision=vision_inputs(processor,row,dataset)
        for arm in items:
            items[arm][(row['sample_id'],'facts')]=encode(processor,row,dataset,'facts',base['max_length'],vision,
                answer=answer(row,arm),keys=config['rungs'][arm]['key_fields'])
    require(all(input_digest(v)==hashes[s] for arm in items for (s,k),v in items[arm].items()),'Input differs from the Phase 2 preflight')
    for arm in config['train512']['arms']:
        for seed in config['train512']['seeds']:
            order=orders[str(seed)]['B'];check_schedule(order,train_ids,'B',base)
            require([[s for s,k in b] for b in order]==[[s for s,k in b] for b in orders[str(seed)]['A']],'Phase 2 arms differ in sample order')
            train512(root,arm,seed,config,assets,processor,validation,items[arm],hashes,order)
    out=report(root,config,runs16,roots,validation)
    write(root/'COMPLETE.json',dict(passed=True,fit16=out['fit16']['reading'],comparisons={k:v['reading'] for k,v in out['comparisons'].items()},
        source_commit=receipt['source_commit'],upstream_commit=UPSTREAM,config_sha256=sha(root/'config.json'),report_sha256=sha(root/'REPORT.json'),
        model_id=MODEL_ID,model_revision=staged['model_revision'],gpu=torch.cuda.get_device_name(),
        packages={k:metadata.version(k) for k in ('torch','transformers','peft','accelerate','numpy','pillow')},
        slurm={k:v for k,v in os.environ.items() if k in ('SLURM_JOB_ID','SLURM_JOB_NODELIST','SLURM_JOB_PARTITION','SLURM_JOB_ACCOUNT')},
        elapsed_seconds=time.monotonic()-started,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
        splits_loaded=sorted({r['split'] for r in rows}),validation_rows_scored=len(validation),test_rows_loaded=0,api_calls=0))


if __name__=='__main__':
    root=Path(sys.argv[1])
    try:run(root)
    except BaseException as error:
        write(root/'STOP_ladder.json',dict(error=type(error).__name__,message=str(error),slurm_job=os.environ.get('SLURM_JOB_ID')))
        raise
