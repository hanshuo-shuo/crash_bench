"""Step 1 fit diagnostic on Quest (docs/generative_verifier/STEP1_PROTOCOL.md).

One GPU stage: 1a scores the saved Phase 2 A adapters on the training rows, 1b trains
A and B from scratch on 16 training rows through the Phase 2 update code, then the
rules frozen in config.json give the decision. No test row is loaded.
"""
import gc
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
from peft import PeftModel
from .common import read,write,sha,require,MODEL_ID,UPSTREAM
from .fd1000_contract import validate,readings,KINDS
from .fd1000_metrics import summarize
from .fd1000_run import TESTS,processor_for,vision_inputs,encode,input_digest,base_model,adapter_model,evaluate,update
from .fit_contract import (configuration,phase2_files,overfit_units,overfit_orders,reading,last_epoch_loss,classify_run,
    arm_class,train_class,decision,FIT_LOSS,SEEDS)


def context(root):
    receipt=read(root/'RECEIPT.json');config=read(root/'config.json')
    require(config==configuration() and sha(root/'config.json')==receipt['configuration_sha256'],'Frozen diagnostic configuration changed')
    require(os.environ['CB_CODE_COMMIT']==receipt['source_commit'],'Frozen diagnostic source differs')
    dataset=Path(receipt['dataset']);assets=Path(receipt['assets']);phase2=Path(receipt['phase2_root'])
    require(set(receipt['phase2_files'])==set(phase2_files()),'Phase 2 evidence list differs')
    for name,digest in receipt['phase2_files'].items():require(sha(phase2/name)==digest,'Phase 2 evidence changed: '+name)
    require(read(phase2/'config.json')==config['phase2_configuration'],'Phase 2 training configuration differs')
    require(sha(dataset/'DEV_ACCEPTANCE.json')==receipt['dataset_acceptance_sha256'],'Dataset acceptance changed')
    rows,acceptance=validate(dataset,mode='development',include_test=False)
    train_rows=[r for r in rows if r['split']=='train'];require(len(train_rows)==512,'Expected the 512 V4 training rows')
    require(sha(assets/'STAGED.json')==receipt['assets_stage_sha256'],'Model staging manifest changed')
    staged=read(assets/'STAGED.json');require(staged['model_id']==MODEL_ID,'Unexpected base model')
    for name,digest in staged['files'].items():require(sha(assets/'model'/name)==digest,'Staged base model changed')
    hashes={r['sample_id']:r['input_sha256'] for r in read(phase2/'preflight/ENCODINGS.json') if r['split']=='train'}
    return receipt,config,dataset,assets,phase2,train_rows,staged,hashes


def release():
    gc.collect();torch.cuda.empty_cache()


def train_evaluation(root,config,assets,phase2,processor,rows,items,hashes):
    """1a: greedy direct readings of the saved A adapters on every training row."""
    spec=config['train_evaluation'];base=config['phase2_configuration'];out=root/'train_evaluation';out.mkdir();result={}
    for seed in spec['seeds']:
        for step in spec['checkpoints']:
            started=time.monotonic();torch.manual_seed(seed)
            model=PeftModel.from_pretrained(base_model(assets),phase2/f'A_seed{seed}/checkpoint_{step}',is_trainable=False);model.eval()
            records=evaluate(model,processor,rows,items,[(spec['reading'],'direct',None,'greedy')],base,hashes,out/f'A_seed{seed}_step{step}.jsonl')
            scores=summarize(records,base)[spec['reading']]
            truth=np.asarray([r['truth'] for r in records]);p=np.asarray([r['p_feasible'] if r['p_feasible'] is not None else .5 for r in records])
            scores.update(mean_gold_verdict_nll=float(-np.log(np.clip(np.where(truth==1,p,1-p),1e-12,1)).mean()),
                predicted_feasible_rate=sum(r['predicted']==1 for r in records)/len(records),elapsed_seconds=time.monotonic()-started)
            result[f'A_seed{seed}_step{step}']=scores;write(out/'SUMMARY.json',result)
            print('train evaluation',seed,step,json.dumps({k:scores[k] for k in ('verdict_accuracy','auroc_invalid_neutral','predicted_feasible_rate')}),flush=True)
            del model,records;release()
    return result


def overfit(root,arm,seed,config,assets,processor,rows,items,hashes,order):
    """1b: fresh LoRA trained on the 16-row subset with the Phase 2 update code and hyperparameters."""
    base=config['phase2_configuration'];spec=config['overfit'];out=root/'overfit'/f'{arm}_seed{seed}';out.mkdir(parents=True)
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);torch.backends.cuda.matmul.allow_tf32=True
    model,targets=adapter_model(assets,base,seed);params=[(n,v) for n,v in model.named_parameters() if v.requires_grad]
    require(params and all('lora_' in n and 'visual' not in n and 'lm_head' not in n for n,v in params),'Unexpected trainable weights')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.config.use_cache=False
    optimizer=torch.optim.AdamW([v for _,v in params],lr=base['learning_rate'],weight_decay=base['weight_decay'])
    name=reading(arm);kind,prefix=readings(arm)[name];plan=[(name,kind,prefix,'greedy')]
    started=time.monotonic();curve=[];checks=[]
    for step,batch in enumerate(order,1):
        totals=update(model,optimizer,items,batch,params,base)
        record=dict(step=step,seconds=time.monotonic()-started,**totals);curve.append(record)
        with (out/'curve.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if step in spec['evaluation_updates']:
            scores=summarize(evaluate(model,processor,rows,items,plan,base,hashes,out/f'eval_step_{step}.jsonl'),base)[name]
            checks.append(dict(step=step,verdict_accuracy=scores['verdict_accuracy'],scores=scores))
            print(arm,seed,'step',step,'accuracy',scores['verdict_accuracy'],FIT_LOSS[arm],record[FIT_LOSS[arm]],flush=True)
    require(checks and checks[-1]['step']==spec['optimizer_updates'],'Final update was not evaluated')
    loss=last_epoch_loss(curve,arm,config);final=checks[-1]['verdict_accuracy']
    model.save_pretrained(out/'final_adapter')
    result=dict(passed=True,arm=arm,seed=seed,rows=len(rows),steps=len(curve),reading=name,final_accuracy=final,
        fit_loss_name=FIT_LOSS[arm],last_epoch_fit_loss=loss,classification=classify_run(final,loss,config['thresholds']),
        checks=checks,adapter_sha256=sha(out/'final_adapter/adapter_model.safetensors'),trainable_parameters=sum(v.numel() for n,v in params),
        lora_targets=len(targets),elapsed_seconds=time.monotonic()-started)
    write(out/'RESULT.json',result);del model,optimizer,params;release();return result


def phase2_curves(phase2):
    """Context only: Phase 2 training losses over the last 32 updates of every run."""
    result={}
    for arm in 'ABC':
        for seed in SEEDS:
            tail=[json.loads(line) for line in (phase2/f'{arm}_seed{seed}/curve.jsonl').read_text().splitlines()][-32:]
            mean=lambda key:(sum(r[key] for r in tail if key in r)/sum(key in r for r in tail)) if any(key in r for r in tail) else None
            result[f'{arm}_seed{seed}']={k:mean(k) for k in ('direct_verdict','facts_verdict','facts_key_values','gradient_norm')}
    return result


def report(root,config,train,runs,subset,curves):
    t=config['thresholds'];a512=train_class([train[f'A_seed{s}_step384']['verdict_accuracy'] for s in config['train_evaluation']['seeds']],t)
    arms={arm:arm_class([runs[f'{arm}_seed{s}']['classification'] for s in config['overfit']['seeds']]) for arm in config['overfit']['arms']}
    outcome=decision(arms['A'],arms['B'],a512)
    result=dict(a_train_512=a512,overfit_16=arms,decision=outcome,meaning=config['rules']['decisions'][outcome],
        train_evaluation={k:{m:v[m] for m in ('verdict_accuracy','auroc_invalid_neutral','predicted_feasible_rate','mean_gold_verdict_nll',
            'unit_all_correct','format_valid','scored_probabilities')} for k,v in train.items()},
        overfit_runs={k:{m:v[m] for m in ('classification','final_accuracy','last_epoch_fit_loss','fit_loss_name')}|
            dict(accuracy_by_update={c['step']:c['verdict_accuracy'] for c in v['checks']}) for k,v in runs.items()},
        subset=subset,phase2_last32_training_losses=curves,rules=config['rules'],thresholds=t,
        scope='Training-set and 16-row fit diagnostic on V4 development data; validation and test rows are not scored')
    write(root/'REPORT.json',result)
    lines=['# Step 1 fit diagnostic','',f'Decision: **{outcome}**. {result["meaning"]}','',
        f'- A on the 512 training rows (update 384): {a512}',f'- 16-row overfit: A {arms["A"]}, B {arms["B"]}','',
        '| A adapter | train accuracy | P(1) AUROC | predicted feasible | gold-verdict NLL |','|---|---:|---:|---:|---:|']
    for k,v in result['train_evaluation'].items():
        auc=v['auroc_invalid_neutral'];lines.append(f'| {k} | {v["verdict_accuracy"]:.3f} | {"-" if auc is None else f"{auc:.3f}"} | '
            f'{v["predicted_feasible_rate"]:.3f} | {v["mean_gold_verdict_nll"]:.3f} |')
    lines+=['','| 16-row run | class | final accuracy | last-epoch fit loss | accuracy by update |','|---|---|---:|---:|---|']
    for k,v in result['overfit_runs'].items():
        lines.append(f'| {k} | {v["classification"]} | {v["final_accuracy"]:.3f} | {v["last_epoch_fit_loss"]:.4f} ({v["fit_loss_name"]}) | '
            +', '.join(f'{s}: {a:.3f}' for s,a in v['accuracy_by_update'].items())+' |')
    lines+=['',f'Subset units: {", ".join(subset["units"])}; labels {subset["labels"]}.','',result['scope']]
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


def run(root):
    receipt,config,dataset,assets,phase2,rows,staged,hashes=context(root)
    require(torch.cuda.is_available(),'Quest GPU required');started=time.monotonic()
    subprocess.run([sys.executable,'-m','unittest',*TESTS,'verifier_training.tests.test_fit_contract'],check=True)
    base=config['phase2_configuration'];processor=processor_for(assets)
    subset=overfit_units(rows);orders=overfit_orders(subset['sample_ids'],config)
    write(root/'SUBSET.json',subset);write(root/'ORDERS.json',orders)
    chosen=set(subset['sample_ids']);small=[r for r in rows if r['sample_id'] in chosen];items={}
    for row in rows:
        vision=vision_inputs(processor,row,dataset)
        for kind in (KINDS if row['sample_id'] in chosen else ('direct',)):
            items[(row['sample_id'],kind)]=encode(processor,row,dataset,kind,base['max_length'],vision)
    require(all(input_digest(v)==hashes[s] for (s,k),v in items.items()),'Training input differs from the Phase 2 preflight')
    train=train_evaluation(root,config,assets,phase2,processor,rows,items,hashes);runs={}
    for arm in config['overfit']['arms']:
        for seed in config['overfit']['seeds']:
            runs[f'{arm}_seed{seed}']=overfit(root,arm,seed,config,assets,processor,small,items,hashes,orders[str(seed)][arm])
    result=report(root,config,train,runs,subset,phase2_curves(phase2))
    write(root/'COMPLETE.json',dict(passed=True,decision=result['decision'],a_train_512=result['a_train_512'],overfit_16=result['overfit_16'],
        source_commit=receipt['source_commit'],upstream_commit=UPSTREAM,phase2_root=str(phase2),config_sha256=sha(root/'config.json'),
        subset_sha256=sha(root/'SUBSET.json'),orders_sha256=sha(root/'ORDERS.json'),report_sha256=sha(root/'REPORT.json'),
        model_id=MODEL_ID,model_revision=staged['model_revision'],gpu=torch.cuda.get_device_name(),
        packages={k:metadata.version(k) for k in ('torch','transformers','peft','accelerate','numpy','pillow')},
        slurm={k:v for k,v in os.environ.items() if k in ('SLURM_JOB_ID','SLURM_JOB_NODELIST','SLURM_JOB_PARTITION','SLURM_JOB_ACCOUNT')},
        elapsed_seconds=time.monotonic()-started,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
        splits_loaded=sorted({r['split'] for r in rows}),validation_rows_scored=0,test_rows_loaded=0,api_calls=0))


if __name__=='__main__':
    root=Path(sys.argv[1])
    try:run(root)
    except BaseException as error:
        write(root/'STOP_fit.json',dict(error=type(error).__name__,message=str(error),slurm_job=os.environ.get('SLURM_JOB_ID')))
        raise
