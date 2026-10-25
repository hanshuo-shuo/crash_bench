"""V5 fit diagnostic on Quest (docs/generative_verifier/V5_FIT_PROTOCOL.md).

Stages: fit SIZE (GPU; B from scratch on the 16- or 64-row set for three seeds, greedy fit checks, teacher-forced
robot-side facts on the held-out training rows; the 16-row job also reads the S3 B adapter there) and report (CPU;
the frozen rules give the decision). Only training rows are read.
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
from peft import PeftModel
from .common import read,write,sha,require,MODEL_ID,UPSTREAM
from .fd1000_run import processor_for,vision_inputs,input_digest,base_model,adapter_model,update
from .fit_run import release
from .v5_contract import validate
from .v5_fit_contract import (configuration,subsets,orders,classify_run,size_class,decision,fact_lookup,S3_FILES,SIDES)
from .v5_run import item,by_image,evaluate
from . import v5_metrics as M

TESTS=['verifier_training.tests.test_v5_fit_contract','verifier_training.tests.test_v5_contract','verifier_training.tests.test_v5_metrics']


def context(root):
    receipt=read(root/'RECEIPT.json');config=read(root/'config.json')
    require(config==configuration() and sha(root/'config.json')==receipt['configuration_sha256'],'Frozen diagnostic configuration changed')
    require(os.environ['CB_CODE_COMMIT']==receipt['source_commit'],'Frozen diagnostic source differs')
    s3=Path(receipt['s3_root']);require(set(receipt['s3_files'])==set(S3_FILES),'S3 evidence list differs')
    for name,digest in receipt['s3_files'].items():require(sha(s3/name)==digest,'S3 evidence changed: '+name)
    require(read(s3/'config.json')==config['s3_configuration'],'S3 training configuration differs')
    dataset=Path(receipt['dataset']);assets=Path(receipt['assets'])
    train=[r for r in validate(dataset,include_test=False) if r['split']=='train'];require(len(train)==1792,'Expected the 1,792 training rows')
    require(sha(assets/'STAGED.json')==receipt['assets_stage_sha256'],'Model staging manifest changed')
    staged=read(assets/'STAGED.json');require(staged['model_id']==MODEL_ID,'Unexpected base model')
    for name,digest in staged['files'].items():require(sha(assets/'model'/name)==digest,'Staged base model changed')
    parts=subsets(train,config);require(parts==read(root/'SUBSETS.json') and sha(root/'SUBSETS.json')==receipt['subsets_sha256'],'Fit sets differ from launch')
    plan={size:orders(parts[size]['sample_ids'],size,config) for size in config['sizes']}
    require(plan==read(root/'ORDERS.json') and sha(root/'ORDERS.json')==receipt['orders_sha256'],'Sample orders differ from launch')
    hashes={r['sample_id']:r['input_sha256'] for r in read(s3/'preflight/ENCODINGS.json') if r['split']=='train'}
    lookup=read(s3/'LOOKUP.json')['predictions']['train']
    return receipt,config,dataset,assets,s3,train,staged,parts,plan,hashes,lookup


def side_readout(model,processor,rows,items,guesses):
    """Teacher-forced robot-side facts: P(yes) vs P(no) at each value token given the gold prefix, and the gold-token CE."""
    model.eval();tokens={w:processor.tokenizer.encode(' '+w,add_special_tokens=False) for w in ('yes','no')}
    require(all(len(v)==1 for v in tokens.values()),'Robot-side values are not single tokens');yes,no=tokens['yes'][0],tokens['no'][0]
    records=[]
    with torch.no_grad():
        for row in rows:
            it=items[(row['sample_id'],'facts')];ids=it['fields']['input_ids'][0]
            logits=model(**{k:v.to('cuda') for k,v in it['fields'].items()},use_cache=False).logits[0].float()
            record=dict(sample_id=row['sample_id'],image_id=row['metadata']['image_id'],source_id=row['metadata']['source_id'])
            for name in SIDES:
                at=it['detail']['key_'+name][0].nonzero().flatten();require(len(at)==1,'Robot-side value is not one token')
                p=int(at[0]);gold=int(ids[p]);require(gold in (yes,no),'Unexpected robot-side token')
                pair=logits[p-1][[yes,no]].softmax(0);truth=row['target']['facts'][name]
                record[name]=dict(truth=truth,p_yes=float(pair[0]),predicted='yes' if pair[0]>=.5 else 'no',
                    gold_ce=float(-logits[p-1].log_softmax(-1)[gold]),lookup=guesses[name](row))
            records.append(record)
    return records


def readout_summary(records):
    n=len(records);result=dict(rows=n,images=len({r['image_id'] for r in records}))
    for name in SIDES:
        part=[r[name] for r in records]
        result[name]=dict(accuracy=sum(x['predicted']==x['truth'] for x in part)/n,lookup_accuracy=sum(x['lookup']==x['truth'] for x in part)/n,
            mean_gold_ce=sum(x['gold_ce'] for x in part)/n,predicted_yes_rate=sum(x['predicted']=='yes' for x in part)/n,
            true_yes_rate=sum(x['truth']=='yes' for x in part)/n,
            auroc=M.auroc([x['truth']=='yes' for x in part],[x['p_yes'] for x in part]))
        result[name]['gain_over_lookup']=result[name]['accuracy']-result[name]['lookup_accuracy']
    result['both_sides_accuracy']=sum(all(r[s]['predicted']==r[s]['truth'] for s in SIDES) for r in records)/n
    result['both_sides_lookup_accuracy']=sum(all(r[s]['lookup']==r[s]['truth'] for s in SIDES) for r in records)/n
    result['mean_gain_over_lookup']=sum(result[s]['gain_over_lookup'] for s in SIDES)/len(SIDES)
    result['mean_gold_ce']=sum(result[s]['mean_gold_ce'] for s in SIDES)/len(SIDES)
    gains=np.asarray([np.mean([(r[s]['predicted']==r[s]['truth'])-(r[s]['lookup']==r[s]['truth']) for s in SIDES]) for r in records])
    result['gain_image_bootstrap_95pct']=M.cluster_interval(gains,[r['image_id'] for r in records])
    return result


def fit_run(root,size,seed,config,assets,processor,fit_rows,held,items,hashes,lookup,guesses,order):
    """Fresh LoRA trained on one fit set with the S3 update code and hyperparameters."""
    base=config['s3_configuration'];spec=config['sizes'][size];out=root/f'fit{size}'/f'B_seed{seed}';out.mkdir(parents=True)
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);torch.backends.cuda.matmul.allow_tf32=True
    model,targets=adapter_model(assets,base,seed);params=[(n,v) for n,v in model.named_parameters() if v.requires_grad]
    require(params and all('lora_' in n and 'visual' not in n and 'lm_head' not in n for n,v in params),'Unexpected trainable weights')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.config.use_cache=False
    optimizer=torch.optim.AdamW([v for _,v in params],lr=base['learning_rate'],weight_decay=base['weight_decay'])
    started=time.monotonic();checks=[]
    for step,batch in enumerate(order,1):
        totals=update(model,optimizer,items,batch,params,base)
        with (out/'curve.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,seconds=time.monotonic()-started,**totals))+'\n')
        if step in spec['evaluation_updates']:
            scores=M.summarize(evaluate(model,processor,fit_rows,items,'facts_free','facts',base,hashes,lookup,out/f'fit_step_{step}.jsonl'))['facts_free']
            checks.append(dict(step=step,verdict_accuracy=scores['verdict_accuracy'],both_sides_accuracy=scores['facts']['both_sides_accuracy'],
                target_robot_side_accuracy=scores['facts']['target_robot_side_accuracy'],goal_robot_side_accuracy=scores['facts']['goal_robot_side_accuracy'],
                format_valid=scores['format_valid'],scores=scores))
            print(size,seed,'step',step,'verdict',scores['verdict_accuracy'],'both sides',scores['facts']['both_sides_accuracy'],flush=True)
    require(checks and checks[-1]['step']==spec['optimizer_updates'],'Final update was not evaluated')
    fitted=side_readout(model,processor,fit_rows,items,guesses);transfer=side_readout(model,processor,held,items,guesses)
    for name,records in (('fit_readout',fitted),('held_out_readout',transfer)):
        with (out/f'{name}.jsonl').open('w') as f:f.writelines(json.dumps(r)+'\n' for r in records)
    fit_summary=readout_summary(fitted);held_summary=readout_summary(transfer);final=checks[-1]
    model.save_pretrained(out/'final_adapter')
    result=dict(passed=True,size=size,seed=seed,rows=len(fit_rows),steps=len(order),final_verdict_accuracy=final['verdict_accuracy'],
        final_both_sides_accuracy=final['both_sides_accuracy'],fit_loss=fit_summary['mean_gold_ce'],
        classification=classify_run(final['verdict_accuracy'],final['both_sides_accuracy'],fit_summary['mean_gold_ce'],config['thresholds']),
        checks=[{k:v for k,v in c.items() if k!='scores'} for c in checks],fit_readout=fit_summary,held_out_readout=held_summary,
        adapter_sha256=sha(out/'final_adapter/adapter_model.safetensors'),lora_targets=len(targets),
        trainable_parameters=sum(v.numel() for n,v in params),elapsed_seconds=time.monotonic()-started,
        peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
    write(out/'RESULT.json',result);del model,optimizer,params;release();return result


def reference(root,s3,assets,processor,held,items,guesses):
    """The S3 B adapter read on the held-out rows (it saw them once during S3 training)."""
    done=read(s3/'B_seed713/RESULT.json');require(sha(s3/'B_seed713/selected_adapter/adapter_model.safetensors')==done['adapter_sha256'],'S3 B adapter changed')
    torch.manual_seed(713);model=PeftModel.from_pretrained(base_model(assets),s3/'B_seed713/selected_adapter',is_trainable=False);model.eval()
    records=side_readout(model,processor,held,items,guesses)
    with (root/'fit16'/'s3_reference_held_out_readout.jsonl').open('w') as f:f.writelines(json.dumps(r)+'\n' for r in records)
    summary=dict(readout_summary(records),adapter_sha256=done['adapter_sha256'],selected_step=done['selected_step'])
    del model;release();return summary


def fit(root,size):
    receipt,config,dataset,assets,s3,train,staged,parts,plan,hashes,lookup=context(root)
    require(size in config['sizes'],'Unplanned fit size');require(torch.cuda.is_available(),'Quest GPU required');started=time.monotonic()
    (root/f'fit{size}').mkdir()
    subprocess.run([sys.executable,'-m','unittest',*TESTS],check=True)
    processor=processor_for(assets);by_id={r['sample_id']:r for r in train}
    fit_rows=[by_id[s] for s in parts[size]['sample_ids']];held=[by_id[s] for s in parts['held_out']['sample_ids']]
    require(not {r['metadata']['image_id'] for r in fit_rows}&{r['metadata']['image_id'] for r in held},'A fit image is in the held-out rows')
    items={}
    for image,members in by_image(fit_rows+held).items():
        vision=vision_inputs(processor,members[0],dataset)
        for row in members:items[(row['sample_id'],'facts')]=item(processor,row,dataset,'facts',config['s3_configuration'],vision)
    require(all(input_digest(v)==hashes[s] for (s,k),v in items.items()),'Training input differs from the S3 preflight')
    guesses={name:fact_lookup(train,name) for name in SIDES};runs={}
    for seed in config['seeds']:
        runs[f'B_seed{seed}']=fit_run(root,size,seed,config,assets,processor,fit_rows,held,items,hashes,lookup,guesses,plan[size][str(seed)])
        write(root/f'fit{size}'/'SUMMARY.json',dict(runs=runs))
    summary=dict(passed=True,size=size,runs=runs,s3_reference=reference(root,s3,assets,processor,held,items,guesses) if size=='16' else None,
        source_commit=receipt['source_commit'],model_id=MODEL_ID,model_revision=staged['model_revision'],gpu=torch.cuda.get_device_name(),
        packages={k:metadata.version(k) for k in ('torch','transformers','peft','accelerate','numpy','pillow')},
        slurm={k:v for k,v in os.environ.items() if k in ('SLURM_JOB_ID','SLURM_JOB_NODELIST','SLURM_JOB_PARTITION','SLURM_JOB_ACCOUNT')},
        elapsed_seconds=time.monotonic()-started,splits_loaded=['train'],validation_rows_scored=0,test_rows_loaded=0,api_calls=0)
    write(root/f'fit{size}'/'SUMMARY.json',summary)


def report(root):
    """CPU: the frozen rules over both fit jobs."""
    receipt,config,dataset,assets,s3,train,staged,parts,plan,hashes,lookup=context(root);t=config['thresholds']
    summaries={size:read(root/f'fit{size}'/'SUMMARY.json') for size in config['sizes']}
    require(all(s['passed'] and s['test_rows_loaded']==0 for s in summaries.values()),'A fit job did not finish')
    classes={size:size_class([s['runs'][f'B_seed{seed}']['classification'] for seed in config['seeds']]) for size,s in summaries.items()}
    gain=sum(summaries['64']['runs'][f'B_seed{seed}']['held_out_readout']['mean_gain_over_lookup'] for seed in config['seeds'])/len(config['seeds'])
    outcome=decision(classes['16'],classes['64'],gain,t)
    runs={f'fit{size}_{k}':{m:v[m] for m in ('classification','final_verdict_accuracy','final_both_sides_accuracy','fit_loss')}|
        dict(held_out={m:v['held_out_readout'][m] for m in ('both_sides_accuracy','mean_gain_over_lookup','gain_image_bootstrap_95pct')}|
            {name:{m:v['held_out_readout'][name][m] for m in ('accuracy','lookup_accuracy','auroc','predicted_yes_rate')} for name in SIDES},
            checks=v['checks']) for size,s in summaries.items() for k,v in s['runs'].items()}
    reference=summaries['16']['s3_reference']
    result=dict(passed=True,decision=outcome,meaning=config['rules']['decisions'][outcome],size_classes=classes,transfer_gain_64=gain,
        transfer_seen=gain>=t['transfer_gain'],runs=runs,s3_reference_held_out=reference,
        subsets={k:{m:v[m] for m in ('images','sources','labels','sides')}|dict(rows=len(v['sample_ids'])) for k,v in parts.items()},
        rules=config['rules'],thresholds=t,scope='Training rows only: fit sets and held-out training images; validation and test are not scored')
    write(root/'REPORT.json',result)
    lines=['# V5 fit diagnostic','',f'Decision: **{outcome}**. {result["meaning"]}','',
        f'- 16-row fit: {classes["16"]}; 64-row fit: {classes["64"]}; held-out gain over the lookup (64-row runs): {gain:+.3f}','',
        '| run | class | verdict | both sides | fit loss | held-out both sides | held-out gain [95%] |','|---|---|---:|---:|---:|---:|---|']
    for k,v in runs.items():
        h=v['held_out'];ci=h['gain_image_bootstrap_95pct']
        lines.append(f'| {k} | {v["classification"]} | {v["final_verdict_accuracy"]:.3f} | {v["final_both_sides_accuracy"]:.3f} | {v["fit_loss"]:.3f} | '
            f'{h["both_sides_accuracy"]:.3f} | {h["mean_gain_over_lookup"]:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] |')
    lines+=['',f'S3 B adapter on the held-out rows: both sides {reference["both_sides_accuracy"]:.3f}, gain {reference["mean_gain_over_lookup"]:+.3f}; '
        f'lookup both sides {reference["both_sides_lookup_accuracy"]:.3f}.','',result['scope']]
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(root/'COMPLETE.json',dict(passed=True,decision=outcome,size_classes=classes,transfer_gain_64=gain,source_commit=receipt['source_commit'],
        upstream_commit=UPSTREAM,s3_root=str(s3),config_sha256=sha(root/'config.json'),report_sha256=sha(root/'REPORT.json'),
        validation_rows_scored=0,test_rows_loaded=0,api_calls=0,slurm_job=os.environ.get('SLURM_JOB_ID')))
    print(outcome,json.dumps(dict(classes=classes,gain=gain)),flush=True)


if __name__=='__main__':
    stage=sys.argv[1];root=Path(sys.argv[2]);label=stage+('_'+sys.argv[3] if len(sys.argv)>3 else '')
    try:
        if stage=='fit':fit(root,sys.argv[3])
        elif stage=='report':report(root)
        else:raise ValueError('Unknown stage')
    except BaseException as error:
        write(root/('STOP_'+label+'.json'),dict(error=type(error).__name__,message=str(error),slurm_job=os.environ.get('SLURM_JOB_ID')))
        raise
