"""Single-pathway label controls on Quest (docs/generative_verifier/CONTROLS_PROTOCOL.md).

One GPU stage: t, s and AP are trained from scratch on the step 1 16-row subset through the
step 1 overfit loop (Phase 2 encode, losses and update), then the rules frozen in config.json give
the readings. No test row is loaded.
"""
import importlib.metadata as metadata
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from .common import read,write,sha,require,MODEL_ID,UPSTREAM
from .fd1000_contract import validate,assistant_target
from .fd1000_run import TESTS,processor_for,vision_inputs,encode,input_digest
from .fit_contract import overfit_units,arm_class
from .fit_run import overfit
from .control_contract import (configuration,ARMS,KIND,PHASE2_FILES,STEP1_FILES,answer,relabel,check_labels,orders,
    pathway_reading,ap_reading)


def context(root):
    receipt=read(root/'RECEIPT.json');config=read(root/'config.json')
    require(config==configuration() and sha(root/'config.json')==receipt['configuration_sha256'],'Frozen control configuration changed')
    require(os.environ['CB_CODE_COMMIT']==receipt['source_commit'],'Frozen control source differs')
    dataset=Path(receipt['dataset']);assets=Path(receipt['assets']);phase2=Path(receipt['phase2_root']);step1=Path(receipt['step1_root'])
    for base,files,key in ((phase2,PHASE2_FILES,'phase2_files'),(step1,STEP1_FILES,'step1_files')):
        require(set(receipt[key])==set(files),'Evidence list differs: '+key)
        for name,digest in receipt[key].items():require(sha(base/name)==digest,'Evidence changed: '+key+' '+name)
    require(read(phase2/'config.json')==config['phase2_configuration'],'Phase 2 training configuration differs')
    require(read(step1/'COMPLETE.json')['passed'] is True,'Step 1 did not complete')
    require(sha(dataset/'DEV_ACCEPTANCE.json')==receipt['dataset_acceptance_sha256'],'Dataset acceptance changed')
    rows,acceptance=validate(dataset,mode='development',include_test=False)
    train_rows=[r for r in rows if r['split']=='train'];require(len(train_rows)==512,'Expected the 512 V4 training rows')
    require(sha(assets/'STAGED.json')==receipt['assets_stage_sha256'],'Model staging manifest changed')
    staged=read(assets/'STAGED.json');require(staged['model_id']==MODEL_ID,'Unexpected base model')
    for name,digest in staged['files'].items():require(sha(assets/'model'/name)==digest,'Staged base model changed')
    hashes={r['sample_id']:r['input_sha256'] for r in read(phase2/'preflight/ENCODINGS.json') if r['split']=='train'}
    return receipt,config,dataset,assets,step1,train_rows,staged,hashes


def control_items(processor,config,dataset,small,hashes):
    """Scored rows and supervised items per arm; inputs must equal the Phase 2 preflight hashes."""
    base=config['phase2_configuration'];data={}
    for arm in ARMS:
        spec=config['controls'][arm];scored=[relabel(r,arm) for r in small];items={}
        for original,row in zip(small,scored):
            text=answer(original,arm)
            if KIND[arm]=='direct':require(text==assistant_target(row,'direct'),'Control target and scored truth disagree')
            items[(row['sample_id'],KIND[arm])]=encode(processor,row,dataset,KIND[arm],base['max_length'],vision_inputs(processor,row,dataset),
                verify=True,answer=text,keys=spec.get('key_fields',[]))
        require(all(input_digest(v)==hashes[s] for (s,k),v in items.items()),'Control input differs from the Phase 2 preflight')
        data[arm]=(scored,items)
    return data


def report(root,config,runs,labels,step1_report):
    seeds=config['overfit']['seeds'];rules=config['rules']
    classes={arm:arm_class([runs[f'{arm}_seed{s}']['classification'] for s in seeds]) for arm in ARMS}
    pathway=pathway_reading(classes['t'],classes['s']);ap=ap_reading(classes['AP'])
    result=dict(classes=classes,pathway_reading=pathway,pathway_meaning=rules['pathway_readings'][pathway],
        ap_reading=ap,ap_meaning=rules['ap_readings'][ap],
        step1_reference=dict(decision=step1_report['decision'],overfit_16=step1_report['overfit_16']),
        runs={k:{m:v[m] for m in ('classification','final_accuracy','last_epoch_fit_loss','fit_loss_name')}|
            dict(accuracy_by_update={c['step']:c['verdict_accuracy'] for c in v['checks']}) for k,v in runs.items()},
        ap_protected_accuracy_by_update={k:{c['step']:c['scores']['facts']['protected_accuracy'] for c in v['checks']}
            for k,v in runs.items() if k.startswith('AP_')},
        labels=labels,rules=rules,thresholds=config['thresholds'],deferred=config['deferred'],
        scope='16-row single-pathway label controls on V4 development training data; validation and test rows are not scored')
    write(root/'REPORT.json',result)
    lines=['# Single-pathway label controls','',f'- t / s reading: **{pathway}**. {result["pathway_meaning"]}',
        f'- AP reading: **{ap}**. {result["ap_meaning"]}',
        f'- Arm classes: t {classes["t"]}, s {classes["s"]}, AP {classes["AP"]}; step 1 on the same rows: '
        f'A {step1_report["overfit_16"]["A"]}, B {step1_report["overfit_16"]["B"]}','',
        '| run | class | final accuracy | last-epoch fit loss | accuracy by update |','|---|---|---:|---:|---|']
    for k,v in result['runs'].items():
        lines.append(f'| {k} | {v["classification"]} | {v["final_accuracy"]:.3f} | {v["last_epoch_fit_loss"]:.4f} ({v["fit_loss_name"]}) | '
            +', '.join(f'{s}: {a:.3f}' for s,a in v['accuracy_by_update'].items())+' |')
    lines+=['','AP protected-colour accuracy by update: '+'; '.join(f'{k} '+', '.join(f'{s}: {a:.3f}' for s,a in v.items())
        for k,v in result['ap_protected_accuracy_by_update'].items()),'',
        f'Label checks: {labels["subset"]} (16 rows); {labels["training_rows"]} (512 rows).','',result['scope']]
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


def run(root):
    receipt,config,dataset,assets,step1,rows,staged,hashes=context(root)
    require(torch.cuda.is_available(),'Quest GPU required');started=time.monotonic()
    subprocess.run([sys.executable,'-m','unittest',*TESTS,'verifier_training.tests.test_fit_contract','verifier_training.tests.test_control_contract'],check=True)
    subset=overfit_units(rows);require(subset==read(step1/'SUBSET.json'),'The 16-row subset differs from step 1')
    plan=orders(subset['sample_ids'],config);previous=read(step1/'ORDERS.json')
    for seed,arms in plan.items():
        for arm,batches in arms.items():
            require([[s for s,k in b] for b in batches]==[[s for s,k in b] for b in previous[seed]['A']],'Sample order differs from step 1')
    write(root/'ORDERS.json',plan)
    chosen=set(subset['sample_ids']);small=[r for r in rows if r['sample_id'] in chosen]
    labels=dict(subset=check_labels(small),training_rows=check_labels(rows));write(root/'LABELS.json',labels)
    processor=processor_for(assets);data=control_items(processor,config,dataset,small,hashes);runs={}
    for arm in ARMS:
        spec=config['controls'][arm];scored,items=data[arm]
        for seed in config['overfit']['seeds']:
            runs[f'{arm}_seed{seed}']=overfit(root,arm,seed,config,assets,processor,scored,items,hashes,plan[str(seed)][arm],
                target=dict(reading=spec['reading'],kind=spec['kind'],prefix=None,loss=spec['loss']))
    result=report(root,config,runs,labels,read(step1/'REPORT.json'))
    write(root/'COMPLETE.json',dict(passed=True,pathway_reading=result['pathway_reading'],ap_reading=result['ap_reading'],classes=result['classes'],
        source_commit=receipt['source_commit'],upstream_commit=UPSTREAM,step1_root=str(step1),config_sha256=sha(root/'config.json'),
        orders_sha256=sha(root/'ORDERS.json'),labels_sha256=sha(root/'LABELS.json'),report_sha256=sha(root/'REPORT.json'),
        model_id=MODEL_ID,model_revision=staged['model_revision'],gpu=torch.cuda.get_device_name(),
        packages={k:metadata.version(k) for k in ('torch','transformers','peft','accelerate','numpy','pillow')},
        slurm={k:v for k,v in os.environ.items() if k in ('SLURM_JOB_ID','SLURM_JOB_NODELIST','SLURM_JOB_PARTITION','SLURM_JOB_ACCOUNT')},
        elapsed_seconds=time.monotonic()-started,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
        splits_loaded=sorted({r['split'] for r in rows}),validation_rows_scored=0,test_rows_loaded=0,api_calls=0))


if __name__=='__main__':
    root=Path(sys.argv[1])
    try:run(root)
    except BaseException as error:
        write(root/'STOP_controls.json',dict(error=type(error).__name__,message=str(error),slurm_job=os.environ.get('SLURM_JOB_ID')))
        raise
