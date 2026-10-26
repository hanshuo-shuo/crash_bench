"""V5 test evaluation on Quest (docs/generative_verifier/V5_TEST_PROTOCOL.md): six frozen adapters, the test split once.

Stages: preflight (CPU; A and P test inputs equal, unit files and test rows against their freezes, the program on the projected
true points agrees with every label, lookup equal to the S2 readout), evaluate TARGET (GPU; the reload check on validation rows,
then each evaluated split once with the arm's own validation code), report (CPU; the preregistered reading). This is the only
V5 module that loads test rows. The receipt names the evaluated splits: test and test_natural, or validation_natural in a smoke.
"""
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import Counter
import torch
from peft import PeftModel
from .common import read,write,sha,require,MODEL_ID,UPSTREAM
from .fd1000_run import vision_inputs,input_digest,processor_for,base_model
from .v5_contract import validate,lookup_predictions,TEST_SPLITS
from .v5_pos_contract import image_points
from .v5_seeds_contract import program_view,seed_average,ARMS,ALL_SEEDS
from .v5_test_contract import (configuration,check_evidence,reload_subset,reload_agreement,test_reading,split_target,target_root,
    TARGETS,SMOKE_SPLITS)
from .v5_run import item as a_item,evaluate as a_evaluate
from .v5_pos_run import p_item,evaluate as p_evaluate,point_errors
from . import v5_metrics as M

TESTS=['verifier_training.tests.test_v5_test_contract','verifier_training.tests.test_v5_seeds_contract',
    'verifier_training.tests.test_v5_pos_contract','verifier_training.tests.test_v5_contract','verifier_training.tests.test_v5_metrics']


def context(root):
    receipt=read(root/'RECEIPT.json');config=read(root/'config.json')
    require(config==configuration() and sha(root/'config.json')==receipt['configuration_sha256'],'Fixed test configuration changed')
    require(os.environ['CB_CODE_COMMIT']==receipt['source_commit'],'Frozen test source differs')
    splits=tuple(receipt['splits'])
    require((receipt['stage'],splits) in (('v5-test',TEST_SPLITS),('v5-test-smoke',SMOKE_SPLITS)),'Unplanned splits')
    roots={k:Path(receipt[k+'_root']) for k in ('s3','s4','seeds')};check_evidence(receipt['evidence'],roots)
    require(read(roots['seeds']/'config.json')==config['seeds_configuration'],'Seed-campaign configuration differs')
    require(sha(root/'FREEZE.json')==receipt['freeze_sha256'] and set(read(root/'FREEZE.json'))==set(TARGETS),'Adapter freeze changed')
    dataset=Path(receipt['dataset']);require(str(dataset)==config['dataset'],'Unexpected dataset root')
    require(sha(dataset/'TEST_FREEZE.json')==receipt['test_freeze_sha256'],'Test freeze changed')
    rows=validate(dataset,include_test=splits==TEST_SPLITS)
    assets=Path(receipt['assets']);require(sha(assets/'STAGED.json')==receipt['assets_stage_sha256'],'Model staging manifest changed')
    staged=read(assets/'STAGED.json');require(staged['model_id']==MODEL_ID,'Unexpected base model')
    for name,digest in staged['files'].items():require(sha(assets/'model'/name)==digest,'Staged base model changed')
    return receipt,config,dataset,assets,roots,splits,rows,staged


def by_image(rows):
    groups={}
    for r in rows:groups.setdefault(r['metadata']['image_id'],[]).append(r)
    return groups


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def preflight(root):
    receipt,config,dataset,assets,roots,splits,rows,staged=context(root);out=root/'preflight';out.mkdir()
    subprocess.run([sys.executable,'-m','unittest',*TESTS],check=True)
    s3=config['seeds_configuration']['s3_configuration'];s4=config['seeds_configuration']['s4_configuration']
    evaluated=[r for r in rows if r['split'] in splits];require({r['split'] for r in evaluated}==set(splits),'Evaluated splits missing')
    if splits==TEST_SPLITS:
        frozen=read(dataset/'TEST_FREEZE.json');require(sorted(r['sample_id'] for r in evaluated)==sorted(frozen['sample_ids']),'Test rows differ from the freeze')
        require(all(frozen['images'][im['path']]==im['sha256'] for r in evaluated for im in r['student']['images']),'Test image differs from the freeze')
    s4_points=read(roots['s4']/'POINTS.json');processor=processor_for(assets);records=[];points={}
    for image,members in sorted(by_image(evaluated).items()):
        unit=dataset/'units'/image;hashes=read(unit/'HASHES.json')
        for name in ('CAMERA.json','RENDER_GEOMETRY.json'):require(sha(unit/name)==hashes[name],'Unit file changed: %s/%s'%(image,name))
        points[image]=image_points(dataset,image);vision=vision_inputs(processor,members[0],dataset)
        require(image not in s4_points or json.loads(json.dumps(points[image]))==s4_points[image],'Projected points differ from S4: '+image)
        for row in members:
            a=a_item(processor,row,dataset,'direct',s3,vision,verify=True)
            p=p_item(processor,row,dataset,s4,vision,points[image]['points'],verify=True)
            require(input_digest(a)==input_digest(p),'A and P inputs differ')
            for it in (a,p):require(it['labels'][:,:it['n']].eq(-100).all() and int(it['vm'].sum())==1,'Prompt supervised or decision mask invalid')
            records.append(dict(sample_id=row['sample_id'],split=row['split'],image_id=image,input_sha256=input_digest(a),prompt_tokens=a['n'],
                tokens_direct=a['fields']['input_ids'].shape[1],tokens_points=p['fields']['input_ids'].shape[1]))
    train=[r for r in rows if r['split']=='train'];part=lambda s:[r for r in evaluated if r['split']==s]
    lookup={s:lookup_predictions(train,part(s)) for s in splits};summary=read(dataset/'S2_SUMMARY.json')
    measured={s:sum(lookup[s][r['sample_id']]==int(r['target']['digit']) for r in part(s))/len(part(s)) for s in splits}
    require(all(abs(measured[s]-summary[s]['lookup_from_train']['label'])<1e-9 for s in splits),'Lookup differs from the S2 readout')
    write(root/'POINTS.json',points);write(root/'LOOKUP.json',dict(predictions=lookup,accuracy=measured));write(out/'ENCODINGS.json',records)
    longest=max(max(r['tokens_direct'],r['tokens_points']) for r in records)
    write(out/'COMPLETE.json',dict(passed=True,splits=list(splits),rows=len(evaluated),by_split=dict(Counter(r['split'] for r in evaluated)),
        images=len(points),inputs_equal_across_arms=True,native_processor_equal=True,program_agrees_with_every_label=True,
        unit_files_checked=True,test_freeze_checked=splits==TEST_SPLITS,lookup_accuracy=measured,longest_sequence=longest,max_length=s3['max_length'],
        no_truncation=longest<=s3['max_length'],source_commit=receipt['source_commit'],encodings_sha256=sha(out/'ENCODINGS.json'),
        points_sha256=sha(root/'POINTS.json'),lookup_sha256=sha(root/'LOOKUP.json'),config_sha256=sha(root/'config.json'),
        test_rows_loaded=sum(r['split'] in TEST_SPLITS for r in evaluated),model_predictions=0,
        packages={k:metadata.version(k) for k in ('torch','transformers','peft','accelerate','numpy','pillow')},slurm_job=os.environ['SLURM_JOB_ID']))


def gate(root,receipt,splits):
    done=read(root/'preflight/COMPLETE.json')
    require(done['passed'] and done['source_commit']==receipt['source_commit'] and done['splits']==list(splits),'Test preflight gate failed')
    for name,key in (('preflight/ENCODINGS.json','encodings_sha256'),('POINTS.json','points_sha256'),('LOOKUP.json','lookup_sha256'),
                     ('config.json','config_sha256')):
        require(sha(root/name)==done[key],'Preflight evidence changed: '+name)
    return {r['sample_id']:r['input_sha256'] for r in read(root/'preflight/ENCODINGS.json')},read(root/'POINTS.json'),read(root/'LOOKUP.json')['predictions']


def run(arm,model,processor,rows,dataset,config,hashes,lookup,points,stream):
    """Generate with the arm's own validation code: v5_run.evaluate for A, v5_pos_run.evaluate for P."""
    s3=config['seeds_configuration']['s3_configuration'];s4=config['seeds_configuration']['s4_configuration'];items={}
    for image,members in by_image(rows).items():
        vision=vision_inputs(processor,members[0],dataset)
        for row in members:
            if arm=='A':items[(row['sample_id'],'direct')]=a_item(processor,row,dataset,'direct',s3,vision)
            else:items[row['sample_id']]=p_item(processor,row,dataset,s4,vision,points[image]['points'])
    require(all(input_digest(v)==hashes[k[0] if arm=='A' else k] for k,v in items.items()),'Inference input differs from the preflight')
    if arm=='A':return a_evaluate(model,processor,rows,items,'direct_free','direct',s3,hashes,lookup,stream)
    return p_evaluate(model,processor,rows,items,s4,hashes,lookup,points,stream)


def evaluate(root,target):
    """One frozen adapter: reload check on validation rows, then every evaluated split exactly once."""
    receipt,config,dataset,assets,roots,splits,rows,staged=context(root);hashes,points,lookup=gate(root,receipt,splits)
    arm,seed=split_target(target);frozen=read(root/'FREEZE.json')[target];folder=Path(frozen['run_dir'])
    require(folder==target_root(target,roots)/target and (frozen['arm'],frozen['seed'])==(arm,seed),'Freeze points elsewhere')
    require(sha(folder/'selected_adapter/adapter_model.safetensors')==frozen['adapter_sha256'] and sha(folder/'RESULT.json')==frozen['result_sha256'],
            'Selected adapter changed after the freeze')
    (root/'evaluations').mkdir(exist_ok=True);out=root/'evaluations'/target;out.mkdir();require(torch.cuda.is_available(),'Quest GPU required')
    # TF32 matmuls as in the training-time evaluation; the fp32 LoRA path otherwise shifts P's generated coordinates.
    torch.manual_seed(seed);torch.backends.cuda.matmul.allow_tf32=True;started=time.monotonic();processor=processor_for(assets)
    model=PeftModel.from_pretrained(base_model(assets),folder/'selected_adapter',is_trainable=False);model.eval()
    require(not any(v.requires_grad for v in model.parameters()),'Adapter loaded as trainable')
    validation=[r for r in rows if r['split']=='validation'];chosen=set(reload_subset([r['sample_id'] for r in validation]))
    s3_hashes={r['sample_id']:r['input_sha256'] for r in read(roots['s3']/'preflight/ENCODINGS.json')}
    new=run(arm,model,processor,[r for r in validation if r['sample_id'] in chosen],dataset,config,s3_hashes,
        read(roots['s3']/'LOOKUP.json')['predictions']['validation'],read(roots['s4']/'POINTS.json'),out/'reload.jsonl')
    check=reload_agreement(new,[r for r in lines(folder/'validation.jsonl') if r['sample_id'] in chosen],arm)
    write(out/'RELOAD.json',dict(check,target=target,adapter_sha256=frozen['adapter_sha256']))
    require(check['passed'],'Reloaded adapter does not reproduce its validation outputs')
    counts={}
    for split in splits:counts[split]=len(run(arm,model,processor,[r for r in rows if r['split']==split],dataset,config,hashes,lookup[split],points,
        out/f'{split}.jsonl'))
    write(out/'COMPLETE.json',dict(passed=True,target=target,arm=arm,seed=seed,splits=list(splits),rows=counts,
        records_sha256={s:sha(out/f'{s}.jsonl') for s in splits},adapter_sha256=frozen['adapter_sha256'],freeze_sha256=sha(root/'FREEZE.json'),
        reload=dict(rows=check['rows'],decisions_equal=check['decisions_equal'],texts_equal=check['texts_equal']),evaluations=1,
        test_rows_loaded=sum(counts.get(s,0) for s in TEST_SPLITS),elapsed_seconds=time.monotonic()-started,
        peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,gpu=torch.cuda.get_device_name(),model_id=MODEL_ID,model_revision=staged['model_revision'],
        upstream_commit=UPSTREAM,source_commit=receipt['source_commit'],api_calls=0,
        slurm={k:v for k,v in os.environ.items() if k in ('SLURM_JOB_ID','SLURM_JOB_NODELIST','SLURM_JOB_PARTITION','SLURM_JOB_ACCOUNT')}))


def arm_scores(a,raw,bootstrap):
    p=program_view(raw);model=[dict(r,predicted=r['model'],reading='model') for r in raw]
    sa=M.score(a,'direct',bootstrap=bootstrap);sp=M.score(p,'direct',bootstrap=bootstrap);sm=M.score(model,'direct',bootstrap=False)
    return p,dict(A=dict(accuracy=sa['verdict_accuracy'],scores=sa),
        P=dict(program_accuracy=sp['verdict_accuracy'],model_accuracy=sm['verdict_accuracy'],scores=sp,point_errors=point_errors(raw),
            format_valid_rate=sum(r['format_valid'] for r in raw)/len(raw),written_separated_no_rate=sum(r['separated']=='no' for r in raw)/len(raw),
            written_verdict_one_rate=sum(r['model']==1 for r in raw)/len(raw)),
        paired_P_minus_A=M.paired_difference(a,p))


def report(root):
    """CPU: per seed A and P on every evaluated split, the three-seed row means on the primary split and the reading."""
    receipt,config,dataset,assets,roots,splits,rows,staged=context(root);gate(root,receipt,splits)
    primary=splits[0];validation=read(roots['seeds']/'SEEDS_READING.json')['per_seed'];runs={};a_by={};p_by={}
    for seed in ALL_SEEDS:
        records={}
        for arm in ARMS:
            target=f'{arm}_seed{seed}';out=root/'evaluations'/target;done=read(out/'COMPLETE.json')
            require(done['passed'] and done['splits']==list(splits) and done['evaluations']==1 and read(out/'RELOAD.json')['passed'],'Evaluation incomplete: '+target)
            for s in splits:require(sha(out/f'{s}.jsonl')==done['records_sha256'][s],'Records changed: %s %s'%(target,s))
            records[arm]={s:lines(out/f'{s}.jsonl') for s in splits}
        runs[str(seed)]={}
        for s in splits:
            ids={r['sample_id'] for r in rows if r['split']==s}
            require(all({r['sample_id'] for r in records[arm][s]}==ids and len(records[arm][s])==len(ids) for arm in ARMS),'Records incomplete: %d %s'%(seed,s))
            p,scores=arm_scores(records['A'][s],records['P'][s],bootstrap=s==primary);runs[str(seed)][s]=scores
            if s==primary:a_by[seed]=records['A'][s];p_by[seed]=p
        v=validation[str(seed)];t=runs[str(seed)][primary]
        runs[str(seed)]['validation_reference']=dict(A=v['A']['R1'],P=v['P']['R1_program'],
            change_A=t['A']['accuracy']-v['A']['R1'],change_P=t['P']['program_accuracy']-v['P']['R1_program'])
    mean=seed_average(a_by,p_by);seeds=[str(s) for s in ALL_SEEDS]
    reading=test_reading({s:runs[s][primary]['P']['program_accuracy'] for s in seeds},
        {s:runs[s][primary]['paired_P_minus_A']['difference'] for s in seeds},mean['image_bootstrap_95pct']['difference'])
    name='TEST_READING.json' if splits==TEST_SPLITS else 'SMOKE_READING.json'
    write(root/name,dict(passed=True,reading=reading,primary_split=primary,splits=list(splits),seeds=list(ALL_SEEDS),per_seed=runs,seed_mean=mean,
        lookup_accuracy=read(root/'LOOKUP.json')['accuracy'],rules=config['rules'],thresholds=config['thresholds'],
        freeze_sha256=sha(root/'FREEZE.json'),source_commit=receipt['source_commit'],
        rows=mean['rows'],test_rows_loaded=sum(r['split'] in TEST_SPLITS for r in rows),api_calls=0,slurm_job=os.environ.get('SLURM_JOB_ID')))
    print(reading,json.dumps(dict(P={s:runs[s][primary]['P']['program_accuracy'] for s in seeds},A={s:runs[s][primary]['A']['accuracy'] for s in seeds},
        delta=mean['difference'],interval=mean['image_bootstrap_95pct']['difference'])),flush=True)


if __name__=='__main__':
    stage=sys.argv[1];root=Path(sys.argv[2]);label=stage+('_'+'_'.join(sys.argv[3:]) if len(sys.argv)>3 else '')
    try:
        if stage=='preflight':preflight(root)
        elif stage=='evaluate':evaluate(root,sys.argv[3])
        elif stage=='report':report(root)
        else:raise ValueError('Unknown stage')
    except BaseException as error:
        write(root/('STOP_'+label+'.json'),dict(error=type(error).__name__,message=str(error),slurm_job=os.environ.get('SLURM_JOB_ID')))
        raise
