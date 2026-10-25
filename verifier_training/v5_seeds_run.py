"""V5 seed campaign on Quest (docs/generative_verifier/V5_SEEDS_PROTOCOL.md): A and P with seeds 714 and 715.

Stages: preflight (CPU; A and P inputs equal to S3's, orders, S3 subsets and lookup, S4 points), train ARM SEED (GPU;
v5_run.train_arm for A, v5_pos_run.train_p for P, both unchanged), report (CPU; three seeds with 713 from the S3 and S4
records, and the preregistered reading). No stage loads test or test_natural.
"""
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from collections import Counter
from .common import read,write,sha,require,MODEL_ID
from .fd1000_run import vision_inputs,input_digest,processor_for
from .v5_contract import validate,schedule,selection_subset,fit_subset,lookup_predictions
from .v5_pos_contract import image_points,s4_reading
from .v5_seeds_contract import configuration,orders,check_orders,program_view,seed_average,seeds_reading,S3_FILES,S4_FILES,REFERENCE_SEED
from .v5_run import item as a_item,train_arm
from .v5_pos_run import p_item,train_p
from . import v5_metrics as M

TESTS=['verifier_training.tests.test_v5_seeds_contract','verifier_training.tests.test_v5_pos_contract',
    'verifier_training.tests.test_v5_contract','verifier_training.tests.test_v5_metrics']
SPLITS=('train','validation','validation_natural')
SCOPE='Validation-only V5 seed campaign (seeds 714 and 715 for A and P); the test split is not read'


def context(root):
    receipt=read(root/'RECEIPT.json');config=read(root/'config.json')
    require(config==configuration() and sha(root/'config.json')==receipt['configuration_sha256'],'Fixed seed-campaign configuration changed')
    require(os.environ['CB_CODE_COMMIT']==receipt['source_commit'],'Frozen campaign source differs')
    s3=Path(receipt['s3_root']);s4=Path(receipt['s4_root'])
    for base,key,names in ((s3,'s3_files',S3_FILES),(s4,'s4_files',S4_FILES)):
        require(set(receipt[key])==set(names),'Evidence list differs: '+key)
        for name,digest in receipt[key].items():require(sha(base/name)==digest,'Evidence changed: %s %s'%(key,name))
    require(read(s3/'config.json')==config['s3_configuration'] and read(s4/'config.json')==config['s4_configuration'],'S3 or S4 configuration differs')
    dataset=Path(receipt['dataset']);require(str(dataset)==config['dataset'],'Unexpected dataset root')
    rows=validate(dataset,include_test=False);require({r['split'] for r in rows}==set(SPLITS),'Unexpected splits loaded')
    assets=Path(receipt['assets']);require(sha(assets/'STAGED.json')==receipt['assets_stage_sha256'],'Model staging manifest changed')
    staged=read(assets/'STAGED.json');require(staged['model_id']==MODEL_ID,'Unexpected base model')
    for name,digest in staged['files'].items():require(sha(assets/'model'/name)==digest,'Staged base model changed')
    return receipt,config,dataset,assets,s3,s4,rows,staged


def split(rows,name):
    return [r for r in rows if r['split']==name]


def by_image(rows):
    groups={}
    for r in rows:groups.setdefault(r['metadata']['image_id'],[]).append(r)
    return groups


def preflight(root):
    receipt,config,dataset,assets,s3,s4,rows,staged=context(root);out=root/'preflight';out.mkdir()
    subprocess.run([sys.executable,'-m','unittest',*TESTS],check=True)
    base=config['s3_configuration'];pos=config['s4_configuration']
    s3_hashes={r['sample_id']:r['input_sha256'] for r in read(s3/'preflight/ENCODINGS.json')}
    s4_hashes={r['sample_id']:r['input_sha256'] for r in read(s4/'preflight/ENCODINGS.json')}
    require(s3_hashes==s4_hashes and set(s3_hashes)=={r['sample_id'] for r in rows},'S3 and S4 inputs cover other rows')
    processor=processor_for(assets);records=[];points={}
    for image,members in sorted(by_image(rows).items()):
        points[image]=image_points(dataset,image);vision=vision_inputs(processor,members[0],dataset)
        for row in members:
            a=a_item(processor,row,dataset,'direct',base,vision,verify=True)
            p=p_item(processor,row,dataset,pos,vision,points[image]['points'],verify=True)
            require(input_digest(a)==input_digest(p)==s3_hashes[row['sample_id']],'A or P input differs from the S3 inputs')
            for it in (a,p):require(it['labels'][:,:it['n']].eq(-100).all() and int(it['vm'].sum())==1,'Prompt supervised or decision mask invalid')
            records.append(dict(sample_id=row['sample_id'],split=row['split'],image_id=image,input_sha256=s3_hashes[row['sample_id']],
                prompt_tokens=a['n'],tokens_direct=a['fields']['input_ids'].shape[1],tokens_points=p['fields']['input_ids'].shape[1]))
    require(json.loads(json.dumps(points))==read(s4/'POINTS.json'),'Projected points differ from the S4 points')
    train=split(rows,'train');validation=split(rows,'validation');train_ids=[r['sample_id'] for r in train]
    reference=schedule(train_ids,base)
    require(reference==read(s3/'ORDERS.json'),'The schedule no longer reproduces the S3 orders')
    require(read(s4/'ORDERS.json')=={str(REFERENCE_SEED):{'P':reference[str(REFERENCE_SEED)]['B']}},'The S4 P order is not the S3 B order')
    plan=orders(train_ids,config);check_orders(plan,train_ids,config)
    old=[s for b in reference[str(REFERENCE_SEED)]['A'] for s,_ in b]
    require(all([s for b in arms['A'] for s,_ in b]!=old for arms in plan.values()),'A new seed repeats the 713 order')
    subsets=dict(selection=selection_subset(validation,base['selection']['images']),training_fit=fit_subset(train,base['training_fit']['rows']))
    require(subsets==read(s3/'SUBSETS.json'),'Subsets differ from S3')
    lookup=read(s3/'LOOKUP.json')
    require({s:lookup_predictions(train,split(rows,s)) for s in SPLITS}==lookup['predictions'],'Lookup differs from S3')
    write(root/'POINTS.json',points);write(root/'ORDERS.json',plan);write(root/'SUBSETS.json',subsets);write(root/'LOOKUP.json',lookup)
    write(out/'ENCODINGS.json',records)
    longest=max(max(r['tokens_direct'],r['tokens_points']) for r in records)
    write(out/'COMPLETE.json',dict(passed=True,rows=len(rows),by_split=dict(Counter(r['split'] for r in rows)),images=len(points),
        inputs_equal_s3=True,native_processor_equal=True,points_equal_s4=True,program_agrees_with_every_label=True,
        orders_reproduce_s3_and_s4=True,subsets_equal_s3=True,lookup_equal_s3=True,longest_sequence=longest,max_length=base['max_length'],
        no_truncation=longest<=base['max_length'],seeds=config['seeds'],updates_per_seed=base['optimizer_updates'],
        source_commit=receipt['source_commit'],encodings_sha256=sha(out/'ENCODINGS.json'),orders_sha256=sha(root/'ORDERS.json'),
        subsets_sha256=sha(root/'SUBSETS.json'),lookup_sha256=sha(root/'LOOKUP.json'),points_sha256=sha(root/'POINTS.json'),
        config_sha256=sha(root/'config.json'),s4_points_sha256=sha(s4/'POINTS.json'),test_rows_loaded=0,optimizer_steps=0,model_predictions=0,
        packages={k:metadata.version(k) for k in ('torch','transformers','peft','accelerate','numpy','pillow')},slurm_job=os.environ['SLURM_JOB_ID']))


def gate(root,receipt):
    done=read(root/'preflight/COMPLETE.json');require(done['passed'] and done['source_commit']==receipt['source_commit'],'Seed preflight gate failed')
    for name,key in (('preflight/ENCODINGS.json','encodings_sha256'),('ORDERS.json','orders_sha256'),('SUBSETS.json','subsets_sha256'),
                     ('LOOKUP.json','lookup_sha256'),('POINTS.json','points_sha256'),('config.json','config_sha256')):
        require(sha(root/name)==done[key],'Preflight evidence changed: '+name)
    hashes={r['sample_id']:r['input_sha256'] for r in read(root/'preflight/ENCODINGS.json')}
    return hashes,read(root/'POINTS.json'),read(root/'SUBSETS.json'),read(root/'LOOKUP.json')['predictions']


def train(root,arm,seed):
    receipt,config,dataset,assets,s3,s4,rows,staged=context(root);hashes,points,subsets,lookup=gate(root,receipt)
    require(arm in config['arms'] and seed in config['seeds'],'Unplanned run')
    provenance=dict(campaign=config['version'],campaign_config_sha256=sha(root/'config.json'),campaign_receipt_sha256=sha(root/'RECEIPT.json'),
        points_sha256=sha(root/'POINTS.json'),lookup_sha256=sha(root/'LOOKUP.json'))
    if arm=='A':train_arm(root,arm,seed,receipt,config['s3_configuration'],dataset,assets,rows,staged,hashes,subsets,lookup,provenance,SCOPE)
    else:train_p(root,arm,seed,receipt,config['s4_configuration'],dataset,assets,rows,staged,hashes,points,subsets,lookup,provenance,SCOPE)


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def run_dirs(root,s3,s4,config):
    dirs={REFERENCE_SEED:dict(A=s3/f'A_seed{REFERENCE_SEED}',P=s4/f'P_seed{REFERENCE_SEED}')}
    for seed in config['seeds']:dirs[seed]=dict(A=root/f'A_seed{seed}',P=root/f'P_seed{seed}')
    return dirs


def report(root):
    """CPU: per-seed A and P, the three-seed row means, the paired image-bootstrap interval and the reading."""
    receipt,config,dataset,assets,s3,s4,rows,staged=context(root);gate(root,receipt)
    validation={r['sample_id'] for r in split(rows,'validation')};runs={};a_by={};p_by={}
    for seed,dirs in sorted(run_dirs(root,s3,s4,config).items()):
        ra=read(dirs['A']/'RESULT.json');rp=read(dirs['P']/'RESULT.json')
        require(ra['passed'] and rp['passed'] and ra['seed']==rp['seed']==seed and ra['arm']=='A' and rp['arm']=='P','Run incomplete: seed %d'%seed)
        require(ra['test_rows_loaded']==rp['test_rows_loaded']==0 and ra['test_evaluations']==rp['test_evaluations']==0,'A run read test rows')
        a=lines(dirs['A']/'validation.jsonl');raw=lines(dirs['P']/'validation.jsonl');p=program_view(raw)
        require({r['sample_id'] for r in a}=={r['sample_id'] for r in p}==validation and len(a)==len(p)==len(validation),'Validation records incomplete')
        sa=M.score(a,'direct',bootstrap=False);sp=M.score(p,'direct',bootstrap=False)
        require(abs(sa['verdict_accuracy']-ra['R1_validation_accuracy'])<1e-12 and abs(sp['verdict_accuracy']-rp['R1_program_accuracy'])<1e-12,
                'Records differ from RESULT files: seed %d'%seed)
        paired=M.paired_difference(a,p)
        runs[str(seed)]=dict(source='S3 A and S4 P records' if seed==REFERENCE_SEED else 'this campaign',
            A=dict(R1=ra['R1_validation_accuracy'],R2=ra['R2_twin_image_all_correct'],R4=ra['R4_validation_natural_accuracy'],
                R5=ra['R5_training_fit_accuracy'],selected_step=ra['selected_step'],checkpoint_values=ra['checkpoint_values'],
                predicted_one_rate=sa['predicted_one_rate'],false_feasible_rate=sa['false_feasible_rate'],lookup_agreement=sa.get('lookup_agreement'),
                adapter_sha256=ra['adapter_sha256'],result_sha256=sha(dirs['A']/'RESULT.json')),
            P=dict(R1_program=rp['R1_program_accuracy'],R1_model=rp['R1_model_accuracy'],R2=rp['R2_twin_image_all_correct'],
                R4=rp['R4_natural_program_accuracy'],R5=rp['R5_training_fit_program_accuracy'],selected_step=rp['selected_step'],
                checkpoint_values=rp['checkpoint_values'],point_errors=rp['scores']['validation']['point_errors'],
                format_valid_rate=sum(r['format_valid'] for r in raw)/len(raw),
                written_separated_no_rate=sum(r['separated']=='no' for r in raw)/len(raw),written_verdict_one_rate=sum(r['model']==1 for r in raw)/len(raw),
                false_feasible_rate=sp['false_feasible_rate'],false_infeasible_rate=sp['false_infeasible_rate'],
                adapter_sha256=rp['adapter_sha256'],result_sha256=sha(dirs['P']/'RESULT.json')),
            paired_P_minus_A=paired,s4_reading_alone=s4_reading(rp['R1_program_accuracy'],ra['R1_validation_accuracy'],paired['image_bootstrap_95pct']))
        a_by[seed]=a;p_by[seed]=p
    s4_delta=read(s4/'S4_READING.json')['delta_P_program_minus_A']
    require(abs(runs[str(REFERENCE_SEED)]['paired_P_minus_A']['difference']-s4_delta['difference'])<1e-12,'Seed 713 differs from the S4 reading')
    mean=seed_average(a_by,p_by)
    reading=seeds_reading({k:v['P']['R1_program'] for k,v in runs.items()},{k:v['paired_P_minus_A']['difference'] for k,v in runs.items()},
        mean['image_bootstrap_95pct']['difference'])
    seeds=sorted(runs);spread=lambda arm,key:dict(min=min(runs[s][arm][key] for s in seeds),max=max(runs[s][arm][key] for s in seeds),
        mean=sum(runs[s][arm][key] for s in seeds)/len(seeds))
    write(root/'SEEDS_READING.json',dict(passed=True,reading=reading,seeds=[int(s) for s in seeds],per_seed=runs,seed_mean=mean,
        spread=dict(A_R1=spread('A','R1'),P_R1_program=spread('P','R1_program'),A_R4=spread('A','R4'),P_R4=spread('P','R4'),
            A_R5=spread('A','R5'),P_R5=spread('P','R5'),P_R2=spread('P','R2'),P_R1_model=spread('P','R1_model')),
        rules=config['rules'],thresholds=config['thresholds'],source_commit=receipt['source_commit'],test_rows_loaded=0,api_calls=0,
        slurm_job=os.environ.get('SLURM_JOB_ID')))
    print(reading,json.dumps(dict(P={s:runs[s]['P']['R1_program'] for s in seeds},A={s:runs[s]['A']['R1'] for s in seeds},
        delta=mean['difference'],interval=mean['image_bootstrap_95pct']['difference'])),flush=True)


if __name__=='__main__':
    stage=sys.argv[1];root=Path(sys.argv[2]);label=stage+('_'+'_'.join(sys.argv[3:]) if len(sys.argv)>3 else '')
    try:
        if stage=='preflight':preflight(root)
        elif stage=='train':train(root,sys.argv[3],int(sys.argv[4]))
        elif stage=='report':report(root)
        else:raise ValueError('Unknown stage')
    except BaseException as error:
        write(root/('STOP_'+label+'.json'),dict(error=type(error).__name__,message=str(error),slurm_job=os.environ.get('SLURM_JOB_ID')))
        raise
