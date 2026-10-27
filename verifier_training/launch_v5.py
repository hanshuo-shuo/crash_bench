"""Launch the V5 S3 pilot from the clean, published Quest checkout (docs/generative_verifier/V5_S3_PROTOCOL.md).

python -m verifier_training.launch_v5 create              new campaign root and its CPU preflight
python -m verifier_training.launch_v5 pilot --root RUN    after preflight: A and B lanes, 3B zero-shot after A,
                                                          32B zero-shot after both lanes, report after all
python -m verifier_training.launch_v5 stage --root RUN --stage train --arm A --seed 713 [--after JOB[,JOB]]
python -m verifier_training.launch_v5 fit --root S3_RUN   fit diagnostic (docs/generative_verifier/V5_FIT_PROTOCOL.md): a new
                                                          campaign with the 16- and 64-row GPU jobs side by side, report after both
python -m verifier_training.launch_v5 s4 --root S3_RUN    S4 (docs/generative_verifier/V5_S4_PROTOCOL.md): CPU preflight and the
                                                          two-GPU zero-shot pointing together, then P training, then the report
python -m verifier_training.launch_v5 seeds --s3 S3_RUN --s4 S4_RUN
                                                          seed campaign (docs/generative_verifier/V5_SEEDS_PROTOCOL.md): CPU
                                                          preflight, then two GPU lanes (A then P for seeds 714 and 715), report last
python -m verifier_training.launch_v5 test --s3 S3_RUN --s4 S4_RUN --seeds SEEDS_RUN
                                                          the one test evaluation (docs/generative_verifier/V5_TEST_PROTOCOL.md):
                                                          CPU preflight, two GPU lanes over the six frozen adapters, report last
python -m verifier_training.launch_v5 det --s3 S3_RUN --s4 S4_RUN --seeds SEEDS_RUN
                                                          detector baseline (docs/generative_verifier/V5_DET_PROTOCOL.md): CPU
                                                          preflight, one GPU job in the AEGIS container, then the report

At most two GPUs at any time: the two training lanes, then the 32B stage (two GPUs) once both lanes end.
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
from .common import BASE,UPSTREAM,MODEL_ID,read,write,sha,require
from .launch import command
from .launch_verdict_diagnostic import ASSETS
from .launch_fd1000 import published_commit,archive,stamp
from .v5_contract import configuration,validate,DATASET,DATASET_HASHES
from . import v5_fit_contract,v5_pos_contract,v5_seeds_contract,v5_test_contract,v5_det_contract

PROTOCOL='docs/generative_verifier/V5_S3_PROTOCOL.md'
SCRIPTS={'cpu':'v5_cpu.sbatch','gpu':'v5_gpu.sbatch','gpu2':'v5_zero_shot_32b.sbatch','det':'v5_det_gpu.sbatch'}
STAGES={'preflight':('cpu',['verifier_training.v5_run','preflight']),'zero-shot-3b':('gpu',['verifier_training.v5_run','zero-shot-3b']),
    'zero-shot-32b':('gpu2',['verifier_training.v5_zero_shot_32b']),'report':('cpu',['verifier_training.v5_run','report'])}
FIT_PROTOCOL='docs/generative_verifier/V5_FIT_PROTOCOL.md'
S4_PROTOCOL='docs/generative_verifier/V5_S4_PROTOCOL.md'
SEEDS_PROTOCOL='docs/generative_verifier/V5_SEEDS_PROTOCOL.md'
TEST_PROTOCOL='docs/generative_verifier/V5_TEST_PROTOCOL.md'
DET_PROTOCOL='docs/generative_verifier/V5_DET_PROTOCOL.md'
DET_AUTHORIZATION=('2026-10-26 user ("先做1") chose the recommended detector baseline after the V5 test evaluation: GroundingDINO '
    '(the AEGIS weights) plus the S4 program on View 1, validation only, inference only, two fractions chosen on the training rows; '
    'no training, no API, no controller change, no test')
TEST_AUTHORIZATION=('2026-10-26 user ("都改一下 然后做测试级") approved the one test evaluation after the seed campaign read stable: '
    'the six selected adapters (A and P, seeds 713-715) scored once on the frozen test and test_natural splits; no training, '
    'no reselection, no API')
SEEDS_AUTHORIZATION=('2026-10-25 user ("收尾下 然后让下一个agent开始干下一步") handed the approved next step to the next agent: arms A '
    'and P with seeds 714 and 715 under the frozen S3 and S4 settings, validation only, read together with seed 713 from the S3 and S4 '
    'records; no test, no API')
S4_AUTHORIZATION=('2026-10-25 user ("换吧 继续 然后这一步干完先收尾 反正要尽力跑通故事线") approved switching B to positions plus a program: '
    'arm P with the S3 settings and seed 713, validation only, against the S3 A records; zero-shot pointing with 3B and local 32B; '
    'targets from stored geometry and cameras, no new rendering, no test, no API')
FIT_AUTHORIZATION=('2026-10-25 user ("做一下吧") approved the recommended fit diagnostic after S3: B from scratch on 16 and 64 '
    'training rows, read on held-out training images; training split only, no validation or test scoring, no API')
AUTHORIZATION=('2026-10-25 user approved S3 ("可以 开始写", handoff in V5_HANDOFF.md) and asked to continue it ("接着做 V5 的 S3"): '
    'A and B with seed 713, validation only, plus zero-shot 3B and local 32B; no test, no API')


def completed(job):
    # Slurm job numbers recur on Quest; keep only this project's records.
    lines=[x for x in command('sacct','-X','-n','-P','-j',job,'-u',os.environ['USER'],'--format=JobIDRaw,State,ExitCode,Account').splitlines()
           if x.endswith('|p33100')]
    return bool(lines) and all(x.split('|')[1:3]==['COMPLETED','0:0'] for x in lines)


def active_gpus():
    lines=command('squeue','-h','-u',os.environ['USER'],'-A','p33100','-p','gengpu','-o','%j|%b').splitlines()
    return sum(int(m.group(1)) if (m:=re.search(r'gpu(?::[a-z0-9]+)?:(\d+)',b)) else 1 for j,b in (x.split('|',1) for x in lines) if j.startswith('cb_'))


def campaign_jobs(root):
    return {r['job']:r for r in map(read,root.glob('*_SUBMISSION.json'))}


def submit(root,label,args,kind,after=(),dependency='afterany',spec=None):
    require(not (root/(label+'_SUBMISSION.json')).exists(),'No silent resubmission or output overwrite')
    jobs=campaign_jobs(root);require(all(j in jobs for j in after),'Dependencies must be jobs of this campaign')
    env=dict(os.environ,CB_V5_ROOT=str(root),CB_V5_ARGS=' '.join(args),CB_VERIFIER_ASSETS=str(ASSETS))
    options=['--dependency='+spec] if spec else ['--dependency=%s:%s'%(dependency,':'.join(after))] if after else []
    job=subprocess.check_output(['sbatch','--parsable','--job-name=cb_v5s3_'+label,'--output='+str(root/(label+'_%j.log')),*options,
        str(root/'source/verifier_training'/SCRIPTS[kind])],env=env,text=True).strip().split(';')[0]
    require(job.isdigit(),'Invalid Slurm receipt')
    result=dict(label=label,job=job,root=str(root),args=args,source_commit=read(root/'RECEIPT.json')['source_commit'],
        partition='short' if kind=='cpu' else 'gengpu',gpus={'cpu':0,'gpu':1,'gpu2':2,'det':1}[kind],account='p33100',after=list(after),
        dependency=spec or (dependency if after else None),api_calls=0)
    write(root/(label+'_SUBMISSION.json'),result);print(result,flush=True);return job


def create():
    commit=published_commit();repo=Path.cwd()
    require('状态：已冻结' in (repo/PROTOCOL).read_text(),'Freeze the S3 protocol before launching')
    for name,digest in DATASET_HASHES.items():require(sha(DATASET/name)==digest,'Frozen S2 file changed: '+name)
    require(read(ASSETS/'STAGED.json')['model_id']==MODEL_ID,'Wrong staged VLM; no substitution permitted')
    root=BASE/'verifier_training/runs'/(stamp()+'_v5-s3_'+commit[:12]);archive(root,commit);write(root/'config.json',configuration())
    receipt=dict(stage='v5-s3-pilot',source_commit=commit,upstream_commit=UPSTREAM,run_root=str(root),dataset=str(DATASET),
        dataset_hashes=dict(DATASET_HASHES),assets=str(ASSETS),assets_stage_sha256=sha(ASSETS/'STAGED.json'),
        configuration_sha256=sha(root/'config.json'),protocol=PROTOCOL,protocol_sha256=sha(root/'source'/PROTOCOL),
        seeds=configuration()['seeds'],arms=configuration()['arms'],api_calls=0,authorization=AUTHORIZATION,
        scope='V5 object/II/task1 A-versus-B pilot; validation splits only; the test split is not read')
    write(root/'RECEIPT.json',receipt);submit(root,'preflight',STAGES['preflight'][1]+[str(root)],'cpu')


def check_stage(root):
    receipt=read(root/'RECEIPT.json');require(command('git','rev-parse','HEAD')==receipt['source_commit'],'Use the campaign source commit for every stage')
    published_commit();require(completed(read(root/'preflight_SUBMISSION.json')['job']),'Preflight job did not finish normally')
    require(read(root/'preflight/COMPLETE.json')['passed'],'Preflight did not pass')
    return receipt


def stage(root,name,arm=None,seed=None,after=()):
    receipt=check_stage(root)
    if name=='train':
        require(arm in receipt['arms'] and seed in receipt['seeds'],'Unplanned arm or seed')
        label,kind,args=f'{arm}_seed{seed}','gpu',['verifier_training.v5_run','train',str(root),arm,str(seed)]
    else:
        require(name in STAGES and name!='preflight','Unknown stage');kind,args=STAGES[name];label=name.replace('-','_');args=args+[str(root)]
    if kind!='cpu':require(active_gpus()+{'gpu':1,'gpu2':2}[kind]<=2 or after,'More than two CrashBench GPUs would run')
    return submit(root,label,args,kind,after)


def pilot(root):
    """Every pilot stage with dependencies that keep at most two GPUs busy."""
    receipt=check_stage(root);require(active_gpus()==0,'Another CrashBench GPU job is active');seed=receipt['seeds'][0]
    a=stage(root,'train','A',seed);b=stage(root,'train','B',seed)
    z3=stage(root,'zero-shot-3b',after=(a,));z32=stage(root,'zero-shot-32b',after=(b,z3))
    stage(root,'report',after=(a,b,z3,z32))


def fit(s3):
    """Fit diagnostic: a new campaign that reads the finished S3 run and only the training split."""
    commit=published_commit();repo=Path.cwd();s3=Path(os.path.abspath(s3));s3.resolve().relative_to((BASE/'verifier_training/runs').resolve())
    require('状态：已冻结' in (repo/FIT_PROTOCOL).read_text(),'Freeze the fit protocol before launching')
    r3=read(s3/'RECEIPT.json');require(r3['stage']=='v5-s3-pilot' and read(s3/'S3_READING.json')['passed'] is True,'Needs the finished S3 pilot')
    for label in ('preflight','B_seed713','report'):require(completed(read(s3/(label+'_SUBMISSION.json'))['job']),'S3 job did not finish normally: '+label)
    require(read(ASSETS/'STAGED.json')['model_id']==MODEL_ID and r3['assets']==str(ASSETS),'Wrong staged VLM; no substitution permitted')
    require(active_gpus()==0,'Another CrashBench GPU job is active')
    config=v5_fit_contract.configuration();require(read(s3/'config.json')==config['s3_configuration'],'S3 configuration differs')
    train=[r for r in validate(DATASET,include_test=False) if r['split']=='train'];parts=v5_fit_contract.subsets(train,config)
    plan={size:v5_fit_contract.orders(parts[size]['sample_ids'],size,config) for size in config['sizes']}
    root=BASE/'verifier_training/runs'/(stamp()+'_v5-fit_'+commit[:12]);archive(root,commit)
    write(root/'config.json',config);write(root/'SUBSETS.json',parts);write(root/'ORDERS.json',plan)
    receipt=dict(stage='v5-fit-diagnostic',source_commit=commit,upstream_commit=UPSTREAM,run_root=str(root),s3_root=str(s3),
        s3_source_commit=r3['source_commit'],s3_files={f:sha(s3/f) for f in v5_fit_contract.S3_FILES},dataset=str(DATASET),
        dataset_hashes=dict(DATASET_HASHES),assets=str(ASSETS),assets_stage_sha256=sha(ASSETS/'STAGED.json'),
        configuration_sha256=sha(root/'config.json'),subsets_sha256=sha(root/'SUBSETS.json'),orders_sha256=sha(root/'ORDERS.json'),
        protocol=FIT_PROTOCOL,protocol_sha256=sha(root/'source'/FIT_PROTOCOL),seeds=config['seeds'],sizes=list(config['sizes']),api_calls=0,
        authorization=FIT_AUTHORIZATION,scope='Fit diagnostic on V5 training rows; validation and test are not scored')
    write(root/'RECEIPT.json',receipt)
    jobs=[submit(root,'fit'+size,['verifier_training.v5_fit_run','fit',str(root),size],'gpu') for size in config['sizes']]
    submit(root,'report',['verifier_training.v5_fit_run','report',str(root)],'cpu',after=tuple(jobs))


def s4(s3):
    """S4: a new campaign that reads the finished S3 run; P is compared with the S3 A records."""
    commit=published_commit();repo=Path.cwd();s3=Path(os.path.abspath(s3));s3.resolve().relative_to((BASE/'verifier_training/runs').resolve())
    require('状态：已冻结' in (repo/S4_PROTOCOL).read_text(),'Freeze the S4 protocol before launching')
    r3=read(s3/'RECEIPT.json');require(r3['stage']=='v5-s3-pilot' and read(s3/'S3_READING.json')['passed'] is True,'Needs the finished S3 pilot')
    for label in ('preflight','A_seed713','B_seed713','report'):require(completed(read(s3/(label+'_SUBMISSION.json'))['job']),'S3 job did not finish normally: '+label)
    require(read(ASSETS/'STAGED.json')['model_id']==MODEL_ID and r3['assets']==str(ASSETS),'Wrong staged VLM; no substitution permitted')
    require(active_gpus()==0,'Another CrashBench GPU job is active')
    config=v5_pos_contract.configuration();require(read(s3/'config.json')==config['s3_configuration'],'S3 configuration differs')
    root=BASE/'verifier_training/runs'/(stamp()+'_v5-s4_'+commit[:12]);archive(root,commit);write(root/'config.json',config)
    receipt=dict(stage='v5-s4',source_commit=commit,upstream_commit=UPSTREAM,run_root=str(root),s3_root=str(s3),s3_source_commit=r3['source_commit'],
        s3_files={f:sha(s3/f) for f in v5_pos_contract.S3_FILES},dataset=str(DATASET),dataset_hashes=dict(DATASET_HASHES),assets=str(ASSETS),
        assets_stage_sha256=sha(ASSETS/'STAGED.json'),configuration_sha256=sha(root/'config.json'),protocol=S4_PROTOCOL,
        protocol_sha256=sha(root/'source'/S4_PROTOCOL),seeds=config['seeds'],arms=['P'],api_calls=0,authorization=S4_AUTHORIZATION,
        scope='Arm P (positions plus a program) against the S3 A records; validation only; the test split is not read')
    write(root/'RECEIPT.json',receipt)
    pre=submit(root,'preflight',['verifier_training.v5_pos_run','preflight',str(root)],'cpu')
    zs=submit(root,'zero_shot',['verifier_training.v5_pos_zero_shot',str(root)],'gpu2')
    tr=submit(root,'P_seed713',['verifier_training.v5_pos_run','train',str(root),'P','713'],'gpu',after=(pre,zs),spec=f'afterok:{pre},afterany:{zs}')
    submit(root,'report',['verifier_training.v5_pos_run','report',str(root)],'cpu',after=(tr,zs))


def seeds(s3,s4):
    """Seed campaign: CPU preflight, then one GPU lane per new seed (A, then P), then the CPU report; at most two GPUs."""
    commit=published_commit();repo=Path.cwd();runs=(BASE/'verifier_training/runs').resolve()
    s3=Path(os.path.abspath(s3));s4=Path(os.path.abspath(s4));s3.resolve().relative_to(runs);s4.resolve().relative_to(runs)
    require('状态：已冻结' in (repo/SEEDS_PROTOCOL).read_text(),'Freeze the seed protocol before launching')
    r3=read(s3/'RECEIPT.json');r4=read(s4/'RECEIPT.json')
    require(r3['stage']=='v5-s3-pilot' and read(s3/'S3_READING.json')['passed'] is True,'Needs the finished S3 pilot')
    require(r4['stage']=='v5-s4' and read(s4/'S4_READING.json')['passed'] is True and Path(r4['s3_root']).resolve()==s3.resolve(),
            'Needs the finished S4 run on this S3 root')
    for prior,labels in ((s3,('preflight','A_seed713','B_seed713','report')),(s4,('preflight','P_seed713','report'))):
        for label in labels:require(completed(read(prior/(label+'_SUBMISSION.json'))['job']),'Prior job did not finish normally: %s %s'%(prior.name,label))
    require(read(ASSETS/'STAGED.json')['model_id']==MODEL_ID and r3['assets']==r4['assets']==str(ASSETS),'Wrong staged VLM; no substitution permitted')
    require(active_gpus()==0,'Another CrashBench GPU job is active')
    config=v5_seeds_contract.configuration()
    require(read(s3/'config.json')==config['s3_configuration'] and read(s4/'config.json')==config['s4_configuration'],'S3 or S4 configuration differs')
    root=BASE/'verifier_training/runs'/(stamp()+'_v5-seeds_'+commit[:12]);archive(root,commit);write(root/'config.json',config)
    receipt=dict(stage='v5-seeds',source_commit=commit,upstream_commit=UPSTREAM,run_root=str(root),s3_root=str(s3),s3_source_commit=r3['source_commit'],
        s4_root=str(s4),s4_source_commit=r4['source_commit'],s3_files={f:sha(s3/f) for f in v5_seeds_contract.S3_FILES},
        s4_files={f:sha(s4/f) for f in v5_seeds_contract.S4_FILES},dataset=str(DATASET),dataset_hashes=dict(DATASET_HASHES),assets=str(ASSETS),
        assets_stage_sha256=sha(ASSETS/'STAGED.json'),configuration_sha256=sha(root/'config.json'),protocol=SEEDS_PROTOCOL,
        protocol_sha256=sha(root/'source'/SEEDS_PROTOCOL),seeds=config['seeds'],arms=config['arms'],api_calls=0,authorization=SEEDS_AUTHORIZATION,
        scope='Arms A and P with seeds 714 and 715; validation only; the test split is not read')
    write(root/'RECEIPT.json',receipt)
    module='verifier_training.v5_seeds_run';pre=submit(root,'preflight',[module,'preflight',str(root)],'cpu');lanes=[]
    for seed in config['seeds']:
        a=submit(root,f'A_seed{seed}',[module,'train',str(root),'A',str(seed)],'gpu',after=(pre,),dependency='afterok')
        p=submit(root,f'P_seed{seed}',[module,'train',str(root),'P',str(seed)],'gpu',after=(pre,a),spec=f'afterok:{pre},afterany:{a}')
        lanes+=[a,p]
    submit(root,'report',[module,'report',str(root)],'cpu',after=tuple(lanes))


def test_root(root,commit,roots,stage,splits,authorization,scope):
    """A test campaign root: source archive, configuration, the adapter freeze and the receipt with every prior file's hash."""
    archive(root,commit);config=v5_test_contract.configuration();write(root/'config.json',config)
    write(root/'FREEZE.json',v5_test_contract.freeze(roots))
    receipt=dict(stage=stage,splits=list(splits),source_commit=commit,upstream_commit=UPSTREAM,run_root=str(root),
        **{k+'_root':str(v) for k,v in roots.items()},evidence=v5_test_contract.evidence(roots),dataset=str(DATASET),
        dataset_hashes=dict(DATASET_HASHES),test_freeze_sha256=sha(DATASET/'TEST_FREEZE.json'),assets=str(ASSETS),
        assets_stage_sha256=sha(ASSETS/'STAGED.json'),configuration_sha256=sha(root/'config.json'),freeze_sha256=sha(root/'FREEZE.json'),
        protocol=TEST_PROTOCOL,protocol_sha256=sha(root/'source'/TEST_PROTOCOL),targets=list(v5_test_contract.TARGETS),api_calls=0,
        authorization=authorization,scope=scope)
    write(root/'RECEIPT.json',receipt)


def test_jobs(root):
    """CPU preflight, two GPU lanes over the six adapters (each job after the preflight and its lane predecessor), report after all."""
    module='verifier_training.v5_test_run';pre=submit(root,'preflight',[module,'preflight',str(root)],'cpu');jobs=[]
    for lane in v5_test_contract.LANES:
        prev=None
        for target in lane:
            spec=f'afterok:{pre}'+(f',afterany:{prev}' if prev else '')
            prev=submit(root,target,[module,'evaluate',str(root),target],'gpu',after=(pre,)+((prev,) if prev else ()),spec=spec);jobs.append(prev)
    submit(root,'report',[module,'report',str(root)],'cpu',after=tuple(jobs))


def test(s3,s4,seeds_root):
    """The one test evaluation: a new campaign over the finished S3, S4 and seed runs; nothing is trained or reselected."""
    commit=published_commit();repo=Path.cwd();runs=(BASE/'verifier_training/runs').resolve()
    roots={k:Path(os.path.abspath(v)) for k,v in (('s3',s3),('s4',s4),('seeds',seeds_root))}
    for path in roots.values():path.resolve().relative_to(runs)
    require('状态：已冻结' in (repo/TEST_PROTOCOL).read_text(),'Freeze the test protocol before launching')
    r3,r4,rs=(read(roots[k]/'RECEIPT.json') for k in ('s3','s4','seeds'))
    require((r3['stage'],r4['stage'],rs['stage'])==('v5-s3-pilot','v5-s4','v5-seeds'),'Unexpected prior runs')
    same=lambda a,b:Path(a).resolve()==Path(b).resolve()
    require(same(r4['s3_root'],roots['s3']) and same(rs['s3_root'],roots['s3']) and same(rs['s4_root'],roots['s4']),'Prior runs do not chain')
    require(read(roots['seeds']/'SEEDS_READING.json')['reading']=='stable','The seed campaign did not read stable')
    for key,labels in (('s3',('preflight','A_seed713','report')),('s4',('preflight','P_seed713','report')),
                       ('seeds',('preflight','A_seed714','A_seed715','P_seed714','P_seed715','report'))):
        for label in labels:require(completed(read(roots[key]/(label+'_SUBMISSION.json'))['job']),'Prior job did not finish normally: %s %s'%(key,label))
    require(read(ASSETS/'STAGED.json')['model_id']==MODEL_ID and r3['assets']==r4['assets']==rs['assets']==str(ASSETS),'Wrong staged VLM; no substitution permitted')
    require(active_gpus()==0,'Another CrashBench GPU job is active')
    require(read(roots['seeds']/'config.json')==v5_test_contract.configuration()['seeds_configuration'],'Seed-campaign configuration differs')
    root=BASE/'verifier_training/runs'/(stamp()+'_v5-test_'+commit[:12])
    test_root(root,commit,roots,'v5-test',v5_test_contract.TEST_SPLITS,TEST_AUTHORIZATION,
        'The six selected adapters scored once on test and test_natural; no training, no reselection')
    test_jobs(root)


def det(s3,s4,seeds_root):
    """Detector baseline: a new campaign over the finished S3, S4 and seed runs; CPU preflight, one GPU detection job, CPU report."""
    commit=published_commit();repo=Path.cwd();runs=(BASE/'verifier_training/runs').resolve()
    roots={k:Path(os.path.abspath(v)) for k,v in (('s3',s3),('s4',s4),('seeds',seeds_root))}
    for path in roots.values():path.resolve().relative_to(runs)
    require('状态：已冻结' in (repo/DET_PROTOCOL).read_text(),'Freeze the detector protocol before launching')
    r3,r4,rs=(read(roots[k]/'RECEIPT.json') for k in ('s3','s4','seeds'))
    require((r3['stage'],r4['stage'],rs['stage'])==('v5-s3-pilot','v5-s4','v5-seeds'),'Unexpected prior runs')
    same=lambda a,b:Path(a).resolve()==Path(b).resolve()
    require(same(r4['s3_root'],roots['s3']) and same(rs['s3_root'],roots['s3']) and same(rs['s4_root'],roots['s4']),'Prior runs do not chain')
    require(read(roots['seeds']/'SEEDS_READING.json')['reading']=='stable','The seed campaign did not read stable')
    for key,labels in (('s3',('preflight','A_seed713','report')),('s4',('preflight','zero_shot','P_seed713','report')),
                       ('seeds',('preflight','A_seed714','A_seed715','P_seed714','P_seed715','report'))):
        for label in labels:require(completed(read(roots[key]/(label+'_SUBMISSION.json'))['job']),'Prior job did not finish normally: %s %s'%(key,label))
    require(active_gpus()==0,'Another CrashBench GPU job is active')
    config=v5_det_contract.configuration();require(read(roots['seeds']/'config.json')==config['seeds_configuration'],'Seed-campaign configuration differs')
    staged=read(BASE/'staging.json')['stages'];detector=config['detector'];container=BASE/'containers/aegis-py38.sif'
    require(staged['groundingdino']['sha256']==detector['checkpoint_sha256'] and staged['bert']['revision']==detector['bert_revision'],
            'Detector staging differs from the AEGIS reproduction')
    assets=dict(checkpoint=staged['groundingdino']['path'],checkpoint_sha256=detector['checkpoint_sha256'],
        config=str(BASE/'GroundingDINO'/detector['config']),config_sha256=sha(BASE/'GroundingDINO'/detector['config']),bert=staged['bert']['path'],
        bert_revision=detector['bert_revision'],container=str(container),
        container_sha256=(BASE/'containers/aegis-py38.sif.sha256').read_text().split()[0],environment_ready_sha256=sha(BASE/'envs/aegis_sim/READY.json'))
    root=BASE/'verifier_training/runs'/(stamp()+'_v5-det_'+commit[:12]);archive(root,commit);write(root/'config.json',config)
    files={'s3':v5_det_contract.S3_FILES,'s4':v5_det_contract.S4_FILES,'seeds':v5_det_contract.SEEDS_FILES}
    receipt=dict(stage='v5-det',source_commit=commit,upstream_commit=UPSTREAM,run_root=str(root),**{k+'_root':str(v) for k,v in roots.items()},
        **{k+'_files':{f:sha(roots[k]/f) for f in names} for k,names in files.items()},dataset=str(DATASET),dataset_hashes=dict(DATASET_HASHES),
        detector_assets=assets,configuration_sha256=sha(root/'config.json'),protocol=DET_PROTOCOL,protocol_sha256=sha(root/'source'/DET_PROTOCOL),
        api_calls=0,authorization=DET_AUTHORIZATION,scope='GroundingDINO plus the S4 program on View 1; validation only; nothing trained; the test split is not read')
    write(root/'RECEIPT.json',receipt)
    pre=submit(root,'preflight',['verifier_training.v5_det_run','preflight',str(root)],'cpu')
    found=submit(root,'detect',['verifier_training/v5_det_detect.py',str(root)],'det',after=(pre,),dependency='afterok')
    submit(root,'report',['verifier_training.v5_det_run','report',str(root)],'cpu',after=(found,))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('create','pilot','stage','fit','s4','seeds','test','det'))
    p.add_argument('--root',type=Path);p.add_argument('--s3',type=Path);p.add_argument('--s4',type=Path);p.add_argument('--seeds',type=Path)
    p.add_argument('--stage');p.add_argument('--arm');p.add_argument('--seed',type=int)
    p.add_argument('--after',default='')
    a=p.parse_args()
    if a.action=='create':create()
    elif a.action=='pilot':pilot(a.root)
    elif a.action=='fit':fit(a.root)
    elif a.action=='s4':s4(a.root)
    elif a.action=='seeds':seeds(a.s3,a.s4)
    elif a.action=='test':test(a.s3,a.s4,a.seeds)
    elif a.action=='det':det(a.s3,a.s4,a.seeds)
    else:stage(a.root,a.stage,a.arm,a.seed,tuple(x for x in a.after.split(',') if x))
