"""Launch the V5 S3 pilot from the clean, published Quest checkout (docs/generative_verifier/V5_S3_PROTOCOL.md).

python -m verifier_training.launch_v5 create              new campaign root and its CPU preflight
python -m verifier_training.launch_v5 pilot --root RUN    after preflight: A and B lanes, 3B zero-shot after A,
                                                          32B zero-shot after both lanes, report after all
python -m verifier_training.launch_v5 stage --root RUN --stage train --arm A --seed 713 [--after JOB[,JOB]]
python -m verifier_training.launch_v5 fit --root S3_RUN   fit diagnostic (docs/generative_verifier/V5_FIT_PROTOCOL.md): a new
                                                          campaign with the 16- and 64-row GPU jobs side by side, report after both
python -m verifier_training.launch_v5 s4 --root S3_RUN    S4 (docs/generative_verifier/V5_S4_PROTOCOL.md): CPU preflight and the
                                                          two-GPU zero-shot pointing together, then P training, then the report

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
from . import v5_fit_contract,v5_pos_contract

PROTOCOL='docs/generative_verifier/V5_S3_PROTOCOL.md'
SCRIPTS={'cpu':'v5_cpu.sbatch','gpu':'v5_gpu.sbatch','gpu2':'v5_zero_shot_32b.sbatch'}
STAGES={'preflight':('cpu',['verifier_training.v5_run','preflight']),'zero-shot-3b':('gpu',['verifier_training.v5_run','zero-shot-3b']),
    'zero-shot-32b':('gpu2',['verifier_training.v5_zero_shot_32b']),'report':('cpu',['verifier_training.v5_run','report'])}
FIT_PROTOCOL='docs/generative_verifier/V5_FIT_PROTOCOL.md'
S4_PROTOCOL='docs/generative_verifier/V5_S4_PROTOCOL.md'
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
        partition='short' if kind=='cpu' else 'gengpu',gpus={'cpu':0,'gpu':1,'gpu2':2}[kind],account='p33100',after=list(after),
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


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('create','pilot','stage','fit','s4'))
    p.add_argument('--root',type=Path);p.add_argument('--stage');p.add_argument('--arm');p.add_argument('--seed',type=int)
    p.add_argument('--after',default='')
    a=p.parse_args()
    if a.action=='create':create()
    elif a.action=='pilot':pilot(a.root)
    elif a.action=='fit':fit(a.root)
    elif a.action=='s4':s4(a.root)
    else:stage(a.root,a.stage,a.arm,a.seed,tuple(x for x in a.after.split(',') if x))
