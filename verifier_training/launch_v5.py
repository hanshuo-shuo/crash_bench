"""Launch the V5 S3 pilot from the clean, published Quest checkout (docs/generative_verifier/V5_S3_PROTOCOL.md).

python -m verifier_training.launch_v5 create              new campaign root and its CPU preflight
python -m verifier_training.launch_v5 pilot --root RUN    after preflight: A and B lanes, 3B zero-shot after A,
                                                          32B zero-shot after both lanes, report after all
python -m verifier_training.launch_v5 stage --root RUN --stage train --arm A --seed 713 [--after JOB[,JOB]]

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
from .v5_contract import configuration,DATASET,DATASET_HASHES

PROTOCOL='docs/generative_verifier/V5_S3_PROTOCOL.md'
SCRIPTS={'cpu':'v5_cpu.sbatch','gpu':'v5_gpu.sbatch','gpu2':'v5_zero_shot_32b.sbatch'}
STAGES={'preflight':('cpu',['verifier_training.v5_run','preflight']),'zero-shot-3b':('gpu',['verifier_training.v5_run','zero-shot-3b']),
    'zero-shot-32b':('gpu2',['verifier_training.v5_zero_shot_32b']),'report':('cpu',['verifier_training.v5_run','report'])}
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


def submit(root,label,args,kind,after=(),dependency='afterany'):
    require(not (root/(label+'_SUBMISSION.json')).exists(),'No silent resubmission or output overwrite')
    jobs=campaign_jobs(root);require(all(j in jobs for j in after),'Dependencies must be jobs of this campaign')
    env=dict(os.environ,CB_V5_ROOT=str(root),CB_V5_ARGS=' '.join(args),CB_VERIFIER_ASSETS=str(ASSETS))
    options=['--dependency=%s:%s'%(dependency,':'.join(after))] if after else []
    job=subprocess.check_output(['sbatch','--parsable','--job-name=cb_v5s3_'+label,'--output='+str(root/(label+'_%j.log')),*options,
        str(root/'source/verifier_training'/SCRIPTS[kind])],env=env,text=True).strip().split(';')[0]
    require(job.isdigit(),'Invalid Slurm receipt')
    result=dict(label=label,job=job,root=str(root),args=args,source_commit=read(root/'RECEIPT.json')['source_commit'],
        partition='short' if kind=='cpu' else 'gengpu',gpus={'cpu':0,'gpu':1,'gpu2':2}[kind],account='p33100',after=list(after),
        dependency=dependency if after else None,api_calls=0)
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


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('create','pilot','stage'))
    p.add_argument('--root',type=Path);p.add_argument('--stage');p.add_argument('--arm');p.add_argument('--seed',type=int)
    p.add_argument('--after',default='')
    a=p.parse_args()
    if a.action=='create':create()
    elif a.action=='pilot':pilot(a.root)
    else:stage(a.root,a.stage,a.arm,a.seed,tuple(x for x in a.after.split(',') if x))
