"""Launch V4 development stages from the clean, published Quest checkout.

python -m verifier_training.launch_fd1000 export
python -m verifier_training.launch_fd1000 create --dataset EXPORT/dataset --schedule trial --name phase1
python -m verifier_training.launch_fd1000 stage --root RUN --stage zero-shot
python -m verifier_training.launch_fd1000 stage --root RUN --stage train --arm A --seed 713 [--after JOB]
python -m verifier_training.launch_fd1000 stage --root RUN --stage report --split validation
python -m verifier_training.launch_fd1000 freeze --root RUN      (Phase 2 only, after every run)
python -m verifier_training.launch_fd1000 stage --root RUN --stage evaluate-test --target A_seed713 [--after JOB]
python -m verifier_training.launch_fd1000 stage --root RUN --stage report --split test

--after chains GPU jobs into lanes (afterany), so two lanes never use more than two GPUs.
"""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
from .common import BASE,UPSTREAM,MODEL_ID,read,write,sha,require
from .launch import command
from .launch_verdict_diagnostic import ASSETS
from .fd1000_contract import configuration
from verifier_dataset.statistics_assets import ROOT as STATISTICS

PREREG='verifier_training/configs/v4_dev_prereg.json'
ORIGINS=('git@github.com:hanshuo-shuo/crash_bench.git','https://github.com/hanshuo-shuo/crash_bench.git')


def normal(job):
    require(command('sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode')==job+'|COMPLETED|0:0',
            'Prerequisite job has not finished normally')


def published_commit():
    repo=Path.cwd().resolve();require(repo==(Path.home()/'crash_bench').resolve(),'Wrong project checkout')
    require(command('git','remote','get-url','origin') in ORIGINS,'Wrong origin')
    require(not command('git','status','--porcelain'),'Dirty source')
    commit=command('git','rev-parse','HEAD');require('origin/' in command('git','branch','-r','--contains',commit),'Source must be published')
    return commit


def archive(root,commit):
    root.mkdir();(root/'source').mkdir()
    with (root/'source.tar').open('wb') as f:subprocess.run(['git','archive',commit],stdout=f,check=True)
    subprocess.run(['tar','-xf',str(root/'source.tar'),'-C',str(root/'source')],check=True)
    (root/'SOURCE_COMMIT').write_text(commit+'\n');(root/'source.sha256').write_text(sha(root/'source.tar')+'  '+str(root/'source.tar')+'\n')


def active_gpu_jobs():
    lines=command('squeue','-h','-u',os.environ['USER'],'-A','p33100','-p','gengpu','-o','%j').splitlines()
    return sum(line.startswith('cb_') for line in lines)


def lane_predecessor(root,job):
    """A lane may only wait on a GPU job of this same campaign."""
    receipts=[read(p) for p in root.glob('*_SUBMISSION.json')]
    require(any(r['job']==job and r['partition']=='gengpu' for r in receipts),'Lane predecessor is not a GPU job of this campaign')
    return job


def submit(root,label,args,gpu,after=None,statistics=False):
    require(not (root/(label+'_SUBMISSION.json')).exists(),'No silent resubmission or output overwrite')
    if gpu and after is None:require(active_gpu_jobs()<2,'Two CrashBench GPU jobs already active')
    env=dict(os.environ,CB_V4_ROOT=str(root),CB_V4_ARGS=' '.join(args),CB_VERIFIER_ASSETS=str(ASSETS),CB_V4_STATISTICS='1' if statistics else '0')
    script=root/'source/verifier_training'/('v4_gpu.sbatch' if gpu else 'v4_cpu.sbatch')
    options=['--dependency=afterany:'+lane_predecessor(root,after)] if after else []
    job=subprocess.check_output(['sbatch','--parsable','--job-name=cb_v4_'+label,'--output='+str(root/(label+'_%j.log')),*options,str(script)],
        env=env,text=True).strip().split(';')[0]
    require(job.isdigit(),'Invalid Slurm receipt')
    result=dict(label=label,job=job,root=str(root),args=args,source_commit=read(root/'RECEIPT.json')['source_commit'],
        partition='gengpu' if gpu else 'short',account='p33100',after=after,api_calls=0)
    write(root/(label+'_SUBMISSION.json'),result);print(result,flush=True)


def stamp():return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def export():
    commit=published_commit();root=BASE/'verifier_dataset'/(stamp()+'_v4-dev-export_'+commit[:12]);archive(root,commit)
    receipt=dict(stage='v4-dev-export',source_commit=commit,upstream_commit=UPSTREAM,run_root=str(root),seed=1014,api_calls=0,
        authorization='User decision 2026-10-16 (V4 as development data) and 2026-10-17 instruction to start Phase 0',
        scope='CPU-only packaging with checked structured facts; no model fit, no test prediction')
    write(root/'RECEIPT.json',receipt);submit(root,'export',['verifier_training.v4_export',str(root)],gpu=False)


def create(dataset,schedule,name):
    commit=published_commit();dataset=Path(os.path.abspath(dataset));dataset.resolve().relative_to((BASE/'verifier_dataset').resolve())
    parent=dataset.parent;normal(read(parent/'export_SUBMISSION.json')['job'])
    require(read(parent/'COMPLETE.json')['passed'] and read(parent/'COMPLETE.json')['dev_acceptance_sha256']==sha(dataset/'DEV_ACCEPTANCE.json'),
            'Development export incomplete or changed')
    require(read(ASSETS/'STAGED.json')['model_id']==MODEL_ID,'Wrong staged VLM; no substitution permitted')
    require(name.replace('-','').isalnum(),'Campaign name must be a short token')
    root=BASE/'verifier_training/runs'/(stamp()+'_'+name+'_'+commit[:12]);archive(root,commit)
    write(root/'config.json',configuration(schedule))
    receipt=dict(stage='campaign',name=name,schedule=schedule,source_commit=commit,upstream_commit=UPSTREAM,run_root=str(root),
        dataset=str(dataset),assets=str(ASSETS),dataset_acceptance_sha256=sha(dataset/'DEV_ACCEPTANCE.json'),
        assets_stage_sha256=sha(ASSETS/'STAGED.json'),statistics_assets_sha256=sha(STATISTICS/'STAGED.json'),
        configuration_sha256=sha(root/'config.json'),prereg_sha256=sha(root/'source'/PREREG),
        seeds=configuration(schedule)['seeds'],arms=configuration(schedule)['arms'],api_calls=0,
        authorization='2026-10-17: user asked to start Phase 0 and continue with Phase 1 if it goes well',
        scope='Development A/B/C verifier comparison; validation-only until a later explicit evaluate-test')
    write(root/'RECEIPT.json',receipt);submit(root,'preflight',['verifier_training.fd1000_run','preflight',str(root)],gpu=False)


def stage(root,name,arm=None,seed=None,target=None,split=None,after=None):
    receipt=read(root/'RECEIPT.json');require(command('git','rev-parse','HEAD')==receipt['source_commit'],'Use the campaign source commit for every stage')
    published_commit();normal(read(root/'preflight_SUBMISSION.json')['job']);require(read(root/'preflight/COMPLETE.json')['passed'],'Preflight did not pass')
    if name=='zero-shot':submit(root,'zero_shot',['verifier_training.fd1000_run','zero-shot',str(root)],gpu=True,after=after)
    elif name=='train':
        require(arm in receipt['arms'] and seed in receipt['seeds'],'Unplanned arm or seed')
        submit(root,f'{arm}_seed{seed}',['verifier_training.fd1000_run','train',str(root),arm,str(seed)],gpu=True,after=after)
    elif name=='evaluate-test':
        freeze=read(root/'TEST_FREEZE.json');targets={f'{r["arm"]}_seed{r["seed"]}' for r in freeze['runs']}|{'zero_shot'}
        require(target in targets and not (root/'test'/target).exists(),'Test evaluation needs a frozen target and runs only once')
        submit(root,'test_'+target,['verifier_training.fd1000_run','evaluate-test',str(root),target],gpu=True,after=after)
    elif name=='report':
        require(split in ('validation','test'),'Report split must be validation or test')
        if split=='test':require(all((root/'test'/t/'COMPLETE.json').exists() for t in
            [f'{a}_seed{s}' for a in receipt['arms'] for s in receipt['seeds']]+['zero_shot']),'Every frozen target needs its single test evaluation first')
        submit(root,'report_'+split,['verifier_training.v4_report',str(root),split],gpu=False,statistics=split=='test')
    else:raise ValueError('Unknown stage')


def freeze(root):
    """Record every selected adapter and the preregistration before the single test evaluation."""
    receipt=read(root/'RECEIPT.json');require(receipt['schedule']=='main','Only the Phase 2 campaign may touch the locked test')
    require(not (root/'TEST_FREEZE.json').exists(),'Already frozen');runs=[]
    for arm in receipt['arms']:
        for seed in receipt['seeds']:
            label=f'{arm}_seed{seed}';normal(read(root/(label+'_SUBMISSION.json'))['job']);result=read(root/label/'RESULT.json')
            require(result['passed'] and result['test_evaluations']==0,'Incomplete or test-contaminated run')
            runs.append(dict(arm=arm,seed=seed,result_sha256=sha(root/label/'RESULT.json'),adapter_sha256=result['adapter_sha256'],
                selected_step=result['selected_step']))
    write(root/'TEST_FREEZE.json',dict(source_commit=receipt['source_commit'],config_sha256=sha(root/'config.json'),
        prereg_sha256=sha(root/'source'/PREREG),runs=runs,frozen_utc=stamp(),
        readings='Fixed by fd1000_contract.readings and fd1000_run.diagnostic_plan at the campaign commit'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('export','create','stage','freeze'))
    p.add_argument('--dataset',type=Path);p.add_argument('--schedule',choices=('trial','main'));p.add_argument('--name')
    p.add_argument('--root',type=Path);p.add_argument('--stage');p.add_argument('--arm');p.add_argument('--seed',type=int)
    p.add_argument('--target');p.add_argument('--split');p.add_argument('--after')
    a=p.parse_args()
    if a.action=='export':export()
    elif a.action=='create':create(a.dataset,a.schedule,a.name)
    elif a.action=='stage':stage(a.root,a.stage,a.arm,a.seed,a.target,a.split,a.after)
    else:freeze(a.root)
