"""Stage scientific verifier work only after the final FD1000 acceptance gate."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
from .common import BASE,UPSTREAM,MODEL_ID,read,write,sha,require
from .launch import command
from .launch_verdict_diagnostic import ASSETS
from .fd1000_contract import configuration


def normal(job):
    require(command('sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode')==job+'|COMPLETED|0:0',
            'Prerequisite job has not finished normally')


def submit(root,stage):
    receipt=read(root/'RECEIPT.json')
    require(command('git','rev-parse','HEAD')==receipt['source_commit'],'Use the same published training source for every stage')
    require(not (root/(stage+'_SUBMISSION.json')).exists(),'No silent resubmission or output overwrite')
    require(sha(Path(receipt['dataset'])/'ACCEPTANCE.json')==receipt['dataset_acceptance_sha256'],'Data approval changed')
    if stage in ('A','B'):
        prior=read(root/'preflight_SUBMISSION.json');normal(prior['job'])
        require(read(root/'preflight/COMPLETE.json')['passed'],'Token/mask preflight did not pass')
        active=command('squeue','-h','-u',os.environ['USER'],'-A','p33100','-p','gengpu','-o','%j|%Z')
        require(sum('crash_bench' in line or 'crashbench_safelibero' in line or line.startswith('cb_') for line in active.splitlines())<2,
                'Two CrashBench GPU jobs already active')
    env=dict(os.environ,CB_VERIFIER_RUN=str(root),CB_VERIFIER_ASSETS=str(ASSETS),CB_FD_TRAIN_STAGE=stage)
    script='fd1000_cpu.sbatch' if stage=='preflight' else 'fd1000_gpu.sbatch'
    job=subprocess.check_output(['sbatch','--parsable','--output='+str(root/(stage+'_%j.log')),
        str(root/'source/verifier_training'/script)],env=env,text=True).strip().split(';')[0]
    require(job.isdigit(),'Invalid Slurm receipt')
    result=dict(stage=stage,job=job,run_root=str(root),source_commit=receipt['source_commit'],dataset=receipt['dataset'],api_calls=0)
    write(root/(stage+'_SUBMISSION.json'),result);print(result,flush=True)


def main(stage,dataset=None,root=None):
    repo=Path.cwd().resolve();require(repo==(Path.home()/'crash_bench').resolve(),'Wrong project checkout')
    require(command('git','remote','get-url','origin') in ('git@github.com:hanshuo-shuo/crash_bench.git','https://github.com/hanshuo-shuo/crash_bench.git'),'Wrong origin')
    require(not command('git','status','--porcelain'),'Dirty source')
    commit=command('git','rev-parse','HEAD');require('origin/' in command('git','branch','-r','--contains',commit),'Source must be published')
    if stage=='create':
        require(dataset is not None,'Explicit accepted dataset required');dataset=dataset.resolve()
        dataset.relative_to(BASE/'verifier_dataset')
        parent=read(dataset.parent/'RECEIPT.json');normal(parent['job'])
        require(parent['stage']=='finalize-constraints' and read(dataset.parent/'COMPLETE.json')['passed'],'Final scientific data export incomplete')
        approval=read(dataset/'ACCEPTANCE.json')
        require(approval.get('passed') is True and approval.get('scientific_training_allowed') is True,'No scientific data acceptance')
        require(sha(dataset/'ACCEPTANCE.json')==read(dataset.parent/'COMPLETE.json')['acceptance_sha256'],'Data acceptance anchor differs')
        require(read(ASSETS/'STAGED.json')['model_id']==MODEL_ID,'Wrong staged VLM; no substitution permitted')
        root=BASE/'verifier_training/runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_fd1000_'+commit[:12])
        root.mkdir();(root/'source').mkdir()
        with (root/'source.tar').open('wb') as f:subprocess.run(['git','archive',commit],stdout=f,check=True)
        subprocess.run(['tar','-xf',str(root/'source.tar'),'-C',str(root/'source')],check=True)
        (root/'SOURCE_COMMIT').write_text(commit+'\n');(root/'source.sha256').write_text(sha(root/'source.tar')+'  '+str(root/'source.tar')+'\n')
        write(root/'config.json',configuration())
        receipt=dict(source_commit=commit,upstream_commit=UPSTREAM,run_root=str(root),dataset=str(dataset),assets=str(ASSETS),
            dataset_acceptance_sha256=sha(dataset/'ACCEPTANCE.json'),assets_stage_sha256=sha(ASSETS/'STAGED.json'),
            configuration_sha256=sha(root/'config.json'),seed=713,api_calls=0,
            authorization='User authorized verifier training only after the 1000-row evidence, grounding and shortcut audit passes',
            scope='Fresh direct/generative language LoRA, three complete epochs, validation checkpoint selection, locked source test once')
        write(root/'RECEIPT.json',receipt);local=repo/'results/verifier_training'/root.name;local.mkdir(parents=True);write(local/'RECEIPT.json',receipt)
        print(receipt,flush=True);submit(root,'preflight')
    else:
        require(root is not None,'Explicit frozen training run required');submit(root,stage)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('create','preflight','A','B'));p.add_argument('--dataset',type=Path);p.add_argument('--root',type=Path)
    a=p.parse_args();main(a.stage,a.dataset,a.root)
