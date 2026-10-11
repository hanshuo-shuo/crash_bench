"""Exact-commit Quest launcher; audit, fit and probes are separate gated stages."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
from .common import *
from .engineering_contract import execution_config,DATA_SHA
from .launch import command


def main(a):
    repo=Path.cwd();require(repo.resolve()==(Path.home()/'crash_bench').resolve(),'Wrong checkout')
    require(not command('git','status','--porcelain'),'Dirty source')
    commit=command('git','rev-parse','HEAD');require('origin/' in command('git','branch','-r','--contains',commit),'Unpublished source')
    if a.stage=='audit':
        require(a.assets and a.prepared and a.smoke,'Existing assets, preparation and passed smoke required')
        require(read(a.prepared/'PREPARED.json')['passed'],'Preparation failed')
        require(sha(a.prepared/'dataset/dataset.jsonl')==DATA_SHA,'Wrong evidence dataset')
        for v in ('A','B'):
            result=read(a.smoke/v/'RESULT.json')
            require(result['passed'] and result['exact_outputs']==8,'Eight-sample smoke must pass')
            require(read(a.smoke/v/'PROVENANCE.json')['input_contract']==INPUT_CONTRACT,'Stale input contract')
        root=BASE/'verifier_training/runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_engineering_'+commit[:12])
        root.mkdir(exist_ok=False);(root/'source').mkdir()
        with (root/'source.tar').open('wb') as f:subprocess.run(['git','archive',commit],stdout=f,check=True)
        subprocess.run(['tar','-xf',str(root/'source.tar'),'-C',str(root/'source')],check=True)
        (root/'SOURCE_COMMIT').write_text(commit+'\n');(root/'source.sha256').write_text(sha(root/'source.tar')+'  '+str(root/'source.tar')+'\n')
        for v in ('A','B'):write(root/('execution_'+v+'.json'),execution_config(read(repo/f'verifier_training/configs/proposed_{v}.json')))
        manifest={'run_root':str(root),'code_commit':commit,'upstream_commit':UPSTREAM,'assets':str(a.assets),
            'prepared':str(a.prepared),'smoke':str(a.smoke),'dataset_sha256':DATA_SHA,'seed':713,
            'authorization':'User requested full120 fixed A/B engineering fit, observability audit and shortcut diagnostics',
            'training':{'epochs':10,'updates':300,'hyperparameter_search':False},'new_simulation':False,'scientific_generalization_claim':False}
        write(root/'CAMPAIGN.json',manifest);stages=['audit']
    else:
        root=a.campaign;require(root is not None,'Campaign required');manifest=read(root/'CAMPAIGN.json')
        require(manifest['code_commit']==commit,'Keep campaign checkout frozen until compute completes')
        if a.stage=='fit':
            require(read(root/'audit/AUDIT.json')['passed_for_engineering_only'],'Audit/isolation must pass before full fit')
            require(read(root/'dataset/TOKEN_AUDIT.json')['count']==240,'All A/B encodings required')
            stages=['fit_A','fit_B']
        elif a.stage=='diagnose':
            require(all(read(root/v/'RESULT.json')['steps']==300 for v in ('A','B')),'Both fixed fits must finish')
            stages=['diagnose_A','diagnose_B']
        else:
            require(all(read(root/('diagnostics_'+v)/'COMPLETE.json')['passed'] for v in ('A','B')),'Diagnostics incomplete')
            stages=['report']
    local=repo/'results/verifier_training'/root.name;local.mkdir(parents=True,exist_ok=True)
    write(local/'CAMPAIGN.json',manifest)
    for stage in stages:
        receipt=root/(stage+'_RECEIPT.json');require(not receipt.exists(),'Stage already submitted')
        gpu=stage.startswith(('fit_','diagnose_'))
        env=dict(os.environ,CB_VERIFIER_RUN=str(root),CB_VERIFIER_ASSETS=manifest['assets'],
                 CB_VERIFIER_DATASET=str(Path(manifest['prepared'])/'dataset'),CB_ENGINEERING_STAGE=stage)
        job=command_with_env(env,'sbatch','--parsable','--job-name','cb_verifier_'+stage,
            '--output',str(root/(stage+'_%j.log')),str(root/'source/verifier_training'/('engineering_gpu.sbatch' if gpu else 'engineering_cpu.sbatch')))
        record={'stage':stage,'job':job,'run_root':str(root),'code_commit':commit,'account':'p33100',
            'partition':'gengpu' if gpu else 'short','wall_minutes':60 if gpu else 30,'seed':713,'api_calls':0}
        write(receipt,record);write(local/receipt.name,record);print(record,flush=True)
    print('CAMPAIGN',root,flush=True)


def command_with_env(env,*args):return subprocess.check_output(args,env=env,text=True).strip()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['audit','fit','diagnose','report'])
    for name in ('assets','prepared','smoke','campaign'):p.add_argument('--'+name,type=Path)
    main(p.parse_args())
