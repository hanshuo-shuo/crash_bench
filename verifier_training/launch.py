"""Lightweight login-node launcher, invoked only through quest_sync.sh exec."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
from .common import *


def command(*args):return subprocess.check_output(args,text=True).strip()


def main(stage,assets,prepared):
    repo=Path.cwd();require(repo.resolve()==(Path.home()/'crash_bench').resolve(),'Wrong checkout')
    require(not command('git','status','--porcelain'),'Dirty source')
    commit=command('git','rev-parse','HEAD')
    # Publishing is checked again; detached or unpublished local commits are rejected.
    branches=command('git','branch','-r','--contains',commit)
    require('origin/' in branches,'Unpublished source')
    if stage=='smoke':
        require(prepared is not None and read(prepared/'PREPARED.json')['passed'],'Preparation required')
        require(read(prepared/'dataset/AUDIT.json')['passed'],'Audit failed')
        require(read(prepared/'dataset/TOKEN_AUDIT.json')['passed'],'Token audit failed')
    require((assets/'STAGED.json').is_file(),'Stage downloads before offline compute')
    run=BASE/'verifier_training/runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+stage+'_'+commit[:12])
    run.mkdir(parents=True,exist_ok=False)
    with (run/'source.tar').open('wb') as f:subprocess.run(['git','archive',commit],stdout=f,check=True)
    (run/'source').mkdir();subprocess.run(['tar','-xf',str(run/'source.tar'),'-C',str(run/'source')],check=True)
    (run/'SOURCE_COMMIT').write_text(commit+'\n')
    (run/'source.sha256').write_text(sha(run/'source.tar')+'  '+str(run/'source.tar')+'\n')
    env=dict(os.environ,CB_VERIFIER_RUN=str(run),CB_VERIFIER_ASSETS=str(assets))
    if prepared:env['CB_VERIFIER_DATASET']=str(prepared/'dataset')
    job=subprocess.check_output(['sbatch','--parsable','--output',str(run/'slurm_%j.log'),
        str(run/'source/verifier_training'/('prepare.sbatch' if stage=='prepare' else 'smoke.sbatch'))],env=env,text=True).strip()
    receipt={'stage':stage,'run_root':str(run),'source_commit':commit,'upstream_commit':UPSTREAM,
             'assets':str(assets),'prepared':str(prepared) if prepared else None,'slurm_job':job,
             'account':'p33100','partition':'short' if stage=='prepare' else 'gengpu','seed':713,
             'max_gpu_minutes':60 if stage=='smoke' else 0,'api_calls':0,'full_training':False}
    write(run/'RECEIPT.json',receipt)
    local=repo/'results/verifier_training'/run.name;local.mkdir(parents=True,exist_ok=False)
    write(local/'RECEIPT.json',receipt)
    print(str(local.relative_to(repo)),flush=True);print(job,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','smoke']);p.add_argument('--assets',type=Path,required=True);p.add_argument('--prepared',type=Path)
    a=p.parse_args();main(a.stage,a.assets,a.prepared)
