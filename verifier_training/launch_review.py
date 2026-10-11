"""Publish-first CPU review of completed engineering records, in a new output root."""
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
from .common import *
from .launch import command


def main(campaign):
    repo=Path.cwd();require(repo.resolve()==(Path.home()/'crash_bench').resolve(),'Wrong checkout')
    require(not command('git','status','--porcelain'),'Dirty checkout')
    commit=command('git','rev-parse','HEAD');require('origin/' in command('git','branch','-r','--contains',commit),'Unpublished source')
    require(read(campaign/'COMPLETE.json')['passed'],'Engineering campaign incomplete')
    root=BASE/'verifier_training/runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_review_'+commit[:12])
    root.mkdir(exist_ok=False);(root/'source').mkdir()
    with (root/'source.tar').open('wb') as f:subprocess.run(['git','archive',commit],stdout=f,check=True)
    subprocess.run(['tar','-xf',str(root/'source.tar'),'-C',str(root/'source')],check=True)
    (root/'SOURCE_COMMIT').write_text(commit+'\n')
    (root/'source.sha256').write_text(sha(root/'source.tar')+'  '+str(root/'source.tar')+'\n')
    env=dict(os.environ,CB_VERIFIER_RUN=str(root),CB_VERIFIER_CAMPAIGN=str(campaign))
    job=subprocess.check_output(['sbatch','--parsable','--output',str(root/'slurm_%j.log'),
        str(root/'source/verifier_training/review.sbatch')],env=env,text=True).strip()
    receipt={'run_root':str(root),'campaign':str(campaign),'source_commit':commit,'upstream_commit':UPSTREAM,
        'slurm_job':job,'account':'p33100','partition':'short','seed':713,
        'scope':'Supplementary output/error review only; zero new training or simulation'}
    write(root/'RECEIPT.json',receipt)
    local=repo/'results/verifier_training'/root.name;local.mkdir(parents=True,exist_ok=False);write(local/'RECEIPT.json',receipt)
    print(receipt,flush=True)


if __name__=='__main__':main(Path(sys.argv[1]))
