"""Submit a CPU-only report of a terminal failed rank64 smoke, in a new root."""
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
from .common import BASE,UPSTREAM,read,write,require,sha
from .launch import command
from .rank_contract import validate_parent


def main(parent):
    repo=Path.cwd();require(repo.resolve()==(Path.home()/'crash_bench').resolve(),'Wrong checkout')
    require(not command('git','status','--porcelain'),'Dirty checkout')
    commit=command('git','rev-parse','HEAD');require('origin/' in command('git','branch','-r','--contains',commit),'Unpublished source')
    receipt=read(parent/'RECEIPT.json');validate_parent(receipt)
    require(read(parent/'smoke/RESULT.json')['passed'] is False,'Smoke did not fail')
    job=read(parent/'smoke_SUBMISSION.json')['slurm_job']
    require(not command('squeue','-h','-u',os.environ['USER'],'-A','p33100','-j',job),'Smoke job not terminal')
    root=BASE/'verifier_training/runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_rank64_smoke_review_'+commit[:12])
    root.mkdir(exist_ok=False);(root/'source').mkdir()
    with (root/'source.tar').open('wb') as f:subprocess.run(['git','archive',commit],stdout=f,check=True)
    subprocess.run(['tar','-xf',str(root/'source.tar'),'-C',str(root/'source')],check=True)
    (root/'SOURCE_COMMIT').write_text(commit+'\n');(root/'source.sha256').write_text(sha(root/'source.tar')+'  '+str(root/'source.tar')+'\n')
    env=dict(os.environ,CB_VERIFIER_RUN=str(root),CB_RANK_PARENT=str(parent),CB_VERIFIER_ASSETS=receipt['assets'])
    result={'run_root':str(root),'source_commit':commit,'upstream_commit':UPSTREAM,'parent_run':str(parent),
        'parent_training_commit':receipt['source_commit'],'parent_result_sha256':sha(parent/'smoke/RESULT.json'),
        'seed':713,'scope':'CPU report only of failed bounded smoke; no training, inference or simulation'}
    write(root/'RECEIPT.json',result)
    submitted=subprocess.check_output(['sbatch','--parsable','--output',str(root/'report_%j.log'),
        str(root/'source/verifier_training/rank_smoke_review.sbatch')],env=env,text=True).strip()
    result['slurm_job']=submitted;write(root/'RECEIPT.json',result)
    dest=repo/'results/verifier_training'/root.name;dest.mkdir(parents=True,exist_ok=False);write(dest/'RECEIPT.json',result)
    print(result,flush=True)


if __name__=='__main__':main(Path(sys.argv[1]))
