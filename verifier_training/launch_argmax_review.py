"""Publish-first, CPU-only final decision/output review in a separate run root."""
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
from .common import BASE,UPSTREAM,require,read,write,sha
from .launch import command

def main(parent):
    repo=Path.cwd();require(repo.resolve()==(Path.home()/'crash_bench').resolve(),'Wrong checkout')
    require(not command('git','status','--porcelain'),'Dirty checkout')
    commit=command('git','rev-parse','HEAD');require('origin/' in command('git','branch','-r','--contains',commit),'Unpublished source')
    require(read(parent/'full/RESULT.json')['passed'] and read(parent/'smoke/RESULT.json')['passed'],'Primary binary training incomplete')
    root=BASE/'verifier_training/runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_binary_review_'+commit[:12])
    root.mkdir(exist_ok=False);(root/'source').mkdir()
    with (root/'source.tar').open('wb') as f:subprocess.run(['git','archive',commit],stdout=f,check=True)
    subprocess.run(['tar','-xf',str(root/'source.tar'),'-C',str(root/'source')],check=True)
    (root/'SOURCE_COMMIT').write_text(commit+'\n');(root/'source.sha256').write_text(sha(root/'source.tar')+'  '+str(root/'source.tar')+'\n')
    env=dict(os.environ,CB_VERIFIER_RUN=str(root),CB_BINARY_PARENT=str(parent),CB_VERIFIER_ASSETS=read(parent/'RECEIPT.json')['assets'])
    gpu=subprocess.check_output(['sbatch','--parsable','--output',str(root/'conditioning_%j.log'),str(root/'source/verifier_training/conditioning.sbatch')],env=env,text=True).strip()
    job=subprocess.check_output(['sbatch','--parsable','--dependency','afterok:'+gpu,'--output',str(root/'review_%j.log'),str(root/'source/verifier_training/argmax_review.sbatch')],env=env,text=True).strip()
    receipt={'run_root':str(root),'source_commit':commit,'upstream_commit':UPSTREAM,'parent':str(parent),
        'slurm_jobs':{'read_only_conditioning':gpu,'cpu_report':job},'seed':713,'scope':'Read-only decision-token and generated-output review. No new training or simulator data.'}
    write(root/'RECEIPT.json',receipt);dest=repo/'results/verifier_training'/root.name;dest.mkdir(parents=True);write(dest/'RECEIPT.json',receipt)
    print(receipt,flush=True)

if __name__=='__main__':main(Path(sys.argv[1]))
