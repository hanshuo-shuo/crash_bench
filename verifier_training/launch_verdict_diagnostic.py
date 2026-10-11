"""Create a uniquely archived, published-source, read-only Quest diagnosis."""
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
from .common import BASE, UPSTREAM, read, write, sha, require
from .launch import command

CAMPAIGN=BASE/'verifier_training/runs/20261011T001506Z_engineering_27e7fe7cf4e3'
ASSETS=BASE/'verifier_training/assets/20261010_d94c245'

def main():
    repo=Path.cwd(); require(repo.resolve()==(Path.home()/'crash_bench').resolve(),'Wrong checkout')
    require(not command('git','status','--porcelain'),'Dirty checkout')
    commit=command('git','rev-parse','HEAD');require('origin/' in command('git','branch','-r','--contains',commit),'Unpublished commit')
    require(read(CAMPAIGN/'COMPLETE.json')['passed'],'Parent incomplete')
    root=BASE/'verifier_training/runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_verdict_diagnostic_'+commit[:12])
    root.mkdir(exist_ok=False);(root/'source').mkdir()
    with (root/'source.tar').open('wb') as f:subprocess.run(['git','archive',commit],stdout=f,check=True)
    subprocess.run(['tar','-xf',str(root/'source.tar'),'-C',str(root/'source')],check=True)
    (root/'SOURCE_COMMIT').write_text(commit+'\n');(root/'source.sha256').write_text(sha(root/'source.tar')+'  '+str(root/'source.tar')+'\n')
    env=dict(os.environ,CB_VERIFIER_RUN=str(root),CB_VERIFIER_CAMPAIGN=str(CAMPAIGN),CB_VERIFIER_ASSETS=str(ASSETS))
    job=subprocess.check_output(['sbatch','--parsable','--output',str(root/'slurm_%j.log'),str(root/'source/verifier_training/verdict_diagnostic.sbatch')],env=env,text=True).strip()
    receipt={'run_root':str(root),'source_commit':commit,'upstream_commit':UPSTREAM,'campaign':str(CAMPAIGN),
        'slurm_job':job,'scope':'Read-only checkpoint/token/loss/gradient diagnosis. No training submitted.','seed':713}
    write(root/'RECEIPT.json',receipt); dest=repo/'results/verifier_training'/root.name;dest.mkdir(parents=True);write(dest/'RECEIPT.json',receipt)
    print(receipt,flush=True)

if __name__=='__main__':main()
