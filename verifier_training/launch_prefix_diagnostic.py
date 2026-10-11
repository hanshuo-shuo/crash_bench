"""Second read-only stage; no training job or training dependency is submitted."""
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
from .common import BASE,UPSTREAM,require,read,write,sha
from .launch import command
from .launch_verdict_diagnostic import CAMPAIGN,ASSETS

def main(parent):
    repo=Path.cwd();require(repo.resolve()==(Path.home()/'crash_bench').resolve(),'Wrong checkout')
    require(not command('git','status','--porcelain'),'Dirty checkout')
    commit=command('git','rev-parse','HEAD');require('origin/' in command('git','branch','-r','--contains',commit),'Unpublished source')
    require(read(parent/'diagnostic/COMPLETE.json')['passed'],'Parent incomplete')
    root=BASE/'verifier_training/runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_prefix_diagnostic_'+commit[:12])
    root.mkdir(exist_ok=False);(root/'source').mkdir()
    with (root/'source.tar').open('wb') as f:subprocess.run(['git','archive',commit],stdout=f,check=True)
    subprocess.run(['tar','-xf',str(root/'source.tar'),'-C',str(root/'source')],check=True)
    (root/'SOURCE_COMMIT').write_text(commit+'\n');(root/'source.sha256').write_text(sha(root/'source.tar')+'  '+str(root/'source.tar')+'\n')
    env=dict(os.environ,CB_VERIFIER_RUN=str(root),CB_VERIFIER_CAMPAIGN=str(CAMPAIGN),CB_VERIFIER_ASSETS=str(ASSETS),CB_DIAGNOSTIC_PARENT=str(parent))
    job=subprocess.check_output(['sbatch','--parsable','--output',str(root/'slurm_%j.log'),str(root/'source/verifier_training/prefix_diagnostic.sbatch')],env=env,text=True).strip()
    receipt={'run_root':str(root),'source_commit':commit,'upstream_commit':UPSTREAM,'campaign':str(CAMPAIGN),'assets':str(ASSETS),
        'parent':str(parent),'slurm_job':job,'scope':'Read-only native/canonical prefix diagnosis; zero training','seed':713}
    write(root/'RECEIPT.json',receipt);dest=repo/'results/verifier_training'/root.name;dest.mkdir(parents=True);write(dest/'RECEIPT.json',receipt)
    print(receipt,flush=True)

if __name__=='__main__':main(Path(sys.argv[1]))
