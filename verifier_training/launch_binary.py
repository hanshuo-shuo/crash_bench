"""Explicit stage-by-stage submission: diagnostic -> tests -> smoke -> fixed fit."""
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
from .common import BASE,UPSTREAM,require,read,write,sha
from .launch import command
from .launch_verdict_diagnostic import CAMPAIGN,ASSETS


def submit(root,stage):
    receipt=read(root/'RECEIPT.json')
    require(command('git','rev-parse','HEAD')==receipt['source_commit'],'Source changed since preflight')
    require(not (root/(stage+'_SUBMISSION.json')).exists(),'Stage already submitted; no silent retry')
    if stage in ('smoke','full'):
        require(read(root/'preflight/COMPLETE.json')['passed'],'Corrected-loss tests not passed')
        require(read(Path(receipt['diagnostic'])/'diagnostic/COMPLETE.json')['passed'],'Diagnostic incomplete')
        require(read(Path(receipt['prefix_diagnostic'])/'prefix/COMPLETE.json')['passed'],'Native prefix diagnostic incomplete')
    if stage=='full':require(read(root/'smoke/RESULT.json')['passed'],'Balanced smoke did not pass')
    if stage=='report':require(read(root/'full/RESULT.json')['passed'],'Full execution incomplete')
    env=dict(os.environ,CB_VERIFIER_RUN=str(root),CB_BINARY_STAGE=stage,CB_VERIFIER_ASSETS=str(ASSETS))
    cpu=stage in ('preflight','report')
    command_line=['sbatch','--parsable','--output',str(root/(stage+'_%j.log'))]
    if not cpu:command_line+=['--time','00:20:00' if stage=='smoke' else '00:55:00']
    command_line+=[str(root/'source/verifier_training'/('binary_cpu.sbatch' if cpu else 'binary_gpu.sbatch'))]
    job=subprocess.check_output(command_line,env=env,text=True).strip()
    result={'stage':stage,'slurm_job':job,'source_commit':receipt['source_commit'],'run_root':str(root),
        'command':command_line,'seed':713,'submitted_utc':datetime.now(timezone.utc).isoformat()}
    write(root/(stage+'_SUBMISSION.json'),result)
    print(result,flush=True)


def main():
    repo=Path.cwd();require(repo.resolve()==(Path.home()/'crash_bench').resolve(),'Wrong checkout')
    require(not command('git','status','--porcelain'),'Dirty checkout')
    commit=command('git','rev-parse','HEAD');require('origin/' in command('git','branch','-r','--contains',commit),'Unpublished source')
    stage=sys.argv[1]
    if stage=='create':
        parent,prefix=map(Path,sys.argv[2:4])
        require(read(parent/'diagnostic/COMPLETE.json')['passed'],'Read-only diagnostic incomplete')
        require(read(prefix/'prefix/COMPLETE.json')['passed'],'Native prefix diagnostic incomplete')
        root=BASE/'verifier_training/runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_binary_correction_'+commit[:12])
        root.mkdir(exist_ok=False);(root/'source').mkdir()
        with (root/'source.tar').open('wb') as f:subprocess.run(['git','archive',commit],stdout=f,check=True)
        subprocess.run(['tar','-xf',str(root/'source.tar'),'-C',str(root/'source')],check=True)
        (root/'SOURCE_COMMIT').write_text(commit+'\n');(root/'source.sha256').write_text(sha(root/'source.tar')+'  '+str(root/'source.tar')+'\n')
        receipt={'run_root':str(root),'source_commit':commit,'upstream_commit':UPSTREAM,'campaign':str(CAMPAIGN),'assets':str(ASSETS),
            'diagnostic':str(parent),'prefix_diagnostic':str(prefix),
            'diagnostic_complete_sha256':sha(parent/'diagnostic/COMPLETE.json'),'prefix_complete_sha256':sha(prefix/'prefix/COMPLETE.json'),
            'scope':'One authorized B correction, original balanced8 smoke then one fresh fixed120 fit; no tuning or new simulator data',
            'seed':713,'training_authorization':'User explicitly authorized conditional smoke and one fixed120 engineering fit after diagnostics and corrected-loss tests pass'}
        write(root/'RECEIPT.json',receipt);dest=repo/'results/verifier_training'/root.name;dest.mkdir(parents=True);write(dest/'RECEIPT.json',receipt)
        submit(root,'preflight')
    else:
        require(stage in ('smoke','full','report'),'Unknown stage');submit(Path(sys.argv[2]),stage)

if __name__=='__main__':main()
