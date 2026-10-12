"""Publish-first, gated submission of exactly one rank64 SFT comparison."""
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
from .common import BASE, UPSTREAM, read, write, sha, require
from .launch import command
from .rank_contract import PROFILE, BASELINE_NAME, BASELINE_REVIEW_NAME, BASELINE_FILES, validate_parent


def submit(root, stage):
    receipt=read(root/'RECEIPT.json')
    require(command('git','rev-parse','HEAD')==receipt['source_commit'],'Source changed')
    require(not (root/(stage+'_SUBMISSION.json')).exists(),'Stage already submitted; no silent retry')
    validate_parent(receipt)
    if stage!='preflight':require(read(root/'preflight/COMPLETE.json')['passed'],'Preflight incomplete')
    if stage in ('full','conditioning','report'):
        require(read(root/'smoke/RESULT.json')['passed'],'Balanced rank64 smoke failed')
    if stage in ('conditioning','report'):
        require(read(root/'full/RESULT.json')['passed'],'Full execution incomplete')
    if stage=='report':require(read(root/'conditioning/COMPLETE.json')['passed'],'Conditioning diagnostic incomplete')
    cpu=stage in ('preflight','report')
    limits={'preflight':'00:15:00','smoke':'00:30:00','full':'01:15:00',
            'conditioning':'00:15:00','report':'00:15:00'}
    env=dict(os.environ,CB_VERIFIER_RUN=str(root),CB_BINARY_STAGE=stage,CB_VERIFIER_ASSETS=receipt['assets'])
    args=['sbatch','--parsable','--time',limits[stage],'--job-name','cb_rank64_'+stage,
          '--output',str(root/(stage+'_%j.log')),
          str(root/'source/verifier_training'/('rank_cpu.sbatch' if cpu else 'rank_gpu.sbatch'))]
    job=subprocess.check_output(args,env=env,text=True).strip()
    result={'stage':stage,'slurm_job':job,'source_commit':receipt['source_commit'],
            'run_root':str(root),'command':args,'seed':713,
            'submitted_utc':datetime.now(timezone.utc).isoformat()}
    write(root/(stage+'_SUBMISSION.json'),result)
    print(result,flush=True)


def main():
    repo=Path.cwd();require(repo.resolve()==(Path.home()/'crash_bench').resolve(),'Wrong checkout')
    require(not command('git','status','--porcelain'),'Dirty checkout')
    commit=command('git','rev-parse','HEAD')
    require('origin/' in command('git','branch','-r','--contains',commit),'Unpublished source')
    stage=sys.argv[1]
    if stage=='create':
        parent=BASE/'verifier_training/runs'/BASELINE_NAME
        old=read(parent/'RECEIPT.json')
        review=BASE/'verifier_training/runs'/BASELINE_REVIEW_NAME
        require(read(review/'conditioning/COMPLETE.json')['passed'],'Baseline conditioning incomplete')
        root=BASE/'verifier_training/runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_rank64_sft_'+commit[:12])
        receipt={key:old[key] for key in ('campaign','assets','diagnostic','prefix_diagnostic',
                 'diagnostic_complete_sha256','prefix_complete_sha256')}
        receipt.update(run_root=str(root),source_commit=commit,upstream_commit=UPSTREAM,
            comparison_profile=PROFILE,rank16_baseline=str(parent),rank16_review=str(review),
            baseline_sha256={name:sha(parent/name) for name in BASELINE_FILES},
            baseline_conditioning_sha256=sha(review/'conditioning/SUMMARY.json'),
            seed=713,scope='One rank64/alpha128 B SFT comparison against corrected rank16/alpha32; same inputs, losses, backbone, frozen vision/projector and 300-step full schedule',
            training_authorization='User: 先继续用sft 先做更高 rank 的 LoRA 对照',
            resource_note='One A100 at a time. Smoke cap30min/full cap75min provide rank64 runtime headroom; no additional updates. No new data, paid API or hyperparameter search.')
        validate_parent(receipt)
        root.mkdir(exist_ok=False);(root/'source').mkdir()
        with (root/'source.tar').open('wb') as f:subprocess.run(['git','archive',commit],stdout=f,check=True)
        subprocess.run(['tar','-xf',str(root/'source.tar'),'-C',str(root/'source')],check=True)
        (root/'SOURCE_COMMIT').write_text(commit+'\n')
        (root/'source.sha256').write_text(sha(root/'source.tar')+'  '+str(root/'source.tar')+'\n')
        write(root/'RECEIPT.json',receipt)
        dest=repo/'results/verifier_training'/root.name;dest.mkdir(parents=True,exist_ok=False)
        write(dest/'RECEIPT.json',receipt)
        submit(root,'preflight')
    else:
        require(stage in ('smoke','full','conditioning','report'),'Unknown stage')
        submit(Path(sys.argv[2]),stage)


if __name__=='__main__':main()
