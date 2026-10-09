"""Queue fixed Task2 smoke/full analysis after the independent full audit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from common import ASSETS,atomic_json,config
from launch_integrity import remaining_dependencies
from preflight import stream_sha


def launch(root,audit):
    source=root/'source';checkout=Path.home()/'crash_bench'
    if Path(__file__).resolve().parents[2]!=source.resolve() or root.parent!=(ASSETS/'uncertainty').resolve() or audit.parent!=(ASSETS/'uncertainty').resolve():raise RuntimeError('Immutable analysis and project audit roots required')
    def git(*args):return subprocess.check_output(['git',*args],cwd=checkout)
    commit=(root/'SOURCE_COMMIT').read_text().strip()
    if git('status','--porcelain') or git('rev-parse','origin/codex/feasibility').decode().strip()!=commit:raise RuntimeError('Analysis code must be clean and published')
    if (root/'SUBMITTED.json').exists():raise RuntimeError('Never resubmit an analysis root')
    receipt=json.loads((audit/'SUBMITTED.json').read_text())
    if receipt['shard'] is not None or not str(receipt['job']).isdigit():raise RuntimeError('Full audit job required')
    dependencies,completed=remaining_dependencies([receipt['job']]);files=[]
    for name in git('ls-tree','-r','--name-only',commit).decode().splitlines():
        if (source/name).read_bytes()!=git('show',commit+':'+name):raise RuntimeError('Analysis source differs')
        files.append(dict(path=name,sha256=stream_sha(source/name)))
    atomic_json(root/'SOURCE_FILES.json',files)
    cfg=config();resources=cfg['task2']['resources']
    if resources!=dict(account='p33100',partition='short',cpus=2,memory_gb=8,minutes=60):raise RuntimeError('Analysis resource envelope changed')
    env=dict(os.environ,CB_UNCERTAINTY_ROOT=str(root),CB_UNCERTAINTY_AUDITED=str(audit/'full_audit'))
    arguments=['sbatch','--parsable','--time='+str(resources['minutes']),'--output='+str(root/'slurm_%j.log')]
    if dependencies:arguments.append('--dependency=afterok:'+':'.join(dependencies))
    arguments.append(str(source/'analysis/uncertainty/analysis.sbatch'))
    job=subprocess.check_output(arguments,env=env,cwd=checkout,text=True).strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Invalid analysis job receipt')
    submitted=dict(job=job,root=str(root),input_audit=str(audit),input_collection=receipt['collection'],code_commit=commit,
        submitted_unix=time.time(),resources=resources,dependency_jobs=dependencies,completed_dependencies=completed,
        configuration=cfg['task2'],stats_plan_sha256=stream_sha(source/'analysis/uncertainty/STATS_PLAN.md'),split_sha256=stream_sha(source/'analysis/uncertainty/split.json'),
        scope='Complete1200-case audit guard; two-scene two-seed heldout smoke with fixed full train fit, then full Task2; no API/newrollouts/threshold search')
    atomic_json(root/'SUBMITTED.json',submitted);print(json.dumps(submitted,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('audit',type=Path);a=p.parse_args();launch(a.root.resolve(),a.audit.resolve())
