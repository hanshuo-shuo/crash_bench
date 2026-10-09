"""Stage validation, then depend the final outcome/report job on both gate shards."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from common import ASSETS,atomic_json
from preflight import stream_sha
from launch_integrity import remaining_dependencies
from audit_shard import terminal_accounting


def launch(root,gate,validate=False):
    source=root/'source';checkout=Path.home()/'crash_bench'
    if Path(__file__).resolve().parents[2]!=source.resolve() or root.parent!=(ASSETS/'uncertainty').resolve() or gate.parent!=(ASSETS/'uncertainty').resolve():raise RuntimeError('Immutable own-project roots required')
    def git(*args):return subprocess.check_output(['git',*args],cwd=checkout)
    commit=(root/'SOURCE_COMMIT').read_text().strip()
    if git('status','--porcelain') or git('rev-parse','origin/codex/feasibility').decode().strip()!=commit:raise RuntimeError('Clean published summary code required')
    receipt=root/('VALIDATION_SUBMITTED.json' if validate else 'SUBMITTED.json')
    if receipt.exists():raise RuntimeError('Never resubmit a summary stage')
    files=[]
    for name in git('ls-tree','-r','--name-only',commit).decode().splitlines():
        if (source/name).read_bytes()!=git('show',commit+':'+name):raise RuntimeError('Summary source differs')
        files.append(dict(path=name,sha256=stream_sha(source/name)))
    if validate:atomic_json(root/'SOURCE_FILES.json',files)
    elif json.loads((root/'SOURCE_FILES.json').read_text())!=files:raise RuntimeError('Validation source identities changed')
    plan=json.loads((gate/'SUBMITTED.json').read_text());task2=Path(plan['analysis_root'])/'analysis'
    if validate:dependencies=[];completed=[]
    else:
        validation=json.loads((root/'VALIDATION_PASS.json').read_text())
        if not validation['passed'] or validation['code_commit']!=commit:raise RuntimeError('Allocated summary/report fixture tests required')
        terminal_accounting(validation['job'])
        if (gate/'STOP.json').exists():raise RuntimeError('Gate stopped')
        jobs=[json.loads((gate/'shards'/str(s)/'SUBMITTED.json').read_text())['job'] for s in [0,1]]
        dependencies,completed=remaining_dependencies(jobs)
    env=dict(os.environ,CB_UNCERTAINTY_ROOT=str(root),CB_GATE_ROOT=str(gate),CB_TASK2_ANALYSIS=str(task2),CB_VALIDATE_ONLY='1' if validate else '0')
    arguments=['sbatch','--parsable','--output='+str(root/('validation_%j.log' if validate else 'slurm_%j.log'))]
    if dependencies:arguments.append('--dependency=afterok:'+':'.join(dependencies))
    arguments.append(str(source/'analysis/uncertainty/gate_summary.sbatch'))
    job=subprocess.check_output(arguments,env=env,cwd=checkout,text=True).strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Invalid summary Slurm receipt')
    submitted=dict(job=job,root=str(root),gate_root=str(gate),task2_analysis=str(task2),code_commit=commit,stage='synthetic_gate_validation' if validate else 'real300_gate_summary_report',submitted_unix=time.time(),dependency_jobs=dependencies,completed_dependencies=completed,
        resources=dict(account='p33100',partition='short',cpus=2,memory_gb=8,minutes=60),new_rollouts=0,api_calls=0)
    atomic_json(receipt,submitted);print(json.dumps(submitted,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('gate',type=Path);p.add_argument('--validate',action='store_true');a=p.parse_args();launch(a.root.resolve(),a.gate.resolve(),a.validate)
