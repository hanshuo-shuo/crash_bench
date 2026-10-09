"""Launch the authorized gate within remaining series GPU time, using frozen code."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time
from common import ASSETS,atomic_json,config
from preflight import stream_sha
from audit_shard import terminal_accounting


def submit(root,shard):
    plan=json.loads((root/'SUBMITTED.json').read_text());directory=root/'shards'/str(shard)
    if (directory/'SUBMITTED.json').exists():raise RuntimeError('Never resubmit a gate shard')
    env=dict(os.environ,CB_UNCERTAINTY_ROOT=str(directory))
    job=subprocess.check_output(['sbatch','--parsable','--time='+str(plan['configuration']['resources']['minutes']),
        '--output='+str(directory/'slurm_%j.log'),str(root/'source/analysis/uncertainty/gate.sbatch')],env=env,cwd=Path.home()/'crash_bench',text=True).strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Invalid gate Slurm receipt')
    atomic_json(directory/'SUBMITTED.json',dict(job=job,shard=shard,code_commit=plan['code_commit'],submitted_unix=time.time(),resources=plan['configuration']['resources']))
    return job


def launch(root,analysis):
    source=root/'source';checkout=Path.home()/'crash_bench'
    if Path(__file__).resolve().parents[2]!=source.resolve() or root.parent!=(ASSETS/'uncertainty').resolve() or analysis.parent!=(ASSETS/'uncertainty').resolve():raise RuntimeError('Valid immutable project roots required')
    def git(*args):return subprocess.check_output(['git',*args],cwd=checkout)
    commit=(root/'SOURCE_COMMIT').read_text().strip()
    if git('status','--porcelain') or git('rev-parse','origin/codex/feasibility').decode().strip()!=commit:raise RuntimeError('Gate code must be clean and published')
    if (root/'SUBMITTED.json').exists():raise RuntimeError('Never resubmit a gate root')
    a=json.loads((analysis/'SUBMITTED.json').read_text());terminal_accounting(a['job'])
    if not json.loads((analysis/'ANALYSIS_PASS.json').read_text())['passed']:raise RuntimeError('Task2 smoke/full analysis pass required')
    completion=json.loads((analysis/'analysis/ANALYSIS_COMPLETE.json').read_text())
    if not completion['passed'] or completion['stage']!='full' or completion['test_rollouts']!=360:raise RuntimeError('Full fixed Task2 required')
    for name,digest in completion['files'].items():
        if stream_sha(analysis/'analysis'/name)!=digest:raise RuntimeError('Task2 output bytes changed')
    thresholds=json.loads((analysis/'analysis/TRAIN_THRESHOLDS.json').read_text());cutoff=thresholds['gate_q90']
    if not math.isfinite(cutoff) or thresholds['train_states']!=42 or thresholds['test_used']:raise RuntimeError('Train-only finite cutoff required')
    collection=Path(a['input_collection']);original=json.loads((collection/'SUBMITTED.json').read_text())
    cfg=config('full');gate=cfg['task3'];resources=gate['resources'];accounting=[]
    for job in gate['prior_smoke_gpu_jobs']+original['jobs']:
        result=subprocess.check_output(['sacct','-j',str(job),'--format=JobID,State,ExitCode,ElapsedRaw','-n','-P'],text=True)
        rows=[x.split('|') for x in result.splitlines() if x.split('|')[0]==str(job)]
        if len(rows)!=1:raise RuntimeError('Series GPU accounting unavailable')
        _,state,exit_code,elapsed=rows[0]
        if state not in ['COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','DEADLINE','PREEMPTED']:raise RuntimeError('Series GPU allocation still active or unknown')
        if job in original['jobs'] and (state!='COMPLETED' or exit_code!='0:0'):raise RuntimeError('Task1 GPU collection not successfully terminal')
        accounting.append(dict(job=str(job),state=state,exit_code=exit_code,elapsed_seconds=int(elapsed)))
    used=sum(x['elapsed_seconds'] for x in accounting);requested=resources['workers']*resources['minutes']*60
    if resources!=dict(account='p33100',partition='gengpu',gpu='a100:1',cpus=8,memory_gb=64,minutes=360,workers=2) or used+requested>gate['series_ceiling_gpu_seconds']:raise RuntimeError('Gate exceeds authorized remaining48 GPUh pool')
    files=[]
    for name in git('ls-tree','-r','--name-only',commit).decode().splitlines():
        if (source/name).read_bytes()!=git('show',commit+':'+name):raise RuntimeError('Gate source differs')
        files.append(dict(path=name,sha256=stream_sha(source/name)))
    atomic_json(root/'SOURCE_FILES.json',files)
    fingerprints=json.loads((collection/'FINGERPRINTS.json').read_text());atomic_json(root/'FINGERPRINTS.json',fingerprints)
    base=ASSETS/'envs/aegis_sim/lib/python3.8/site-packages/robosuite'
    controllers=[dict(path=str(base/p),sha256=stream_sha(base/p)) for p in ['controllers/osc.py','controllers/base_controller.py','models/grippers/panda_gripper.py','robots/single_arm.py','robots/manipulator.py']]
    cfg.update(mode='gate',repeats=gate['repeats'],resources=resources,gate_threshold=cutoff,controller_files=controllers,collection_root=str(collection),audited_input=str(Path(a['input_audit'])/'full_audit'),max_fresh_calls_per_run_root=0,api_limit_usd='0.00')
    plan=dict(root=str(root),code_commit=commit,upstream_root=original['upstream_root'],configuration=cfg,analysis_root=str(analysis),threshold_file_sha256=stream_sha(analysis/'analysis/TRAIN_THRESHOLDS.json'),
        prior_gpu_accounting=accounting,prior_gpu_seconds=used,requested_gpu_seconds=requested,maximum_series_gpu_seconds=used+requested,authorized_ceiling_gpu_seconds=gate['series_ceiling_gpu_seconds'],expected_rollouts=300,
        scope='Fixed train nominal q90;60×5 nominal zero-command gate;4-case included smoke then two workers; no VLM/no threshold search/no mid-state branch')
    (root/'shards').mkdir()
    for shard in [0,1]:
        directory=root/'shards'/str(shard);directory.mkdir()
        atomic_json(directory/'plan.json',dict(configuration=cfg,code_commit=commit,upstream_root=original['upstream_root'],shard=shard))
    atomic_json(root/'SUBMITTED.json',plan);job=submit(root,0)
    atomic_json(root/'FIRST_SHARD_SUBMITTED.json',dict(job=job,second_shard_held_until_smoke=True));print(json.dumps(dict(root=str(root),job=job,threshold=cutoff,prior_gpu_seconds=used,maximum_series_gpu_seconds=used+requested),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('analysis',type=Path);a=p.parse_args();launch(a.root.resolve(),a.analysis.resolve())
