"""Bounded report release after validation and the second gate shard's smoke release."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from common import atomic_json


def main(root,gate,watch_hours=72):
    if not 1<=watch_hours<=72:raise ValueError('Bounded monitor duration1..72h required')
    started=time.time()
    while time.time()-started<watch_hours*3600:
        if (gate/'STOP.json').exists():raise RuntimeError('Gate campaign stopped; no report release')
        if (root/'SUBMITTED.json').exists():return
        validation=root/'VALIDATION_PASS.json';second=gate/'shards/1/SUBMITTED.json'
        atomic_json(root/'REPORT_WORKER_STATUS.json',dict(checked_unix=time.time(),validation_ready=validation.exists(),second_gate_shard_submitted=second.exists(),new_rollouts=0,api_calls=0))
        v=root/'VALIDATION_SUBMITTED.json'
        if v.exists():
            job=json.loads(v.read_text())['job'];result=subprocess.check_output(['sacct','-j',job,'--format=JobID,State,ExitCode','-n','-P'],text=True)
            rows=[r.split('|') for r in result.splitlines() if r.split('|')[0]==job]
            if len(rows)==1 and rows[0][1] not in ['PENDING','RUNNING','CONFIGURING','COMPLETING','COMPLETED']:raise RuntimeError('Report validation job failed')
        if validation.exists() and second.exists():
            subprocess.run(['/projects/p33100/siosio/envs/openpi/bin/python',str(root/'source/analysis/uncertainty/gate_summary_launch.py'),str(root),str(gate)],check=True);return
        time.sleep(20)
    raise RuntimeError('Bounded report release monitor expired after%d wall hours'%watch_hours)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('gate',type=Path);p.add_argument('--watch-hours',type=int,default=72);a=p.parse_args()
    try:main(a.root.resolve(),a.gate.resolve(),a.watch_hours)
    except BaseException as error:
        atomic_json(a.root/'REPORT_WORKER_FAILURE.json',dict(error=type(error).__name__,reason=str(error),unix=time.time()));raise
