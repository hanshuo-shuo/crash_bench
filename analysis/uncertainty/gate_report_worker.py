"""Bounded report release after validation and the second gate shard's smoke release."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from common import atomic_json
from gate_summary_launch import launch


def main(root,gate):
    started=time.time()
    while time.time()-started<24*3600:
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
            launch(root,gate);return
        time.sleep(20)
    raise RuntimeError('Bounded report release monitor24h expired')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('gate',type=Path);a=p.parse_args()
    try:main(a.root.resolve(),a.gate.resolve())
    except BaseException as error:
        atomic_json(a.root/'REPORT_WORKER_FAILURE.json',dict(error=type(error).__name__,reason=str(error),unix=time.time()));raise
