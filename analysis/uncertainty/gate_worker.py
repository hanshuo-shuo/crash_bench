"""Bounded own-job monitor releases shard1 only after the four-case gate smoke."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from common import atomic_json
from gate_launch import submit
from audit_shard import terminal_accounting


def state(job):
    result=subprocess.run(['squeue','-j',job,'-h','-o','%T'],capture_output=True,text=True)
    if result.returncode==0 and result.stdout.strip():return result.stdout.strip()
    result=subprocess.run(['sacct','-j',job,'--format=JobID,State','-n','-P'],capture_output=True,text=True,check=True)
    rows=[x.split('|')[1] for x in result.stdout.splitlines() if x.split('|')[0]==job]
    if len(rows)!=1:raise RuntimeError('Gate own-job accounting unavailable')
    return rows[0]


def main(root):
    jobs={};started=time.time()
    try:
        while True:
            for shard in [0,1]:
                p=root/'shards'/str(shard);receipt=p/'SUBMITTED.json'
                if receipt.exists():jobs[shard]=json.loads(receipt.read_text())['job']
                if (p/'STOP.json').exists() or (p/'PARTIAL.json').exists():raise RuntimeError('Gate shard failure/whole-case partial checkpoint; preserve executed cases')
            status={str(k):dict(job=j,state=state(j)) for k,j in jobs.items()}
            atomic_json(root/'WORKER_STATUS.json',dict(checked_unix=time.time(),jobs=status,new_vlm_calls=0))
            if any(x['state'] not in ['PENDING','RUNNING','CONFIGURING','COMPLETING','COMPLETED'] for x in status.values()):raise RuntimeError('Gate own Slurm job failed or unknown')
            smoke=root/'shards/0/GATE_SMOKE_PASS.json'
            if 1 not in jobs and smoke.exists():
                proof=json.loads(smoke.read_text())
                if not proof['passed'] or proof['runs']!=4 or not proof['independent_all_sample_reduction'] or not proof['paired_settled_physical_arrays_exact']:raise RuntimeError('Gate smoke proof incomplete')
                jobs[1]=submit(root,1);atomic_json(root/'SECOND_SHARD_RELEASED.json',dict(job=jobs[1],smoke=proof,released_unix=time.time()))
            if len(jobs)==2 and all(state(j)=='COMPLETED' for j in jobs.values()):
                accounting=[terminal_accounting(j) for j in jobs.values()]
                receipts=[json.loads((root/'shards'/str(s)/'COMPLETE.json').read_text()) for s in [0,1]]
                if any(not x['passed'] or x['runs']!=150 for x in receipts):raise RuntimeError('Incomplete300 gate matrix')
                atomic_json(root/'GATE_COLLECTION_COMPLETE.json',dict(passed=True,runs=300,jobs=jobs,accounting=accounting,actions=sum(x['actions'] for x in receipts),inferences=sum(x['inferences'] for x in receipts),new_vlm_calls=0));return
            if time.time()-started>24*3600:raise RuntimeError('Bounded gate monitor24h expired')
            time.sleep(20)
    except BaseException as error:
        atomic_json(root/'STOP.json',dict(error=type(error).__name__,reason=str(error),jobs=jobs,unix=time.time()))
        for job in jobs.values():
            try:active=state(job) in ['PENDING','RUNNING','CONFIGURING','COMPLETING']
            except BaseException:active=True
            if active:subprocess.run(['scancel',job],check=False)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();main(a.root.resolve())
