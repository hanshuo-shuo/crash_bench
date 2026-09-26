"""Bounded network-only worker; stop our Slurm arrays on budget/infrastructure failure."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from api_budget import Budget,atomic_json
from openrouter_perception import fill
from safelibero_batch import collect

def main(root):
    plan=json.loads((root/'batch.json').read_text());jobs=plan['slurm_arrays'];budget=None
    try:
        budget=Budget(root,plan['prior_api_spend_usd']);budget.refresh_prices()
        atomic_json(root/'WORKER_READY.json',{'started_unix':time.time()})
        deadline=time.monotonic()+7*86400
        last_slurm_check=0
        while time.monotonic()<deadline:
            summary=collect(root);summary['reported_and_reserved_cost_usd']=str(budget.committed())
            summary['updated_unix']=time.time();atomic_json(root/'status.json',summary)
            if summary['failed_cells']:raise RuntimeError('GPU infrastructure failure in cells '+str(summary['failed_cells']))
            if len(summary['complete_cells'])==64:
                atomic_json(root/'COMPLETE.json',summary);return
            if (root/'STOP.json').exists():raise RuntimeError('Explicit batch stop requested')
            if time.monotonic()-last_slurm_check>=60:
                state=subprocess.run(['sacct','-X','-n','-P','-j',','.join(jobs),'--format=JobIDRaw,State'],text=True,capture_output=True,check=True).stdout
                bad=[line for line in state.splitlines() if any(s in line for s in ['FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'])]
                if bad:raise RuntimeError('Slurm failed: '+str(bad[:4]))
                last_slurm_check=time.monotonic()
            for request in sorted((root/'perception_cache').glob('*/request.json')):
                if (request.parent/'response.json').exists():continue
                obstacle=fill(request.parent,budget=budget)
                print(json.dumps({'request_sha256':request.parent.name,'obstacle':obstacle,'cost_with_reservations_usd':str(budget.committed())}),flush=True)
            time.sleep(5)
        raise RuntimeError('Seven-day worker deadline reached')
    except BlockingIOError:
        raise  # A duplicate worker must not cancel the legitimate owner's jobs.
    except BaseException as e:
        atomic_json(root/'STOP.json',{'reason':str(e),'unix':time.time(),'cost_with_reservations_usd':str(budget.committed()) if budget else None})
        subprocess.run(['scancel',*jobs],check=False)
        raise
    finally:
        if budget:budget.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args();main(args.root)
