"""Bounded network-only worker; stop our Slurm arrays on budget/infrastructure failure."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import hashlib
import urllib.error
from decimal import Decimal
from api_budget import Budget,atomic_json
from openrouter_perception import fill,read_response,request_hash,MODEL
from safelibero_batch import collect

def reuse_cache(directory,sources):
    request=json.loads((directory/'request.json').read_text())
    if request_hash(request)!=directory.name:raise RuntimeError('Request hash mismatch')
    for source in sources:
        old=Path(source)/directory.name
        if not (old/'response.json').exists():continue
        if json.loads((old/'request.json').read_text())!=request:raise RuntimeError('Inherited cache identity mismatch')
        read_response(old)
        raw=(old/'response.json').read_bytes();response=json.loads(raw)['response']
        if response.get('model')!=MODEL or response.get('provider')!='Z.AI' or response['choices'][0].get('finish_reason')!='stop':
            raise RuntimeError('Inherited cache is not a complete pinned-provider response')
        atomic_json(directory/'cache_origin.json',{'source':str(old),'response_sha256':hashlib.sha256(raw).hexdigest()})
        temporary=directory/'response.tmp';temporary.write_bytes(raw);temporary.replace(directory/'response.json')
        return True
    return False

def fill_with_transport_retry(directory,budget):
    # Only a transient HTTP failure gets one further attempt. Unknown charges
    # remain reserved; malformed/truncated responses and budget failures stop.
    for attempt in [1,2]:
        try:return fill(directory,budget=budget,attempt=attempt)
        except urllib.error.HTTPError as e:
            key=directory.name if attempt==1 else directory.name+':attempt'+str(attempt)
            reservation=budget.state['calls'].get(key)
            if reservation is None:raise  # A pricing lookup failed before reservation.
            reservation.update(http_error=e.code,error_unix=time.time())
            atomic_json(budget.path,budget.state)
            if attempt==2 or e.code not in {408,429,500,502,503,504,520,521,522,523,524}:raise
            time.sleep(10)

def main(root):
    plan=json.loads((root/'batch.json').read_text());jobs=plan['slurm_arrays'];budget=None
    try:
        budget=Budget(root,plan['prior_api_spend_usd']);budget.refresh_prices()
        atomic_json(root/'WORKER_READY.json',{'started_unix':time.time()})
        deadline=time.monotonic()+7*86400
        last_slurm_check=0
        while time.monotonic()<deadline:
            summary=collect(root);summary['reported_and_reserved_cost_usd']=str(budget.committed())
            summary['known_api_cost_usd']=str(Decimal(plan.get('prior_known_cost_usd',plan['prior_api_spend_usd']))+sum((Decimal(c['cost_usd']) for c in budget.state['calls'].values() if 'cost_usd' in c),Decimal(0)))
            summary['unresolved_reserved_usd']=str(Decimal(plan.get('prior_reserved_usd','0'))+sum((Decimal(c['reserved_usd']) for c in budget.state['calls'].values() if c['status']!='settled'),Decimal(0)))
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
                if reuse_cache(request.parent,plan.get('cache_sources',[])):
                    print(json.dumps({'reused_request_sha256':request.parent.name}),flush=True);continue
                obstacle=fill_with_transport_retry(request.parent,budget)
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
