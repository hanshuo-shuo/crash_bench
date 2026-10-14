"""Only the approved fresh Goal requests, with a shared $20 ledger."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
from .budget import DatasetBudget
from .inventory import read, write

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from openrouter_perception import fill


class ScopedBudget:
    def __init__(self,budget,prefix):self.budget,self.prefix=budget,prefix
    def reserve(self,key,request):self.budget.reserve(self.prefix+':'+key,request)
    def settle(self,key,response):self.budget.settle(self.prefix+':'+key,response)


def main(root):
    plan=read(root/'RECEIPT.json');budget=None
    if plan.get('api_mode')=='cache_only':raise RuntimeError('Paid worker forbidden for cache-only collection')
    expected=len(read(root/'COLLECTION_SOURCES.json'))
    allowed={'fd_goal%02d_aegis'%i for i in range(expected)}
    try:
        budget=DatasetBudget(Path(plan['budget_root']))
        budget.refresh_prices()
        write(root/'COST_ESTIMATE.json',dict(planned_logical_requests=expected,max_attempts=2*expected,
            maximum_reserved_for_batch_usd=round(.2*expected,2),historical_perception_mean_usd=.002182933125,
            historical_mean_estimate_usd=.002182933125*expected,forecast_is_not_guarantee=True,
            campaign_limit_usd='20.00',campaign_already_committed_usd=str(budget.committed()),
            exact_cache_hits_cost_zero=True))
        write(root/'WORKER_READY.json',dict(limit_usd='20.00',ledger=str(budget.path),pricing_checked=True))
        last_check=0;logical=set();deadline=time.monotonic()+3*3600
        while time.monotonic()<deadline:
            if (root/'STOP.json').exists():break
            for path in sorted((root/'requests').glob('*/*/request.json')):
                if (path.parent/'response.json').exists():continue
                if not (path.parent/'NEEDS_API.json').exists():continue
                name=path.parent.parent.name
                if name not in allowed:raise RuntimeError('Unapproved request source')
                key=str(path.parent)
                if key not in logical and len(logical)>=expected:raise RuntimeError('Logical request ceiling reached')
                logical.add(key)
                for attempt in (1,2):
                    try:
                        fill(path.parent,budget=ScopedBudget(budget,root.name+':'+name),attempt=attempt);break
                    except urllib.error.HTTPError as error:
                        if attempt==2 or error.code not in {408,429,500,502,503,504,520,521,522,523,524}:raise
                        time.sleep(10)
                write(root/'API_STATUS.json',dict(committed_usd=str(budget.committed()),logical_requests=len(logical)))
            if time.monotonic()-last_check>=30:
                status=subprocess.check_output(['sacct','-X','-n','-P','-j',plan['job'],
                    '--format=JobIDRaw,State,ExitCode'],text=True).strip()
                write(root/'ACCOUNTING_LATEST.json',dict(raw=status,checked_unix=time.time()))
                if '|COMPLETED|0:0' in status:
                    if not (root/'COLLECTION_COMPLETE.json').exists():raise RuntimeError('Normal exit missing collection receipt')
                    break
                if any('|'+s in status for s in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL')):
                    raise RuntimeError('Collection job failed')
                last_check=time.monotonic()
            time.sleep(3)
        else:raise RuntimeError('Bounded API worker deadline reached')
    except BaseException as error:
        # No credential or request headers enter diagnostics.
        write(root/'STOP.json',dict(stage='api_worker',error=type(error).__name__,reason=str(error)))
        subprocess.run(['scancel',plan['job']],check=False)
        raise
    finally:
        if budget:
            write(root/'API_FINAL.json',dict(committed_usd=str(budget.committed()),
                calls={k:v for k,v in budget.state['calls'].items() if k.startswith(root.name+':')},limit_usd='20.00'))
            budget.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
