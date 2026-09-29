"""Fresh VLM calls with conserved charges and an explicit experiment ceiling."""
import argparse
from decimal import Decimal
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from api_budget import atomic_json
from paired_budget import PairedBudget
from slurm_monitor import read_job_state, FAILED_STATES

def prior_spend(assets, exclude):
    known = Decimal('0'); reserved = Decimal('0')
    evidence = []
    for path in (assets/'perception_cache').glob('*/response.json'):
        value = json.loads(path.read_text())['response']['usage']['cost']
        known += Decimal(str(value)); evidence.append(str(path))
    if list((assets/'perception_cache').glob('*/failed_response.json')):
        raise RuntimeError('Unreconciled initial smoke response')
    # Initial balances already carry history: sum calls only, never initial_spent.
    for path in list((assets/'batches').glob('*/budget.json')) + list((assets/'paired_repeats').glob('*/budget.json')):
        if path.parent.resolve() == exclude.resolve():
            continue
        data = json.loads(path.read_text()); evidence.append(str(path))
        for call in data['calls'].values():
            if 'cost_usd' in call:
                known += Decimal(call['cost_usd'])
            else:
                reserved += Decimal(call['reserved_usd'])
    return known, reserved, evidence

class ScopedBudget:
    """Keep the original request hash, prefix ledger keys with unique run identity."""
    def __init__(self, budget, scope):
        self.budget, self.scope = budget, scope
        self.path, self.state = budget.path, budget.state
    def reserve(self, key, request):
        self.budget.reserve(self.scope+':'+key, request)
    def settle(self, key, response):
        self.budget.settle(self.scope+':'+key, response)

def fresh_fill(directory, budget, scope):
    # Retry implementation needs matching scoped keys for both reserve and audit.
    import urllib.error
    from openrouter_perception import fill
    for attempt in [1,2]:
        try:
            return fill(directory, budget=ScopedBudget(budget, scope), attempt=attempt)
        except urllib.error.HTTPError as error:
            key = scope+':'+directory.name+('' if attempt == 1 else ':attempt2')
            call = budget.state['calls'].get(key)
            if call is None:
                raise
            call.update(http_error=error.code, error_unix=time.time()); atomic_json(budget.path,budget.state)
            if attempt == 2 or error.code not in {408,429,500,502,503,504,520,521,522,523,524}:
                raise
            time.sleep(10)

def active_jobs(ids):
    if not ids:
        return False
    listing = subprocess.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%i'], text=True)
    active = {line.strip().split('_')[0] for line in listing.splitlines()}
    return bool(active.intersection(ids))

def main(root, stage):
    plan = json.loads((root/'plan.json').read_text())
    job = plan['jobs'][stage]; budget = None
    try:
        # Never snapshot a changing baseline ledger or spend concurrently with it.
        deadline = time.monotonic()+7*86400
        while active_jobs(plan['baseline_jobs']):
            if time.monotonic() > deadline:
                raise RuntimeError('Baseline wait deadline')
            time.sleep(30)
        assets = Path(plan['assets'])
        known,reserved,evidence = prior_spend(assets,root)
        if stage == 'checks':
            if (root/'budget.json').exists():
                raise RuntimeError('Checks budget already exists')
            atomic_json(root/'cost_estimate.json', {'prior_known_usd':str(known), 'prior_reserved_usd':str(reserved),
                'prior_evidence':evidence, 'planned_fresh_vlm_calls':302, 'forecast_at_first_smoke_cost_usd':str(Decimal('0.0018084')*302),
                'reference_uncached_call_cost_usd':'0.0018084',
                'limit_including_all_reproduction_usd':'5.00', 'reservation_per_attempt_usd':'0.10',
                'forecast_is_not_a_guarantee':True})
        else:
            previous = json.loads((root/'budget.json').read_text())
            if Decimal(previous['initial_spent_usd']) != known+reserved:
                raise RuntimeError('Other reproduction spend changed after checks')
        budget = PairedBudget(root, str(known+reserved), plan.get('api_limit_usd','5.00'))
        atomic_json(root/('WORKER_READY_'+stage+'.json'), {'unix':time.time(),'limit_usd':str(budget.limit)})
        last_slurm_check=0
        completed_without_receipt_since=None
        # Prices and key are needed only if the first two self-checks pass.
        while time.monotonic() < deadline:
            if (root/'STOP.json').exists():
                return
            if (root/('CHECKS_AWAIT_REVIEW.json' if stage == 'checks' else 'COMPLETE.json')).exists():
                return
            for request in sorted((root/stage).glob('requests/*/*/request.json')):
                if (request.parent/'response.json').exists():
                    continue
                expected_max = plan.get('max_fresh_vlm_calls',2 if stage == 'checks' else 302)
                if not isinstance(expected_max,int) or not 0 < expected_max <= 302:
                    raise RuntimeError('Invalid planned VLM call ceiling')
                logical = {key.rsplit(':',1)[0] if ':attempt' not in key else key.rsplit(':',2)[0] for key in budget.state['calls']}
                if len(logical) >= expected_max:
                    raise RuntimeError('Paired per-stage request bound reached')
                name = request.parent.parent.name
                fresh_fill(request.parent,budget,stage+'/'+name)
                atomic_json(root/'api_status.json',{'committed_usd':str(budget.committed()),'attempts':len(budget.state['calls']),'last_run':name})
            if time.monotonic()-last_slurm_check >= 60:
                observed=read_job_state(job)
                atomic_json(root/'slurm_monitor.json',dict(observed,job=job,checked_unix=time.time()))
                if observed['state'] in FAILED_STATES:
                    raise RuntimeError('Verified paired GPU job failure: '+observed['state'])
                if observed['state']=='COMPLETED':
                    if completed_without_receipt_since is None:
                        completed_without_receipt_since=time.monotonic()
                    elif time.monotonic()-completed_without_receipt_since>120:
                        raise RuntimeError('Compute completed but no result-completion receipt appeared')
                else:
                    completed_without_receipt_since=None
                last_slurm_check=time.monotonic()
            time.sleep(5)
        raise RuntimeError('Seven-day experiment deadline')
    except BlockingIOError:
        raise
    except BaseException as error:
        atomic_json(root/'STOP.json',{'stage':'api_'+stage,'reason':type(error).__name__+': '+str(error),'unix':time.time()})
        # Never cancel, signal, or write to baseline jobs/results.
        subprocess.run(['scancel',job],check=False)
        raise
    finally:
        if budget:
            budget.close()

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('stage',choices=['checks','full']);a=p.parse_args()
    main(a.root.resolve(),a.stage)
