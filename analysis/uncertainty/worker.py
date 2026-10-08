"""Bounded login-node network worker; never loads models or simulates."""
import argparse
from decimal import Decimal
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
from common import BASE, atomic_json, config
from budget import CampaignBudget
sys.path.insert(0, str(BASE/'scripts'))
from openrouter_perception import fill
sys.path.insert(0, str(BASE/'experiments/paired_repeats'))
from slurm_monitor import read_job_state, FAILED_STATES


class Scoped:
    def __init__(self, budget, prefix): self.budget, self.prefix = budget, prefix
    def reserve(self, key, request): self.budget.reserve(self.prefix+':'+key, request)
    def settle(self, key, response): self.budget.settle(self.prefix+':'+key, response)


def serve(root):
    plan = json.loads((root/'plan.json').read_text()); cfg = config(); budget = None
    try:
        budget = CampaignBudget(Path(plan['campaign_root']))
        # Inspect provider metadata before any worker readiness receipt or paid call.
        budget.refresh_prices()
        atomic_json(root/'cost_estimate.json', dict(planned_fresh_calls=4, max_attempts=8,
            conservative_new_reservation_usd='0.80', campaign_limit_usd='3.00',
            campaign_committed_usd=str(budget.committed()), reference_mean_call_usd='0.002363864210526316',
            expected_at_historical_mean_usd='0.009455456842105264', forecast_is_not_a_guarantee=True))
        atomic_json(root/'WORKER_READY.json', dict(campaign_limit_usd='3.00', campaign_ledger=str(budget.path)))
        deadline = time.monotonic()+7*86400; logical = set(); terminal_since = None; last_check = 0
        while time.monotonic() < deadline:
            if (root/'STOP.json').exists() or (root/'COMPLETE.json').exists(): break
            for file in sorted((root/'requests').glob('*/*/request.json')):
                if (file.parent/'response.json').exists(): continue
                name = file.parent.parent.name
                if name not in {'smoke_s%d_r%d_aegis'%(s,r) for s in [0,1] for r in [0,1]}:
                    raise RuntimeError('Unapproved VLM request run')
                key = str(file.parent)
                if key not in logical and len(logical) >= cfg['max_fresh_calls_per_run_root']:
                    raise RuntimeError('Smoke fresh-call bound reached')
                logical.add(key); prefix=root.name+':'+name
                for attempt in [1,2]:
                    try:
                        fill(file.parent, budget=Scoped(budget,prefix), attempt=attempt); break
                    except urllib.error.HTTPError as error:
                        ledger_key=prefix+':'+file.parent.name+('' if attempt==1 else ':attempt2')
                        call=budget.state['calls'].get(ledger_key)
                        if call: call.update(http_error=error.code); atomic_json(budget.path,budget.state)
                        if attempt==2 or error.code not in {408,429,500,502,503,504,520,521,522,523,524}: raise
                        time.sleep(10)
                atomic_json(root/'api_status.json', dict(campaign_committed_usd=str(budget.committed()),
                    root_logical_calls=len(logical), campaign_attempts=len(budget.state['calls']), last_run=name))
            if time.monotonic()-last_check >= 60:
                observed = read_job_state(plan['job'])
                atomic_json(root/'slurm_monitor.json', dict(observed, job=plan['job'], checked_unix=time.time()))
                if observed['state'] in FAILED_STATES: raise RuntimeError('Verified compute failure: '+observed['state'])
                if observed['state']=='COMPLETED':
                    terminal_since=terminal_since or time.monotonic()
                    if time.monotonic()-terminal_since>120 and not (root/'COMPLETE.json').exists():
                        raise RuntimeError('Compute terminal without final artifact receipt')
                else: terminal_since=None
                last_check=time.monotonic()
            time.sleep(3)
    except BaseException as error:
        atomic_json(root/'STOP.json', dict(stage='api_worker', error=type(error).__name__, reason=str(error)))
        subprocess.run(['scancel',plan['job']],check=False)
        raise
    finally:
        if budget:
            atomic_json(root/'API_FINAL.json', dict(campaign_committed_usd=str(budget.committed()),
                root_calls={k:v for k,v in budget.state['calls'].items() if k.startswith(root.name+':')},
                campaign_ledger=str(budget.path)))
            budget.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('root',type=Path); a=p.parse_args(); serve(a.root)
