"""Single bounded login-node VLM worker for exactly two recorded full shards."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.error
from common import atomic_json, full_schedule
from budget import CampaignBudget
from worker import Scoped, fill, read_job_state, FAILED_STATES


def serve(root):
    plan=json.loads((root/'plan.json').read_text());cfg=plan['configuration'];budget=None
    allowed={x['name'] for x in full_schedule(cfg) if x['method']=='aegis'}
    jobs=plan['jobs'];logical=set();last_check=0;second_released=False
    try:
        budget=CampaignBudget(Path(plan['campaign_root']),cfg)
        budget.refresh_prices()
        atomic_json(root/'cost_estimate.json',dict(planned_logical_calls=600,max_attempts=1200,
            expected_at_smoke_mean_usd='1.35',dollar_cap=None,forecast_is_not_a_guarantee=True,
            per_attempt_reservation_usd='0.10',prior_committed_usd=str(budget.committed())))
        atomic_json(root/'WORKER_READY.json',dict(dollar_cap=None,ledger=str(budget.path),jobs=jobs))
        deadline=time.monotonic()+7*86400
        while time.monotonic()<deadline:
            stop=[p for p in (root/'shards').glob('*/STOP.json')]
            if stop: raise RuntimeError('Shard failure: '+stop[0].read_text())
            if (root/'STOP.json').exists():
                for job in jobs:subprocess.run(['scancel',job],check=False)
                break
            smoke=root/'shards/0/FULL_SMOKE_PASS.json'
            if not second_released and smoke.exists():
                gate=json.loads(smoke.read_text())
                if not gate['passed'] or gate['runs']!=8:raise RuntimeError('Full-stage smoke failed')
                subprocess.run(['scontrol','release',jobs[1]],check=True)
                second_released=True
                atomic_json(root/'SECOND_SHARD_RELEASED.json',dict(job=jobs[1],smoke=gate,unix=time.time()))
            for file in sorted((root/'shards').glob('*/requests/*/*/request.json')):
                if (file.parent/'response.json').exists():continue
                name=file.parent.parent.name
                if name not in allowed:raise RuntimeError('Request outside frozen full schedule')
                key=str(file.parent)
                if key not in logical and len(logical)>=600:raise RuntimeError('Six-hundred logical-call bound exceeded')
                logical.add(key);prefix=root.name+':'+name
                for attempt in [1,2]:
                    try:fill(file.parent,budget=Scoped(budget,prefix),attempt=attempt);break
                    except urllib.error.HTTPError as error:
                        ledger_key=prefix+':'+file.parent.name+('' if attempt==1 else ':attempt2')
                        call=budget.state['calls'].get(ledger_key)
                        if call:call.update(http_error=error.code);atomic_json(budget.path,budget.state)
                        if attempt==2 or error.code not in {408,429,500,502,503,504,520,521,522,523,524}:raise
                        time.sleep(10)
                atomic_json(root/'api_status.json',dict(campaign_committed_usd=str(budget.committed()),
                    root_logical_calls=len(logical),campaign_attempts=len(budget.state['calls']),last_run=name))
            if time.monotonic()-last_check>=60:
                observed={job:read_job_state(job) for job in jobs}
                atomic_json(root/'slurm_monitor.json',dict(jobs=observed,checked_unix=time.time()))
                if any(x['state'] in FAILED_STATES for x in observed.values()):
                    raise RuntimeError('Recorded compute job failed: '+json.dumps(observed))
                if any((root/'shards'/str(i)/'PARTIAL_EXPORTED.json').exists() for i in [0,1]):
                    atomic_json(root/'PARTIAL.json',dict(reason='Allocation exhausted after completed-case checkpoint',jobs=observed))
                    for i,job in enumerate(jobs):
                        if not (root/'shards'/str(i)/'COMPLETE.json').exists():
                            subprocess.run(['scancel',job],check=False)
                    break
                if all((root/'shards'/str(i)/'COMPLETE.json').exists() for i in [0,1]):
                    atomic_json(root/'COLLECTION_COMPLETE.json',dict(runs=1200,jobs=observed,
                        shard_receipts=[json.loads((root/'shards'/str(i)/'COMPLETE.json').read_text()) for i in [0,1]]))
                    break
                if all(x['state']=='COMPLETED' for x in observed.values()):
                    raise RuntimeError('Terminal compute without complete artifact receipts')
                last_check=time.monotonic()
            time.sleep(3)
        else:raise RuntimeError('Bounded worker lifetime exhausted')
    except BaseException as error:
        atomic_json(root/'STOP.json',dict(stage='api_worker',error=type(error).__name__,reason=str(error)))
        for job in jobs:subprocess.run(['scancel',job],check=False)
        raise
    finally:
        if budget:
            atomic_json(root/'API_FINAL.json',dict(campaign_committed_usd=str(budget.committed()),
                root_calls={k:v for k,v in budget.state['calls'].items() if k.startswith(root.name+':')},
                campaign_ledger=str(budget.path)))
            budget.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();serve(a.root)
