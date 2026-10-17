"""Phase 1.4 API zero-shot baseline: openai/gpt-6-luna through OpenRouter, validation rows only.

The user named the model ("gpt luna 6") on 2026-10-17. Run on the login node
(compute nodes are offline) from a clean, published checkout:

  python -m verifier_training.api_baseline prepare --dataset EXPORT/dataset
  python -m verifier_training.api_baseline smoke ROOT      # 2 paid calls; inspect usage first
  python -m verifier_training.api_baseline run ROOT        # remaining calls under the ledger
  python -m verifier_training.api_baseline score ROOT

A durable ledger caps this campaign at $3.00. Each call reserves its worst-case
cost before it is sent; unknown charges keep their reservation. The key is read
from ~/.config/crashbench/openrouter.key and never printed, logged or stored.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from decimal import Decimal
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request
from .common import BASE,read,write,sha,require
from .fd1000_contract import validate,zero_shot_messages
from .fd1000_metrics import parse_output,score,row_record

MODEL='openai/gpt-6-luna'
LIMIT=Decimal('3.00')
RESERVE=Decimal('0.02')
MAX_TOKENS=8000
REASONING={'effort':'medium'}
# USD per million tokens; OpenRouter refuses endpoints above these prices.
MAX_PRICE={'prompt':0.2,'completion':1.0,'request':0,'image':0}
MODES=('direct','facts')
ENDPOINT='https://openrouter.ai/api/v1/chat/completions'
TRANSIENT={408,429,500,502,503,504,520,521,522,523,524}
IMAGE_TOKEN_ALLOWANCE=1500  # conservative per-image prompt-token bound for the reservation estimate


def payload(row,dataset,mode):
    """Exactly the zero-shot Qwen prompt, with the two PNGs inlined as data URLs."""
    system,user=zero_shot_messages(row,mode);content=[];images=iter(row['student']['images'])
    for part in user['content']:
        if part['type']=='image':
            spec=next(images);data=(Path(dataset)/spec['path']).read_bytes()
            require(hashlib.sha256(data).hexdigest()==spec['sha256'],'Image changed')
            content.append(dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(data).decode())))
        else:content.append(dict(type='text',text=part['text']))
    return dict(model=MODEL,messages=[dict(role='system',content=system['content']),dict(role='user',content=content)],
        max_tokens=MAX_TOKENS,reasoning=REASONING,seed=1014,usage=dict(include=True),
        provider=dict(only=['openai'],allow_fallbacks=False,require_parameters=True,max_price=MAX_PRICE))


def prompt_estimate(request):
    text=sum(len(p['text']) for m in request['messages'] for p in (m['content'] if isinstance(m['content'],list) else [dict(text=m['content'])]) if 'text' in p)
    images=sum(p.get('type')=='image_url' for m in request['messages'] if isinstance(m['content'],list) for p in m['content'])
    return text//3+IMAGE_TOKEN_ALLOWANCE*images


def worst_case(prompt_tokens):
    return Decimal(str(MAX_PRICE['prompt']))*prompt_tokens/10**6+Decimal(str(MAX_PRICE['completion']))*MAX_TOKENS/10**6


def check_prices():
    with urllib.request.urlopen('https://openrouter.ai/api/v1/models/openai/gpt-6-luna/endpoints',timeout=30) as r:data=json.load(r)
    endpoints=[e for e in data['data']['endpoints'] if e['provider_name']=='OpenAI']
    require(endpoints,'OpenAI endpoint for gpt-6-luna unavailable')
    for e in endpoints:
        p=e['pricing']
        for name,ceiling in (('prompt',MAX_PRICE['prompt']),('completion',MAX_PRICE['completion'])):
            require(Decimal(str(p[name]))*10**6<=Decimal(str(ceiling)),'Endpoint price above the frozen ceiling')
        require(all(Decimal(str(p.get(k,'0')))==0 for k in ('request','image')),'Unexpected per-request or per-image price')
    return endpoints


class Ledger:
    def __init__(self,root):
        self.path=root/'budget.json';self.lock_file=(root/'budget.lock').open('a');self.lock=threading.Lock()
        try:
            fcntl.flock(self.lock_file,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if not self.path.exists():self.state=dict(campaign='V4-dev-api-gpt-6-luna',limit_usd=str(LIMIT),calls={});self.save()
            self.state=read(self.path)
            require(Decimal(self.state['limit_usd'])==LIMIT,'Ledger ceiling changed')
            require(all(c['status'] in ('settled','unknown_charge_reserved') for c in self.state['calls'].values()),'Unreconciled reservation')
        except BaseException:
            self.lock_file.close();raise

    def save(self):
        tmp=self.path.with_suffix('.tmp');tmp.write_text(json.dumps(self.state,indent=2)+'\n');tmp.replace(self.path)

    def committed(self):
        return sum((Decimal(c.get('cost_usd',c['reserved_usd'])) for c in self.state['calls'].values()),Decimal('0'))

    def reserve(self,key,amount):
        with self.lock:
            require(key not in self.state['calls'],'Duplicate paid attempt')
            require(self.committed()+amount<=LIMIT,'API ceiling of $3.00 reached; no request sent')
            self.state['calls'][key]=dict(status='reserved',reserved_usd=str(amount),started_unix=time.time());self.save()

    def settle(self,key,response):
        with self.lock:
            cost=(response.get('usage') or {}).get('cost');record=self.state['calls'][key]
            if cost is None:
                record['status']='unknown_charge_reserved';self.save();raise RuntimeError('Unknown API charge; reservation kept')
            cost=Decimal(str(cost));require(cost.is_finite() and 0<=cost<=Decimal(record['reserved_usd']),'Charge outside its reservation')
            record.update(status='settled',cost_usd=str(cost),finished_unix=time.time(),usage=response.get('usage'),
                model=response.get('model'),provider=response.get('provider'),generation_id=response.get('id'));self.save()

    def unknown(self,key):
        with self.lock:self.state['calls'][key]['status']='unknown_charge_reserved';self.save()

    def close(self):self.lock_file.close()


def credential():
    path=Path.home()/'.config/crashbench/openrouter.key'
    require(path.is_file(),'Configure the key with scripts/configure_openrouter_key.py first')
    require(not path.stat().st_mode&0o077,'Credential file must have mode 600')
    return path.read_text().strip()


def call(root,ledger,key,request,secret):
    """One logical request with at most one retry for transient HTTP failures."""
    folder=root/'responses';target=folder/(key+'.json')
    if target.exists():return read(target)
    body=json.dumps(request).encode();estimate=prompt_estimate(request)
    for attempt in (1,2):
        name=key+('' if attempt==1 else ':attempt2');ledger.reserve(name,max(RESERVE,worst_case(estimate)))
        req=urllib.request.Request(ENDPOINT,data=body,method='POST',headers={'Authorization':'Bearer '+secret,'Content-Type':'application/json'})
        started=time.monotonic()
        try:
            with urllib.request.urlopen(req,timeout=600) as r:result=json.load(r)
        except urllib.error.HTTPError as error:
            ledger.unknown(name)
            if attempt==1 and error.code in TRANSIENT:time.sleep(15);continue
            raise RuntimeError(f'OpenRouter HTTP {error.code}; reservation kept') from None
        except BaseException:
            ledger.unknown(name);raise
        ledger.settle(name,result)
        if 'error' in result:raise RuntimeError('OpenRouter returned an API error')
        record=dict(key=key,attempt=attempt,created_utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-started,response=result)
        tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=1)+'\n');tmp.replace(target);return record
    raise RuntimeError('Transient failure persisted')


def prepare(dataset):
    from .launch_fd1000 import published_commit,stamp
    commit=published_commit();dataset=Path(os.path.abspath(dataset));dataset.resolve().relative_to((BASE/'verifier_dataset').resolve())
    rows,acceptance=validate(dataset,mode='development',include_test=False)
    validation=[r for r in rows if r['split']=='validation']
    root=BASE/'verifier_training/runs'/(stamp()+'_api-luna_'+commit[:12]);root.mkdir();(root/'responses').mkdir()
    plan=[];estimate=Decimal('0')
    for row in validation:
        for mode in MODES:
            request=payload(row,dataset,mode);key=f'{mode}_{row["sample_id"]}';tokens=prompt_estimate(request)
            estimate+=worst_case(tokens);plan.append(dict(key=key,sample_id=row['sample_id'],mode=mode,
                request_sha256=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()))
    endpoints=check_prices()
    write(root/'RECEIPT.json',dict(stage='api-baseline',model=MODEL,model_choice='User said "gpt luna 6" on 2026-10-17; resolved to openai/gpt-6-luna on OpenRouter',
        source_commit=commit,dataset=str(dataset),dataset_acceptance_sha256=sha(dataset/'DEV_ACCEPTANCE.json'),split='validation',
        rows=len(validation),modes=list(MODES),logical_requests=len(plan),limit_usd=str(LIMIT),max_tokens=MAX_TOKENS,reasoning=REASONING,
        max_price_per_million=MAX_PRICE,worst_case_total_usd=str(estimate.quantize(Decimal('0.0001'))),
        authorization='2026-10-17 user message chose the Phase 1 API baseline model; PLAN.md caps this baseline at $3',
        egress='Two 448x448 simulator renders plus task/constraint text per request; no keys, paths or metadata',
        provider_endpoints=endpoints,logprobs='Unsupported by this model; verdicts are discrete, so no AUROC'))
    write(root/'PLAN.json',plan);print(json.dumps(dict(root=str(root),requests=len(plan),worst_case_usd=str(estimate))),flush=True)


def requests_for(root):
    receipt=read(root/'RECEIPT.json');dataset=Path(receipt['dataset'])
    require(sha(dataset/'DEV_ACCEPTANCE.json')==receipt['dataset_acceptance_sha256'],'Dataset changed')
    rows,_=validate(dataset,mode='development',include_test=False);by_id={r['sample_id']:r for r in rows}
    for entry in read(root/'PLAN.json'):
        request=payload(by_id[entry['sample_id']],dataset,entry['mode'])
        require(hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()==entry['request_sha256'],'Request changed since prepare')
        yield entry,request


def send(root,limit,workers):
    check_prices();ledger=Ledger(root);secret=credential();done=0
    try:
        pending=[(e,r) for e,r in requests_for(root) if not (root/'responses'/(e['key']+'.json')).exists()]
        if limit is not None:pending=pending[:limit]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for record in pool.map(lambda x:call(root,ledger,x[0]['key'],x[1],secret),pending):
                done+=1
                if done%20==0:print(json.dumps(dict(completed=done,committed_usd=str(ledger.committed()))),flush=True)
    except BaseException as error:
        write(root/'STOP.json',dict(error=type(error).__name__,reason=str(error),completed_this_run=done));raise
    finally:
        write(root/'API_STATUS.json',dict(committed_usd=str(ledger.committed()),calls=len(ledger.state['calls']),limit_usd=str(LIMIT)))
        ledger.close()
    print(json.dumps(dict(completed=done,committed_usd=str(read(root/'API_STATUS.json')['committed_usd']))),flush=True)


def usage_report(root):
    state=read(root/'budget.json');settled=[c for c in state['calls'].values() if c['status']=='settled']
    def mean(path):
        values=[]
        for c in settled:
            v=c['usage']
            for k in path:v=(v or {}).get(k)
            if v is not None:values.append(v)
        return sum(values)/len(values) if values else None
    return dict(settled=len(settled),cost_usd=str(sum((Decimal(c['cost_usd']) for c in settled),Decimal('0'))),
        mean_prompt_tokens=mean(('prompt_tokens',)),mean_completion_tokens=mean(('completion_tokens',)),
        mean_reasoning_tokens=mean(('completion_tokens_details','reasoning_tokens')),
        mean_cost_usd=float(sum(Decimal(c['cost_usd']) for c in settled)/len(settled)) if settled else None,
        models=sorted({str(c.get('model')) for c in settled}),providers=sorted({str(c.get('provider')) for c in settled}))


def score_root(root):
    receipt=read(root/'RECEIPT.json');rows={r['sample_id']:r for r in validate(Path(receipt['dataset']),mode='development',include_test=False)[0]}
    records=[]
    for entry in read(root/'PLAN.json'):
        row=rows[entry['sample_id']];path=root/'responses'/(entry['key']+'.json');kind=entry['mode']
        record=row_record(row,'api_'+entry['mode'],kind,None,None)
        if path.exists():
            response=read(path)['response'];choice=(response.get('choices') or [{}])[0]
            text=(choice.get('message') or {}).get('content') or ''
            record.update(parse_output(text,kind),generated=text,finish_reason=choice.get('finish_reason'),model=response.get('model'))
        else:record.update(predicted=None,format_valid=False,facts=None,generated=None,finish_reason='not_sent')
        record.update(p_feasible=None,gold_rationale_prefix_used=False);records.append(record)
    write(root/'RECORDS.json',records)
    summary={}
    for mode in MODES:
        part=[r for r in records if r['reading']=='api_'+mode];summary['api_'+mode]=score(part,mode,bootstrap=True)
        summary['api_'+mode]['finish_reasons']={str(k):sum(r['finish_reason']==k for r in part) for k in {r['finish_reason'] for r in part}}
    summary['usage']=usage_report(root);summary['scope']='Validation only; the model returns no logprobs, so verdicts are discrete and the reported AUROC (all scores 0.5) is not meaningful'
    write(root/'SUMMARY.json',summary);print(json.dumps({k:v.get('verdict_accuracy') for k,v in summary.items() if isinstance(v,dict) and 'verdict_accuracy' in v}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','smoke','run','score','usage'))
    p.add_argument('root',nargs='?',type=Path);p.add_argument('--dataset',type=Path);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.dataset)
    elif a.action=='smoke':send(a.root,limit=2,workers=1);print(json.dumps(usage_report(a.root),indent=1))
    elif a.action=='run':send(a.root,limit=None,workers=a.workers)
    elif a.action=='usage':print(json.dumps(usage_report(a.root),indent=1))
    else:score_root(a.root)
