"""Single-writer, durable accounting for this reproduction's OpenRouter calls."""
from decimal import Decimal
import fcntl
import json
from pathlib import Path
import time
import urllib.request

RESERVE=Decimal('0.10')
LIMIT=Decimal('5.00')
MAX_PRICE={'prompt':0.6,'completion':1.8,'request':0,'image':0}

def atomic_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as f:
        json.dump(value,f,indent=2);f.write('\n');f.flush()
        import os
        os.fsync(f.fileno())
    temporary.replace(path)

def validate_endpoint(data):
    endpoints=[e for e in data['data']['endpoints'] if e['provider_name']=='Z.AI']
    if not endpoints:raise RuntimeError('Pinned Z.AI endpoint unavailable')
    for e in endpoints:
        prices=e['pricing']
        if not {'prompt','completion'}<=set(prices):raise RuntimeError('Incomplete pricing data')
        if not 0<int(e['context_length'])<=65536 or not 0<int(e['max_completion_tokens'])<=16384:
            raise RuntimeError('Provider token bounds changed; recompute reservation before spending')
        for name,ceiling in [('prompt','0.0000006'),('completion','0.0000018'),('request','0'),('image','0')]:
            price=Decimal(str(prices.get(name,'0')))
            if not price.is_finite() or price<0 or price>Decimal(ceiling):raise RuntimeError('Provider price exceeds frozen ceiling')
    # Even independently filling both token bounds costs <= $0.0688128.
    # $0.10 is reserved BEFORE the request; unknown/failed charges keep that hold.
    return endpoints

class Budget:
    def __init__(self,directory,initial_spent='0'):
        self.directory=Path(directory);self.path=self.directory/'budget.json'
        self.lock=(self.directory/'budget.lock').open('a')
        try:fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:
            self.lock.close();raise
        if not self.path.exists():
            atomic_json(self.path,{'limit_usd':str(LIMIT),'initial_spent_usd':str(initial_spent),'calls':{}})
        self.state=json.loads(self.path.read_text())
        if Decimal(self.state['limit_usd'])!=LIMIT:
            self.lock.close();raise RuntimeError('Budget limit changed')
        if any(r['status']!='settled' for r in self.state['calls'].values()):
            self.lock.close()
            raise RuntimeError('Unresolved prior charge; no automatic retries')
        self.last_price_check=0

    def refresh_prices(self):
        if time.monotonic()-self.last_price_check<600:return
        with urllib.request.urlopen('https://openrouter.ai/api/v1/models/z-ai/glm-4.5v/endpoints',timeout=30) as r:data=json.load(r)
        endpoints=validate_endpoint(data)
        atomic_json(self.directory/'provider_price_check.json',{'checked_unix':time.time(),'endpoints':endpoints})
        self.last_price_check=time.monotonic()

    def committed(self):
        return Decimal(self.state['initial_spent_usd'])+sum((Decimal(r.get('cost_usd',r['reserved_usd'])) for r in self.state['calls'].values()),Decimal('0'))

    def reserve(self,key,request):
        if request.get('model')!='z-ai/glm-4.5v' or request.get('max_tokens')!=16384 or request.get('provider')!={'only':['z-ai'],'allow_fallbacks':False,'require_parameters':True,'max_price':MAX_PRICE}:
            raise RuntimeError('Request lacks the frozen model and billing bounds')
        if key in self.state['calls']:raise RuntimeError('Request was already charged or reserved; refusing retry')
        if len(self.state['calls'])>=1600:raise RuntimeError('1600-call batch ceiling reached')
        self.refresh_prices()
        if self.committed()+RESERVE>LIMIT:raise RuntimeError('OpenRouter $5 budget boundary reached; no request sent')
        self.state['calls'][key]={'status':'reserved','reserved_usd':str(RESERVE),'started_unix':time.time()}
        atomic_json(self.path,self.state)

    def settle(self,key,response):
        value=response.get('usage',{}).get('cost')
        if value is None:raise RuntimeError('Unknown API charge; reservation retained, stopping batch')
        cost=Decimal(str(value))
        if not cost.is_finite() or cost<0 or cost>RESERVE:
            raise RuntimeError('API charge outside reserved bound; stopping for reconciliation')
        self.state['calls'][key].update(status='settled',cost_usd=str(cost),finished_unix=time.time())
        atomic_json(self.path,self.state)

    def close(self):self.lock.close()
