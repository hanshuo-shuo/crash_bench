"""New campaign ledger: $20 ceiling, never reuse an old unlimited ledger."""
from decimal import Decimal
import fcntl
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from api_budget import Budget, MAX_PRICE, atomic_json

LIMIT = Decimal('20.00')


class DatasetBudget(Budget):
    def __init__(self, directory):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.path=self.directory/'budget.json';self.lock=(self.directory/'budget.lock').open('a')
        try:
            fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if not self.path.exists():
                atomic_json(self.path,dict(campaign='FD1000',limit_usd=str(LIMIT),initial_spent_usd='0',calls={}))
            self.state=json.loads(self.path.read_text());self.last_price_check=0
            if self.state.get('campaign')!='FD1000' or Decimal(str(self.state.get('limit_usd')))!=LIMIT:
                raise RuntimeError('Wrong campaign or changed API ceiling')
            if any(x['status']!='settled' for x in self.state['calls'].values()):
                raise RuntimeError('Unknown prior charge retained; reconciliation required before restart')
        except BaseException:
            self.lock.close();raise

    def reserve(self,key,request):
        if (request.get('model')!='z-ai/glm-4.5v' or request.get('max_tokens')!=16384 or
            request.get('provider')!={'only':['z-ai'],'allow_fallbacks':False,'require_parameters':True,'max_price':MAX_PRICE}):
            raise RuntimeError('Unapproved provider/model/token or pricing bounds')
        if key in self.state['calls']: raise RuntimeError('Duplicate paid attempt')
        if len(self.state['calls'])>=2000: raise RuntimeError('Campaign attempt ceiling reached')
        self.refresh_prices()
        if self.committed()+Decimal('.10')>LIMIT:
            raise RuntimeError('FD1000 $20 ceiling reached before any request is sent')
        self.state['calls'][key]=dict(status='reserved',reserved_usd='0.10',started_unix=time.time())
        atomic_json(self.path,self.state)

    def settle(self,key,response):
        super().settle(key,response)
        self.state['calls'][key].update(usage=response.get('usage'),model=response.get('model'),
                                     provider=response.get('provider'),generation_id=response.get('id'))
        atomic_json(self.path,self.state)
