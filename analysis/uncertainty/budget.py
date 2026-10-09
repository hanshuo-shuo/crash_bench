"""Campaign-wide $3 durable ledger, including unknown charges and retries."""
from decimal import Decimal
import fcntl
import json
from pathlib import Path
import sys
import time
from common import BASE, atomic_json, config

sys.path.insert(0, str(BASE/'scripts'))
from api_budget import Budget, MAX_PRICE


class CampaignBudget(Budget):
    def __init__(self, campaign, cfg=None):
        self.directory = Path(campaign); self.path = self.directory/'budget.json'
        self.lock = (self.directory/'budget.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            limit = (cfg or config())['api_limit_usd']
            self.limit = None if limit is None else Decimal(limit); self.reserve_usd = Decimal('0.10')
            if not self.path.exists(): atomic_json(self.path, dict(limit_usd='3.00', initial_spent_usd='0', calls={}))
            self.state = json.loads(self.path.read_text()); self.last_price_check = 0
            if self.limit is None and self.state['limit_usd'] is not None:
                atomic_json(self.directory/'API_AUTHORIZATION_CHANGE_20261009.json',dict(
                    previous_ledger=self.state,previous_limit_usd=self.state['limit_usd'],new_limit_usd=None,
                    authority='User 2026-10-09 00:23 UTC: 之后都确认吧 暂时不限制api 帮我把这一系列跑完'))
                self.state['limit_usd']=None;atomic_json(self.path,self.state)
            elif self.state['limit_usd'] is not None and Decimal(self.state['limit_usd']) != self.limit:
                raise RuntimeError('Campaign ceiling changed')
            elif self.state['limit_usd'] is None and self.limit is not None:
                raise RuntimeError('Cannot silently restore a prior API cap')
            if any(x['status']!='settled' for x in self.state['calls'].values()):
                raise RuntimeError('Prior unknown charge retained; do not restart paid worker without reconciliation')
        except BaseException:
            self.lock.close()
            raise

    def reserve(self, key, request):
        if request.get('model') != 'z-ai/glm-4.5v' or request.get('max_tokens') != 16384 or request.get('provider') != {'only':['z-ai'],'allow_fallbacks':False,'require_parameters':True,'max_price':MAX_PRICE}:
            raise RuntimeError('Pinned model/provider/billing bounds missing')
        if key in self.state['calls']: raise RuntimeError('Duplicate paid attempt')
        self.refresh_prices()
        if self.limit is not None and self.committed()+self.reserve_usd > self.limit:
            raise RuntimeError('Authorized $3 boundary reached before sending request')
        self.state['calls'][key] = dict(status='reserved', reserved_usd='0.10', started_unix=time.time())
        atomic_json(self.path, self.state)

    def settle(self, key, response):
        super().settle(key, response)
        # Full usage includes prompt/completion/reasoning/cache token metadata when returned.
        self.state['calls'][key].update(usage=response.get('usage'), model=response.get('model'),
            provider=response.get('provider'), generation_id=response.get('id'))
        atomic_json(self.path, self.state)
