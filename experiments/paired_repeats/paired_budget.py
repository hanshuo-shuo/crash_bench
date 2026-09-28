"""Experiment-specific, explicitly authorized ceiling; baseline guard unchanged."""
from decimal import Decimal
import fcntl
import json
from pathlib import Path
import time
from api_budget import Budget, RESERVE, MAX_PRICE, atomic_json

class PairedBudget(Budget):
    def __init__(self, directory, initial_spent='0', limit='5.00'):
        self.limit = Decimal(str(limit))
        if not self.limit.is_finite() or not 0 < self.limit <= Decimal('65.00'):
            raise RuntimeError('Paired ceiling must be positive and at most the authorized $65')
        self.directory = Path(directory); self.path = self.directory/'budget.json'
        self.lock = (self.directory/'budget.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            if not self.path.exists():
                atomic_json(self.path, {'limit_usd':str(self.limit),'initial_spent_usd':str(initial_spent),'calls':{}})
            self.state = json.loads(self.path.read_text())
            if Decimal(self.state['limit_usd']) != self.limit:
                raise RuntimeError('Plan and durable budget ceiling disagree')
            if any(r['status'] != 'settled' for r in self.state['calls'].values()):
                raise RuntimeError('Unresolved prior charge; no automatic restart')
        except BaseException:
            self.lock.close(); raise
        self.last_price_check = 0

    def reserve(self, key, request):
        if request.get('model') != 'z-ai/glm-4.5v' or request.get('max_tokens') != 16384 or request.get('provider') != {
            'only':['z-ai'],'allow_fallbacks':False,'require_parameters':True,'max_price':MAX_PRICE}:
            raise RuntimeError('Request lacks frozen model and price bounds')
        if key in self.state['calls']:
            raise RuntimeError('Attempt already charged or reserved')
        if len(self.state['calls']) >= 604:
            raise RuntimeError('302 logical calls times two attempts ceiling reached')
        self.refresh_prices()
        if self.committed()+RESERVE > self.limit:
            raise RuntimeError('Paired API budget boundary reached; no request sent')
        self.state['calls'][key] = {'status':'reserved','reserved_usd':str(RESERVE),'started_unix':time.time()}
        atomic_json(self.path,self.state)
