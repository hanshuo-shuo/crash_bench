from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest
from verifier_dataset.budget import DatasetBudget, MAX_PRICE


REQUEST=dict(model='z-ai/glm-4.5v',max_tokens=16384,
    provider={'only':['z-ai'],'allow_fallbacks':False,'require_parameters':True,'max_price':MAX_PRICE})


class SpendGuard(unittest.TestCase):
    def test_twenty_dollars_includes_unresolved_reservations(self):
        with tempfile.TemporaryDirectory() as root:
            b=DatasetBudget(root);b.refresh_prices=lambda:None
            b.state['initial_spent_usd']='19.90'
            b.reserve('first',REQUEST)
            self.assertEqual(b.committed(),Decimal('20.00'))
            with self.assertRaises(RuntimeError):b.reserve('second',REQUEST)
            b.close()
            with self.assertRaises(RuntimeError):DatasetBudget(root)

    def test_settled_cost_is_durable_and_cannot_be_double_billed(self):
        with tempfile.TemporaryDirectory() as root:
            b=DatasetBudget(root);b.refresh_prices=lambda:None
            b.reserve('first',REQUEST);b.settle('first',{'usage':{'cost':'0.002'}});b.close()
            b=DatasetBudget(root);b.refresh_prices=lambda:None
            self.assertEqual(b.committed(),Decimal('.002'))
            with self.assertRaises(RuntimeError):b.reserve('first',REQUEST)
            b.close()

    def test_historical_unlimited_ledger_cannot_be_inherited(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root)/'budget.json').write_text(json.dumps(dict(limit_usd=None,calls={})))
            with self.assertRaises(RuntimeError):DatasetBudget(root)


if __name__=='__main__':unittest.main()
