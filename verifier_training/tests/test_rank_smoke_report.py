import unittest
from verifier_training.tests import test_engineering as fixtures
from verifier_training.engineering_contract import score_rationale
from verifier_training.rank_smoke_report import audit


class SmokeReadback(unittest.TestCase):
    def example(self, digit='0', zero=False):
        row=fixtures.Engineering().row();row['sample_id']='test'
        text='Rationale: '+row['target']['rationale']+'\nVerdict: '+digit
        x={'sample_id':'test','generated':text,'predicted_verdict':{'0':'infeasible','1':'feasible'}[digit],
           'format_valid':True,'verdict_correct':digit=='0','exact_match':digit=='0',
           'rationale_assessment':score_rationale(text,row,'horizon_0' if zero else 'original')}
        return row,x

    def test_correct_rationale_does_not_fix_wrong_verdict(self):
        row,x=self.example('1');r=audit([row],[x])[0]
        self.assertTrue(r['rationale_correct']);self.assertFalse(r['verdict_correct']);self.assertFalse(r['full_output_correct'])

    def test_stale_stored_success_rejected(self):
        row,x=self.example('1');x['verdict_correct']=True
        with self.assertRaises(ValueError):audit([row],[x])

    def test_zero_label_is_recomputed(self):
        row,x=self.example('0',True);row['target']['verdict']='feasible'
        self.assertTrue(audit([row],[x],True)[0]['verdict_correct'])
        with self.assertRaises(ValueError):audit([row],[x],False)
