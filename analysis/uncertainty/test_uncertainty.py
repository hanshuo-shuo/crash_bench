import ast
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
import common
from budget import CampaignBudget
from adapter import adapt


class MetricTests(unittest.TestCase):
    def test_frozen_schedule_pairing_and_no_full_stage(self):
        plans=list(common.schedule(common.config()))
        self.assertEqual(len(plans),9)
        self.assertFalse(plans[0]['diagnostics'])
        self.assertEqual(sum(x['method']=='aegis' for x in plans),4)
        for i in range(1,9,2):
            self.assertEqual(common.seed_for(plans[i]['scene'],plans[i]['repeat']),
                             common.seed_for(plans[i+1]['scene'],plans[i+1]['repeat']))
        self.assertEqual(common.seed_for(plans[0]['scene'],0),2628542810)

    def test_sample_std_not_variance_and_valid_axes(self):
        chunks=[[[float(i)]*7 for _ in range(10)] for i in range(8)]
        self.assertAlmostEqual(common.disagreement(chunks,[2.]*7),math.sqrt(6)/2.)
        self.assertEqual(common.disagreement([chunks[0]]*8,[1.]*7),0.)

    def test_churn_uses_shifted_overlap(self):
        previous=[[float(t)]*7 for t in range(10)]
        current=[[float(t+5)]*7 for t in range(10)]
        self.assertEqual(common.churn(current,previous,[1.]*7),0.)
        self.assertIsNone(common.churn(current,None,[1.]*7))
        self.assertEqual(common.norm([0.]*6+[1.],[1.]*7),1.)
        self.assertEqual(common.norm([0.]*6+[1.],[1.]*7,6),0.)

    def test_collision_clock_and_outcome(self):
        self.assertEqual([common.time_to_crash(x,3) for x in [1,2,3,4]],[2,1,0,None])
        self.assertEqual(common.time_to_crash(300,None),-1)
        self.assertEqual(common.outcome(True,3),'crash')
        self.assertEqual(common.outcome(False,None),'safe_incomplete')
        self.assertEqual(common.outcome(True,None),'safe_success')

    def test_sampling_seed_is_method_independent_and_distinct(self):
        values=[common.noise_seed('scene',7,2,i) for i in range(8)]
        self.assertEqual(len(set(values)),8)
        self.assertNotEqual(values,[common.noise_seed('scene',7,3,i) for i in range(8)])


class BudgetTests(unittest.TestCase):
    def request(self):
        from api_budget import MAX_PRICE
        return dict(model='z-ai/glm-4.5v',max_tokens=16384,
                    provider={'only':['z-ai'],'allow_fallbacks':False,'require_parameters':True,'max_price':MAX_PRICE})

    def test_unknown_reservations_and_retries_hit_exact_three_dollars(self):
        with tempfile.TemporaryDirectory() as t:
            b=CampaignBudget(t);b.refresh_prices=lambda: None
            try:
                for i in range(30):b.reserve(str(i),self.request())
                self.assertEqual(str(b.committed()),'3.00')
                with self.assertRaisesRegex(RuntimeError,'boundary'): b.reserve('31',self.request())
            finally:b.close()
            with self.assertRaisesRegex(RuntimeError,'unknown charge'): CampaignBudget(t)

    def test_actual_usage_settlement_and_unknown_cost(self):
        with tempfile.TemporaryDirectory() as t:
            b=CampaignBudget(t);b.refresh_prices=lambda: None
            try:
                b.reserve('a',self.request())
                with self.assertRaisesRegex(RuntimeError,'Unknown API charge'):b.settle('a',{})
                self.assertEqual(str(b.committed()),'0.10')
                b.settle('a',dict(usage={'cost':.002,'prompt_tokens':100,'completion_tokens':50},model='z-ai/glm-4.5v',provider='Z.AI'))
                self.assertEqual(str(b.committed()),'0.002')
                self.assertEqual(b.state['calls']['a']['usage']['prompt_tokens'],100)
            finally:b.close()


class AdapterTests(unittest.TestCase):
    def test_pinned_scoring_and_qp_stay_byte_identical(self):
        source=Path('/tmp/crashbench_uncertainty_main_aegis.py')
        if not source.exists():self.skipTest('Pinned evaluator supplied on Quest or during local verification')
        original=source.read_text()
        for method in ['nominal','aegis']:
            text=adapt(original,method);ast.parse(text)
            for start,end in [('                        v_ref =','                        action_input ='),
                              ('                    if collide_flag == False:','                    eef_pos ='),
                              ('                    if done:','                except Exception as e:')]:
                a=original[original.index(start):original.index(end,original.index(start))]
                b=text[text.index(start):text.index(end,text.index(start))]
                self.assertEqual(a,b)
            self.assertEqual(text.count('observer.after(obs, done, t)'),2)
            self.assertIn('action_plan.extend(action_chunk[: args.replan_steps])',text)


if __name__=='__main__':unittest.main()
