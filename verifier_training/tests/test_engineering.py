import copy
import unittest
from verifier_training.engineering_contract import *
from verifier_training.common import TASK, CONSTRAINTS


class Engineering(unittest.TestCase):
    def row(self):
        v={'views':[{'camera':c,'entities':{k:{'pixels_224':0 if k=='target' else 40} for k in ('target','goal','panels')}} for c in ('agentview','robot0_eye_in_hand')]}
        return {'student':{'images':[{'path':'a'},{'path':'b'}],'task':TASK,'constraints':CONSTRAINTS},
                'target':{'verdict':'infeasible','rationale':rationale(v)[0]},'metadata':{'visibility':v}}
    def test_complete_epochs_and_identical_order(self):
        orders=epoch_orders(120,713,10,4)
        self.assertEqual(len(orders),300)
        self.assertEqual(orders,epoch_orders(120,713,10,4))
        for e in range(10):self.assertEqual(sorted(sum(orders[e*30:(e+1)*30],[])),list(range(120)))
    def test_shuffling_is_derangement_without_labels(self):
        d=shuffled_donors(120);self.assertEqual(sorted(d),list(range(120)))
        self.assertTrue(all(i!=j for i,j in enumerate(d)))
    def test_constraint_edit_only_horizon(self):
        row=self.row();changed=diagnostic_row(row,'horizon_0')
        self.assertEqual(changed['student']['constraints'].replace('at most 0','at most 300'),CONSTRAINTS)
        self.assertEqual(row['student']['constraints'],CONSTRAINTS)
        self.assertEqual(changed['student']['images'],row['student']['images'])
    def test_text_only_has_no_images(self):
        messages=diagnostic_messages(self.row(),'B','text_only')
        self.assertEqual(len(messages[1]['content']),1)
        self.assertIn('No images are supplied',str(messages))
    def test_grounded_and_hallucinated_visibility(self):
        r=self.row();text='Rationale: '+r['target']['rationale']+'\nVerdict: infeasible'
        self.assertEqual(score_rationale(text,r,'original')['status'],'supported_complete')
        bad=text.replace('not visible in either view','visible in both views')
        self.assertEqual(score_rationale(bad,r,'original')['status'],'contradiction')
        self.assertEqual(score_rationale(text,r,'text_only')['status'],'unsupported_no_images')
    def test_unparsed_claim_never_passes(self):
        r=self.row();t='Rationale: The milk cannot pass through the gap.\nVerdict: infeasible'
        self.assertEqual(score_rationale(t,r,'original')['status'],'needs_manual_review')
    def test_format_requires_rationale_for_B(self):
        self.assertIsNone(parse_verdict('Verdict: feasible','B'))
        self.assertEqual(parse_verdict('Verdict: feasible','A'),'feasible')

if __name__=='__main__':unittest.main()
