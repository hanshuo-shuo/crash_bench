import unittest
from verifier_training.argmax_review import decision_rows,decision_summary

class ArgmaxReview(unittest.TestCase):
    def fixture(self,truth):
        candidates={'feasible':{'sequence_logp':-.7},'infeasible':{'sequence_logp':-.7001}}
        native={'sample_id':truth,'truth':truth,'own_prefix_cached':{'first_greedy_token_id':7,
                'processed_first_greedy_token_id':42,'candidates':candidates},'native_greedy_token_id':42,
            'native_raw_decisive_logp':{'feasible':-.7,'infeasible':-.7},
            'native_decisive_logp':{'feasible':-.3,'infeasible':-1.3},
            'canonical_and_generated_prefix_ids_identical':True,'native_replay_logp_max_error':0.,'matches_saved_text':True}
        teacher={'sample_id':truth,'conditions':{'ground_truth_rationale':{'candidates':{truth:{'greedy_first_token_id':7}}}}}
        return native,teacher
    def test_native_ties_do_not_follow_dictionary_order(self):
        n,t=self.fixture('infeasible')
        row=decision_rows([n],[t],{'feasible':[42],'infeasible':[7,8]})[0]
        self.assertEqual(row['raw_argmax_label'],'infeasible')
        self.assertEqual(row['raw_complete_sequence_choice'],'feasible')
        self.assertTrue(row['exact_raw_class_tie'])
    def test_penalty_can_hurt_negative_and_help_positive(self):
        pairs=[self.fixture(c) for c in ('infeasible','feasible')]
        rows=decision_rows([x[0] for x in pairs],[x[1] for x in pairs],{'feasible':[42],'infeasible':[7,8]})
        s=decision_summary(rows)
        self.assertEqual(s['penalty_transitions']['infeasible']['correct_to_wrong'],1)
        self.assertEqual(s['penalty_transitions']['feasible']['wrong_to_correct'],1)
        self.assertEqual(len(s['class_ties']),2)
    def test_missing_diagnostic_row_rejected(self):
        n,t=self.fixture('infeasible')
        with self.assertRaises(ValueError):decision_rows([n],[],{'feasible':[42],'infeasible':[7,8]})

if __name__=='__main__':unittest.main()
