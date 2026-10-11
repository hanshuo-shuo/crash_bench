import unittest
from verifier_training.review_cases import issues,manual_review,MANUAL
from verifier_training.common import read


class ReviewCases(unittest.TestCase):
    def test_correct_visibility_cannot_hide_wrong_zero_budget_verdict(self):
        r={'format_valid':True,'verdict_correct':False,
           'rationale_assessment':{'status':'supported_complete','contradictions':[],'unsupported':[],'unparsed':[]}}
        self.assertIn('violates_zero_action_contract_on_recorded_initial_state',issues(r,'B','horizon_0'))
    def test_image_shuffle_recipient_error_is_qualified(self):
        r={'format_valid':True,'verdict_correct':False}
        self.assertIn('recipient_label_mismatch_is_not_a_donor_label_error',issues(r,'A','shuffled_pairs'))
    def test_unparsed_prose_is_never_automatically_supported(self):
        r={'format_valid':True,'verdict_correct':True,
           'rationale_assessment':{'status':'needs_manual_review','contradictions':[],'unsupported':[],'unparsed':['claim']}}
        self.assertIn('unparsed_prose_needs_manual_review',issues(r,'B','original'))
    def test_manual_review_applies_only_to_exact_record(self):
        fixture=read(MANUAL)
        r={'input_sha256':fixture['input_sha256'],'generated':fixture['generated']}
        self.assertEqual(manual_review(r,'B','text_only'),fixture['rationale_status'])
        self.assertIsNone(manual_review(r,'B','original'))
        self.assertIsNone(manual_review(dict(r,generated=r['generated']+' altered'),'B','text_only'))
        self.assertIsNone(manual_review(dict(r,input_sha256='different'),'B','text_only'))

if __name__=='__main__':unittest.main()
