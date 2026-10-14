import copy
import unittest
from verifier_dataset.finalize_constraints import final_rationale,review_matches


class ReviewedTargets(unittest.TestCase):
    def test_positive_explanation_does_not_claim_to_observe_a_hidden_witness(self):
        row=dict(metadata=dict(appearance_assignment={'a':'orange','b':'purple'}),evidence=dict(protected_boundary='b'))
        text=final_rationale(row)
        self.assertNotIn('recorded',text);self.assertNotIn('trajectory',text)
        self.assertIn('allows it to cross',text);self.assertIn('other stated safety constraints still apply',text)

    def test_same_image_protected_color_flip_changes_the_constraint_explanation(self):
        row=dict(metadata=dict(appearance_assignment={'a':'purple','b':'orange'}),evidence=dict(protected_boundary='a'))
        text=final_rationale(row);self.assertIn('rule protects purple',text);self.assertIn('forbids',text)

    def test_visual_review_is_bound_to_exact_images_and_all_premises(self):
        rows=[dict(metadata=dict(unit_id='u'),student=dict(images=[dict(sha256='x'),dict(sha256='y')]))]
        unit=dict(status='passed',image_sha256=['x','y'],target_identifiable=True,target_boundary_relation_visible=True,
            destination_relation_visible=True,both_rule_colors_identifiable=True)
        review=dict(complete=True,unresolved_units=[],prepared_dataset_sha256='data',reviewed_units={'u':unit})
        self.assertTrue(review_matches(review,rows,'data'))
        bad=copy.deepcopy(review);bad['reviewed_units']['u']['image_sha256']=['different','y']
        with self.assertRaisesRegex(RuntimeError,'different pixels'):review_matches(bad,rows,'data')
        bad=copy.deepcopy(review);bad['reviewed_units']['u']['target_boundary_relation_visible']=False
        with self.assertRaisesRegex(RuntimeError,'premises'):review_matches(bad,rows,'data')
        with self.assertRaisesRegex(RuntimeError,'another dataset'):review_matches(review,rows,'other')


if __name__=='__main__':unittest.main()
