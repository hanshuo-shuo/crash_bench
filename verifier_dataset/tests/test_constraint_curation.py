import copy
import unittest
from verifier_dataset.constraint_curation import exclude,reviewed_only,selection_settings
from verifier_dataset.tests import test_constraint_selection as fixtures


class ImageBoundCuration(unittest.TestCase):
    def test_reduction_retains_frozen_splits_balance_and_pre_fit_scope(self):
        from verifier_dataset.constraint_sides import selection_quotas
        original=selection_quotas();protocol=dict(selection_quotas=original)
        review=dict(model_predictions_used=False,final_selection=dict(target_units=224,
            reviewed_visible_only=True,basis='completed_visibility_reviews_before_v4_model_fitting',
            original_protocol_sha256='frozen',selection_quotas={s:{c:n for c in original[s]}
                for s,n in [('train',32),('validation',12),('test',12)]}))
        self.assertEqual(selection_settings(review,protocol,'frozen')[0],224)
        for mutation in ('digest','prediction','unreviewed','balance','enlarge','count'):
            bad=copy.deepcopy(review);p=bad['final_selection']
            if mutation=='digest':p['original_protocol_sha256']='other'
            if mutation=='prediction':bad['model_predictions_used']=True
            if mutation=='unreviewed':p['reviewed_visible_only']=False
            if mutation=='balance':p['selection_quotas']['train'][next(iter(original['train']))]=31
            if mutation=='enlarge':p['selection_quotas']['train']={c:40 for c in original['train']}
            if mutation=='count':p['target_units']=225
            with self.subTest(mutation=mutation),self.assertRaises(RuntimeError):selection_settings(bad,protocol,'frozen')

    def test_smaller_selection_never_admits_unreviewed_or_different_pixels(self):
        units=[fixtures.ContrastSelection().unit('a'),fixtures.ContrastSelection().unit('b')]
        review=dict(all_reviewed_units=[dict(source_id='a',checkpoint=0,status='premises_visible',
            failed_premises=[],image_sha256=['a00','a01'])])
        self.assertEqual(reviewed_only(units,review),units[:1])
        bad=copy.deepcopy(review);bad['all_reviewed_units'][0]['image_sha256']=['other']
        with self.assertRaisesRegex(RuntimeError,'different pixels'):reviewed_only(units,bad)
        bad=copy.deepcopy(review);bad['all_reviewed_units'][0]['failed_premises']=['target_identifiable']
        with self.assertRaisesRegex(RuntimeError,'Invalid'):reviewed_only(units,bad)
        bad=copy.deepcopy(review);bad['all_reviewed_units']*=2
        with self.assertRaisesRegex(RuntimeError,'Repeated'):reviewed_only(units,bad)

    def test_exclusion_removes_all_four_records_and_requires_actual_pixels(self):
        units=[fixtures.ContrastSelection().unit('a'),fixtures.ContrastSelection().unit('b')]
        review=dict(purpose='observable_premises_only',model_predictions_used=False,excluded_units=[dict(
            source_id='a',checkpoint=0,image_sha256=['a00','a01'],failed_premises=['target_identifiable'],
            reason='Target is occluded in both supplied views')])
        result=exclude(units,review)
        self.assertEqual(len(result),1);self.assertEqual(len(result[0]),4)
        self.assertEqual(result[0][0]['metadata']['source_id'],'b')
        bad=copy.deepcopy(review);bad['excluded_units'][0]['image_sha256']=['different']
        with self.assertRaisesRegex(RuntimeError,'different pixels'):exclude(units,bad)
        bad=copy.deepcopy(review);bad['model_predictions_used']=True
        with self.assertRaisesRegex(RuntimeError,'model outcomes'):exclude(units,bad)
        bad=copy.deepcopy(review);bad['excluded_units'][0]['failed_premises']=['model_wrong']
        with self.assertRaisesRegex(RuntimeError,'failed premise'):exclude(units,bad)


if __name__=='__main__':unittest.main()
