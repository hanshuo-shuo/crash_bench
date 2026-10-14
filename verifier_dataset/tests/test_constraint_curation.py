import copy
import unittest
from verifier_dataset.constraint_curation import exclude
from verifier_dataset.tests import test_constraint_selection as fixtures


class ImageBoundCuration(unittest.TestCase):
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
