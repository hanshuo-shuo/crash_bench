import copy
import unittest
from verifier_dataset.pool_audit import group_components,shortcut_readiness,validate_public


class CandidateAudit(unittest.TestCase):
    def row(self,source='a',image='x'):
        return dict(student=dict(task='Move the bowl',constraints='No protected contact',
            images=[dict(camera=c,path='images/a/'+c+'.png') for c in ('external','wrist')]),
            metadata=dict(source_id=source,image_pair_sha256=image),training_eligible=False,
            reasoning_target=None,evidence=dict(kind='replayed_safe_success_suffix',feasibility='feasible'))

    def test_shared_source_and_shared_pixels_form_transitive_components(self):
        rows=[self.row('a','x'),self.row('a','y'),self.row('b','y'),self.row('c','z')]
        groups=group_components(rows)
        self.assertEqual(groups[0],groups[2]);self.assertNotEqual(groups[2],groups[3])
        reverse=group_components(list(reversed(rows)))
        self.assertEqual(groups,list(reversed(reverse)))

    def test_future_field_and_failure_evidence_are_rejected(self):
        for key in ('actions','rationale','verdict','source_id'):
            row=self.row();row['student'][key]='leak'
            with self.assertRaisesRegex(RuntimeError,'student field'):validate_public(row)
        row=self.row();row['evidence']['kind']='policy_failed'
        with self.assertRaisesRegex(RuntimeError,'unsupported evidence'):validate_public(row)

    def test_one_class_cannot_produce_a_binary_metric(self):
        result=shortcut_readiness(['feasible']*1001)
        self.assertFalse(result['binary_metrics_available']);self.assertIsNone(result['auroc'])

    def test_relative_image_reference_cannot_escape_source_export(self):
        row=self.row();row['student']['images'][0]['path']='../../evidence/private.png'
        with self.assertRaisesRegex(RuntimeError,'escapes'):validate_public(row)


if __name__=='__main__':unittest.main()
