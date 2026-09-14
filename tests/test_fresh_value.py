import copy
import unittest

from crashbench.fresh_value import (
    freeze_a, score_choices, summarize_ab, validate_anchors, validate_records,
)


def fixture():
    anchors = [dict(episode_id='n'+str(i), source='s', source_id='s00',
        condition=c, role='fresh_development', triggered=i==0, risk=.6,
        bundle_id='bundle', bundle_sha256='hash',
        horizons={'440':dict(success=1,accident=0,steps=70)})
        for i,c in enumerate(('glass','offpath','noglass'))]
    blocks = {}
    # Same-policy pseudo-choice has an A winner but reverses in B/C.
    for pi,p in enumerate(('A','B','C')):
        records = []
        for j in (0,1):
            for o in (0,1):
                success = (o==1 or j==1) if p=='A' else (o==0 and j==0)
                records.append(dict(episode_id='n0',source='s',condition='glass',
                    role='fresh_development',bundle_id='bundle',bundle_sha256='hash',
                    phase=p,repeat=2*pi+j,option=o,
                    horizons={'440':dict(success=int(success),accident=int(not success),steps=60)}))
        blocks[p] = records
    return anchors,blocks


class FreshValueTest(unittest.TestCase):
    def test_complete_denominator_includes_shared_outcomes(self):
        anchors, blocks = fixture()
        validate_anchors(anchors,expected_sources=1)
        choices = freeze_a(anchors,blocks['A'],dict(BenefitGate_threshold=None,RiskDetour_threshold=.53))
        rows = score_choices(anchors,blocks,choices)
        keep = [r for r in rows if r['method']=='real_full' and
                r['selection_horizon']==440 and r['evaluation_horizon']==440]
        self.assertEqual(len(keep),3)
        self.assertAlmostEqual(sum(r['A_gain'] for r in keep)/3,1/6)
        self.assertAlmostEqual(sum(r['C_gain'] for r in keep)/3,-1/6)
        self.assertTrue(all(r['choice']==0 for r in keep if r['shared']))

    def test_b_c_cannot_supply_a_selection(self):
        anchors,blocks=fixture()
        for phase in ('B','C'):
            with self.assertRaises(ValueError):
                freeze_a(anchors,blocks[phase],dict(BenefitGate_threshold=None,RiskDetour_threshold=.53))

    def test_duplicate_wrong_bundle_and_missing_repeat_rejected(self):
        anchors,blocks=fixture()
        for bad in (blocks['A'][:-1],blocks['A']+[blocks['A'][0]]):
            with self.assertRaises(ValueError): validate_records(anchors,bad,'A')
        bad=copy.deepcopy(blocks['A']);bad[0]['bundle_sha256']='different'
        with self.assertRaises(ValueError): validate_records(anchors,bad,'A')

    def test_pseudo_arm_index_stays_frozen_in_new_blocks(self):
        anchors,blocks=fixture()
        choices=freeze_a(anchors,blocks['A'],dict(BenefitGate_threshold=None,RiskDetour_threshold=.53))
        row=next(r for r in score_choices(anchors,blocks,choices) if r['id']=='n0' and
            r['method']=='pseudo_one' and r['selection_horizon']==440 and r['evaluation_horizon']==440)
        self.assertEqual((row['A_gain'],row['B_gain'],row['C_gain']),(1,-1,-1))

    def test_forecast_is_unaffected_by_later_c(self):
        anchors,blocks=fixture()
        choices=freeze_a(anchors,blocks['A'],dict(BenefitGate_threshold=None,RiskDetour_threshold=.53))
        before=summarize_ab(score_choices(anchors,{p:blocks[p] for p in ('A','B')},choices))
        after=summarize_ab(score_choices(anchors,blocks,choices))
        self.assertEqual(before,after)

    def test_history_all_base_gate_does_not_refit(self):
        anchors,blocks=fixture()
        choices=freeze_a(anchors,blocks['A'],dict(BenefitGate_threshold=None,RiskDetour_threshold=.53))
        self.assertTrue(all(v==0 for f in choices if f['method']=='BenefitGate' for v in f['choices'].values()))


if __name__ == '__main__': unittest.main()
