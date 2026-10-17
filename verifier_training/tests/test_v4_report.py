import copy
import unittest
from verifier_training.fd1000_contract import configuration
from verifier_training.v4_report import comparison,arm_summaries,diagnostics_by_correctness,markdown,correct_map,seed_average,PRIMARY

EXIT='start_inside_goal_outside'


def base(unit,source,assignment,boundary):
    truth=int(boundary=='b');color=('orange','purple')[(assignment+truth)%2]
    facts=dict(protected=color,target_inside='yes',goal_inside='yes' if truth else 'no',separated='no' if truth else 'yes')
    return dict(sample_id=f'{unit}_{assignment}{boundary}',unit_id=unit,source_group=source,assignment=assignment,protected_boundary=boundary,
        construction_mode=EXIT,positive_side='both_inside',checkpoint=0,truth=truth,truth_facts=facts,generation_seconds=.1,generated_token_count=5)


def records(arm,wrong):
    out=[]
    for u in range(4):
        for assignment in (0,1):
            for boundary in 'ab':
                r=base(f'u{u}',f'g{u}',assignment,boundary);ok=r['sample_id'] not in wrong;pred=r['truth'] if ok else 1-r['truth']
                facts=dict(r['truth_facts']) if ok else dict(r['truth_facts'],target_inside='no')
                greedy=dict(predicted=pred,format_valid=True,p_feasible=.9 if pred else .1)
                names={'A':[('direct_free','direct')],'B':[('facts_free','facts')],'C':[('direct_forced','direct'),('facts_forced','facts')]}[arm]
                for reading,kind in names:
                    out.append(dict(r,reading=reading,kind=kind,facts=facts if kind=='facts' else None,**greedy))
                if arm!='A':
                    out.append(dict(r,reading=('facts_free' if arm=='B' else 'facts_forced')+'_sampled',kind='facts',
                        samples=[dict(predicted=pred,p_feasible=.8 if pred else .2)]*16))
                    out.append(dict(r,reading='oracle_facts',kind='facts',facts=r['truth_facts'],predicted=r['truth'],format_valid=True,p_feasible=.9 if r['truth'] else .1))
                    for field in ('target','goal'):
                        out.append(dict(r,reading='edit_'+field,kind='facts',predicted=1-r['truth'],text_implied_verdict=1-r['truth'],facts=None,format_valid=True,p_feasible=None))
    return out


class Report(unittest.TestCase):
    def setUp(self):
        self.runs={('A',713):records('A',{'u0_0a','u1_0a','u2_0b'}),('A',714):records('A',{'u0_0a'}),
                   ('B',713):records('B',set()),('B',714):records('B',{'u3_1b'}),
                   ('C',713):records('C',set()),('C',714):records('C',set())}
        self.groups={r['sample_id']:r['source_group'] for rs in self.runs.values() for r in rs}

    def test_primary_pairs_seed_averaged_rows_by_source(self):
        result=comparison(self.runs,*PRIMARY,self.groups)
        self.assertAlmostEqual(result['control_accuracy'],1-2/16);self.assertAlmostEqual(result['treatment_accuracy'],1-.5/16)
        self.assertAlmostEqual(result['mean_difference'],1.5/16);self.assertEqual(result['source_groups'],4)
        a=seed_average([correct_map(self.runs[('A',s)],'direct_free','verdict') for s in (713,714)]);self.assertEqual(a['u0_0a'],0.)
        rule=comparison(self.runs,('B','facts_free','rule'),('B','facts_free','verdict'),self.groups)
        self.assertEqual(rule['mean_difference'],0.)
        sampled=comparison(self.runs,('B','facts_free_sampled','sampled'),('B','facts_free','verdict'),self.groups,k=4)
        self.assertEqual(sampled['mean_difference'],0.)

    def test_summaries_diagnostics_and_markdown(self):
        arms=arm_summaries(self.runs,configuration('main'))
        self.assertAlmostEqual(arms['B facts_free']['verdict_accuracy'],1-.5/16)
        self.assertEqual(arms['B oracle_facts']['verdict_accuracy'],1.)
        self.assertEqual(arms['C edit_goal']['follows_edited_text'],1.)
        self.assertIn('16',arms['B facts_free_sampled'])
        diag=diagnostics_by_correctness(self.runs)
        self.assertEqual(diag['B']['oracle_facts']['greedy_wrong']['rows'],1)
        self.assertEqual(diag['B']['field_target_inside']['greedy_wrong']['rate'],0.)
        report=dict(split='validation',source_commit='x',seeds=[713,714],arms=arms,primary=comparison(self.runs,*PRIMARY,self.groups),
            secondary={'s':comparison(self.runs,('C','facts_forced','verdict'),('C','direct_forced','verdict'),self.groups)},
            success=dict(development_milestone_met=None,scientific_met=None))
        text=markdown(report);self.assertIn('PRIMARY B facts_free - A direct_free',text);self.assertIn('| B facts_free |',text)


if __name__=='__main__':unittest.main()
