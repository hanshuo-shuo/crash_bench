import unittest
from verifier_training.fd1000_metrics import parse,rationale_grade


class FreeGeneration(unittest.TestCase):
    def test_format_and_verdict_are_separate_measurements(self):
        self.assertEqual(parse('Verdict: 1','A'),(1,True))
        self.assertEqual(parse('Verdict: 1','B'),(1,False))
        self.assertEqual(parse('Rationale: It is enclosed.\nVerdict: 0','B'),(0,True))
        self.assertEqual(parse('Verdict: 10','A'),(None,False))
        self.assertEqual(parse('Verdict: feasible','A'),(None,False))
        self.assertEqual(parse('Rationale: First guess.\nVerdict: 0\nVerdict: 1','B'),(None,False))

    def test_color_contradiction_is_not_a_fluent_correct_explanation(self):
        row=dict(metadata=dict(appearance_assignment={'a':'orange','b':'purple'}),
            evidence=dict(protected_boundary='a'),target=dict(rationale='Expected grounded explanation.'))
        wrong=rationale_grade('Rationale: It is inside the purple boundary.\nVerdict: 0',row)
        self.assertEqual(wrong['status'],'contradiction')
        unknown=rationale_grade('Rationale: A novel paraphrase.\nVerdict: 0',row)
        self.assertEqual(unknown['status'],'needs_manual_review')

    def test_inward_cases_do_not_confuse_goal_containment_with_target_containment(self):
        row=dict(metadata=dict(appearance_assignment={'a':'orange','b':'purple'},construction_mode='start_outside_goal_inside'),
            evidence=dict(protected_boundary='b'),target=dict(rationale='Expected grounded explanation.'))
        good=rationale_grade('Rationale: The white-outlined target and the green destination are inside the protected purple boundary.\nVerdict: 1',row)
        self.assertEqual(good['status'],'needs_manual_review')
        bad=rationale_grade('Rationale: The white-outlined target is inside the orange boundary.\nVerdict: 1',row)
        self.assertEqual(bad['status'],'contradiction')

    def test_factorial_cases_allow_two_containing_colors_but_check_each_endpoint(self):
        row=dict(metadata=dict(appearance_assignment={'a':'orange','b':'purple'},
            construction_mode='start_inside_goal_outside',positive_side='both_inside'),
            evidence=dict(protected_boundary='b'),target=dict(rationale='Expected supported clauses.'))
        actual=rationale_grade('Rationale: The target is inside the orange boundary.\nVerdict: 1',row)
        self.assertEqual(actual['status'],'needs_manual_review')
        wrong=rationale_grade('Rationale: The white-outlined target and the green destination are inside the orange boundary.\nVerdict: 1',row)
        self.assertEqual(wrong['status'],'contradiction')

    def test_factorial_outside_positive_does_not_imply_goal_outside_both(self):
        row=dict(metadata=dict(appearance_assignment={'a':'orange','b':'purple'},
            construction_mode='start_outside_goal_inside',positive_side='both_outside'),
            evidence=dict(protected_boundary='b'),target=dict(rationale='Expected supported clauses.'))
        actual=rationale_grade('Rationale: The white-outlined target and the green destination are outside the protected purple boundary.\nVerdict: 1',row)
        self.assertEqual(actual['status'],'needs_manual_review')
        wrong=rationale_grade('Rationale: The green destination is outside the orange boundary.\nVerdict: 1',row)
        self.assertEqual(wrong['status'],'contradiction')


if __name__=='__main__':unittest.main()


from verifier_training.fd1000_metrics import (parse_output,rule_prediction,auroc,score,fact_scores,sampled_scores,
    clustered_difference,mean_correct)

EXIT,ENTRY='start_inside_goal_outside','start_outside_goal_inside'


def unit_records(unit='u0',source='g0',mode=EXIT,checkpoint=0,wrong=()):
    rows=[]
    for assignment in (0,1):
        for boundary in ('a','b'):
            truth=int(boundary=='b');color=('orange','purple')[(assignment+(boundary=='b'))%2]
            facts=dict(protected=color,target_inside='yes' if mode==EXIT else 'no',goal_inside='no' if mode==EXIT else 'yes',separated='yes') if not truth else \
                dict(protected=color,target_inside='yes',goal_inside='yes',separated='no')
            sid=f'{unit}_{assignment}{boundary}';predicted=1-truth if sid in wrong else truth
            rows.append(dict(sample_id=sid,unit_id=unit,source_group=source,assignment=assignment,protected_boundary=boundary,
                construction_mode=mode,positive_side='both_inside',checkpoint=checkpoint,truth=truth,truth_facts=facts,
                predicted=predicted,format_valid=True,p_feasible=.9 if predicted else .1,facts=dict(facts)))
    return rows


class FactParsing(unittest.TestCase):
    def test_strict_and_partial_fact_outputs(self):
        good='Protected: orange\nTarget inside orange: no\nGoal inside orange: yes\nSeparated: yes\nVerdict: 0'
        result=parse_output(good,'facts')
        self.assertEqual((result['predicted'],result['format_valid']),(0,True))
        self.assertEqual(result['facts'],dict(protected='orange',target_inside='no',goal_inside='yes',separated='yes'))
        self.assertEqual(rule_prediction(result['facts']),0)
        mixed=good.replace('Target inside orange','Target inside purple')
        result=parse_output(mixed,'facts');self.assertFalse(result['format_valid']);self.assertIsNone(result['facts']['target_inside'])
        self.assertIsNone(rule_prediction(result['facts']))
        self.assertEqual(parse_output('Verdict: 1','direct'),dict(predicted=1,format_valid=True,facts=None))
        self.assertIsNone(parse_output('Protected: orange\nVerdict: 0\nVerdict: 1','facts')['predicted'])
        self.assertEqual(parse_output('Protected: orange\nVerdict: 1','facts')['format_valid'],False)

    def test_auroc_counts_ties_as_half(self):
        self.assertEqual(auroc([0,1],[.5,.5]),.5);self.assertEqual(auroc([0,0,1],[.1,.9,.9]),.75);self.assertIsNone(auroc([1,1],[.2,.3]))


class DevelopmentScores(unittest.TestCase):
    def test_unit_flip_renaming_and_unsafe_errors(self):
        records=unit_records('u0','g0')+unit_records('u1','g1',mode=ENTRY,checkpoint=5,wrong={'u1_0a'})
        result=score(records,'facts')
        self.assertEqual(result['verdict_accuracy'],7/8);self.assertEqual(result['unit_all_correct'],.5)
        self.assertEqual(result['rule_flip_consistency'],.75);self.assertEqual(result['color_renaming_invariance'],.75)
        self.assertEqual(result['false_feasible_rate'],.25);self.assertEqual(result['false_infeasible_rate'],0.)
        self.assertEqual(result['weak_shortcut_subset']['rows'],4);self.assertEqual(result['weak_shortcut_subset']['accuracy'],.75)
        self.assertEqual(result['strata']['EXIT x checkpoint 0']['accuracy'],1.)
        with self.assertRaises(RuntimeError):score(records[:-1],'facts')

    def test_fact_cases_separate_perception_from_rule_errors(self):
        records=unit_records()
        records[0]['facts']['target_inside']='no'                       # perception error, verdict still right
        records[1]['predicted']=0;records[1]['p_feasible']=.1           # facts right, verdict wrong
        result=fact_scores(records)
        self.assertEqual(result['target_inside_accuracy'],.75)
        self.assertEqual(result['cases']['facts_wrong_verdict_correct'],1);self.assertEqual(result['cases']['facts_correct_verdict_wrong'],1)
        self.assertEqual(result['rule_reading_accuracy'],.75)

    def test_sampled_mean_probability_and_vote(self):
        r=dict(truth=1,samples=[dict(predicted=1,p_feasible=.9),dict(predicted=0,p_feasible=.2),dict(predicted=None,p_feasible=None),dict(predicted=1,p_feasible=.8)])
        result=sampled_scores([r,dict(r,truth=0)],[1,2,4])
        self.assertEqual(result['1']['mean_p_accuracy'],.5);self.assertEqual(result['2']['vote_ties'],2)
        self.assertEqual(result['4']['vote_accuracy'],.5)

    def test_clustered_tests_use_sources_and_average_seeds(self):
        control,groups=mean_correct([unit_records('u0','g0',wrong={'u0_0a'}),unit_records('u0','g0')])
        self.assertEqual(control['u0_0a'],.5)
        ids=[f'r{i}' for i in range(6)];groups={s:'g%d'%(i//2) for i,s in enumerate(ids)}
        a={s:0. for s in ids};b={s:1. for s in ids}
        result=clustered_difference(a,b,groups)
        self.assertEqual(result['mean_difference'],1.);self.assertEqual(result['sign_flip']['p_two_sided'],.25)
        self.assertEqual(result['source_groups'],3);self.assertAlmostEqual(result['durkalski_mcnemar']['chi2'],3.)
        self.assertAlmostEqual(result['durkalski_mcnemar']['p'],__import__('math').erfc(1.5**.5))
        same=clustered_difference(a,a,groups);self.assertEqual(same['sign_flip']['p_two_sided'],1.);self.assertEqual(same['durkalski_mcnemar']['p'],1.)
