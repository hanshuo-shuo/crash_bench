import unittest
from verifier_training.v5_metrics import parse_output,program_verdict,score,paired_difference,cluster_interval,row_record,summarize
from verifier_training.tests.test_v5_contract import v5_row

GOOD='Line: red coffee mug | salad dressing bottle\nTarget on robot side: yes\nGoal on robot side: no\nSeparated: yes\nVerdict: 0'


def record(sample,image,truth,predicted,facts=None,source='s0',kind='facts',lookup=None):
    t=dict(line='a | b',target_robot_side='yes',goal_robot_side='no' if truth==0 else 'yes',separated='yes' if truth==0 else 'no')
    return dict(sample_id=sample,image_id=image,source_id='x/'+source,truth=truth,truth_facts=t,predicted=predicted,p_feasible=None,
        format_valid=predicted is not None,facts=facts if facts is not None else dict(t),reading='facts_free',kind=kind,time_bin=0,
        forced_prefix=None,student_input_sha256=sample,lookup_prediction=lookup,generation_seconds=1.,generated_token_count=30)


class Parsing(unittest.TestCase):
    def test_strict_facts_format(self):
        p=parse_output(GOOD,'facts')
        self.assertEqual(p['predicted'],0);self.assertTrue(p['format_valid'])
        self.assertEqual(p['facts'],dict(line='red coffee mug | salad dressing bottle',target_robot_side='yes',goal_robot_side='no',separated='yes'))
        self.assertEqual(program_verdict(p['facts']),0)

    def test_malformed_outputs(self):
        cases={'missing goal':GOOD.replace('Goal on robot side: no\n',''),'extra text':'Sure.\n'+GOOD,
               'bad value':GOOD.replace('Target on robot side: yes','Target on robot side: maybe'),
               'two verdicts':GOOD+'\nVerdict: 1','reordered':GOOD.replace('Target on robot side: yes\nGoal on robot side: no','Goal on robot side: no\nTarget on robot side: yes')}
        for name,text in cases.items():self.assertFalse(parse_output(text,'facts')['format_valid'],name)
        self.assertIsNone(parse_output(cases['two verdicts'],'facts')['predicted'])
        self.assertIsNone(parse_output(cases['missing goal'],'facts')['facts']['goal_robot_side'])
        self.assertIsNone(program_verdict(parse_output(cases['bad value'],'facts')['facts']))
        self.assertEqual(parse_output(cases['extra text'],'facts')['predicted'],0)
        self.assertEqual(parse_output('Verdict: 1','direct'),dict(predicted=1,format_valid=True,facts=None))
        self.assertFalse(parse_output('Verdict: 1 because','direct')['format_valid'])
        self.assertEqual(program_verdict(dict(target_robot_side='no',goal_robot_side='no')),1)


class Scores(unittest.TestCase):
    def test_twin_images_facts_and_lookup(self):
        wrong_goal=dict(line='a | b',target_robot_side='yes',goal_robot_side='yes',separated='no')
        rs=[record('a','i0',0,0,lookup=0),record('b','i0',1,1,lookup=0),record('c','i1',0,1,facts=wrong_goal,lookup=0),record('d','i1',1,1,lookup=0),
            record('e','i2',1,1,source='s1',lookup=1)]
        s=score(rs,'facts')
        self.assertEqual((s['rows'],s['images'],s['twin_images'],s['correct']),(5,3,2,4))
        self.assertEqual(s['twin_image_all_correct'],.5);self.assertAlmostEqual(s['image_all_correct'],2/3)
        self.assertEqual(s['false_feasible_rate'],.5);self.assertEqual(s['lookup_accuracy'],.6);self.assertEqual(s['lookup_agreement'],.4)
        f=s['facts'];self.assertEqual(f['goal_robot_side_accuracy'],.8);self.assertEqual(f['target_robot_side_accuracy'],1.)
        self.assertEqual(f['side_errors'],dict(target_only=0,goal_only=1,both=0,none=4));self.assertEqual(f['fact_verdict_consistency'],1.)
        self.assertEqual(f['cases']['sides_wrong_verdict_wrong'],1)
        low,high=s['image_bootstrap_95pct']['verdict_accuracy'];self.assertLessEqual(low,.8);self.assertGreaterEqual(high,.8)
        self.assertIn('facts_free',summarize(rs))

    def test_row_record_reads_v5_metadata(self):
        r=v5_row();rec=row_record(r,'direct_free','direct',None,'h',1)
        self.assertEqual((rec['truth'],rec['image_id'],rec['rule'],rec['lookup_prediction']),(0,'i0000',r['metadata']['rule'],1))


class Paired(unittest.TestCase):
    def test_paired_difference(self):
        a=[record(s,'i%d'%(i//2),i%2,0,kind='direct') for i,s in enumerate('abcdefgh')]
        b=[record(s,'i%d'%(i//2),i%2,i%2) for i,s in enumerate('abcdefgh')]
        d=paired_difference(a,b)
        self.assertEqual((d['control_accuracy'],d['treatment_accuracy'],d['difference'],d['images']),(.5,1.,.5,4))
        self.assertEqual((d['treatment_only_correct'],d['control_only_correct']),(4,0))
        self.assertEqual(d['image_bootstrap_95pct'],[.5,.5])
        with self.assertRaisesRegex(RuntimeError,'Unpaired'):paired_difference(a,b[:-1])
        forced=[dict(r,forced_prefix='Verdict: ') for r in a]
        with self.assertRaisesRegex(RuntimeError,'free generation'):paired_difference(forced,b)

    def test_cluster_interval_is_seeded(self):
        values=[1,0,1,1,0,0,1,0];groups=list('aabbccdd')
        self.assertEqual(cluster_interval(values,groups),cluster_interval(values,groups));self.assertIsNone(cluster_interval([1,0],['a','a']))


if __name__=='__main__':unittest.main()
