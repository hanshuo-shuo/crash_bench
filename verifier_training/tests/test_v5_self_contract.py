import ast
import json
from pathlib import Path
import unittest
from verifier_training.v5_contract import validate,DATASET
from verifier_training.v5_pos_contract import configuration as s4_configuration,anchors
from verifier_training.v5_self_contract import (configuration,chain,cross_text,facts,parse,s5_reading,word,STEPS,ARM)
from verifier_training.tests.test_v5_contract import v5_row,RULES

ROOT=Path(__file__).resolve().parents[2]
S4_POINTS=Path('/projects/p33100/siosio/crashbench_safelibero/verifier_training/runs/20261025T180525Z_v5-s4_20913c71c3f3/POINTS.json')
HAVE_DATA=(DATASET/'dataset.jsonl').exists() and S4_POINTS.exists()


def points_for(row,target,goal,a=(137,193),b=(283,212)):
    x,y=anchors(row);return {x:a,y:b,'target':target,'goal':goal}


class Chain(unittest.TestCase):
    def test_protocol_example(self):
        c=chain(dict(a=[137,193],b=[283,212],target=[282,263],goal=[203,153]))
        self.assertEqual(c['vector'],[146,19]);self.assertEqual(c['target_offset'],[145,70]);self.assertEqual(c['goal_offset'],[66,-40])
        self.assertEqual(cross_text(c['vector'],c['target_offset'],c['target_cross']),'146*70 - 19*145 = 10220 - 2755 = 7465')
        self.assertEqual(cross_text(c['vector'],c['goal_offset'],c['goal_cross']),'146*(-40) - 19*66 = -5840 - 1254 = -7094')
        self.assertEqual(c['separated'],'yes')
        c=chain(dict(a=[300,180],b=[120,300],target=[250,330],goal=[140,150]))
        self.assertEqual(cross_text(c['vector'],c['target_offset'],c['target_cross']),'-180*150 - 120*(-50) = -27000 - (-6000) = -21000')
        self.assertEqual((word(5),word(-5),word(0)),('positive','negative','zero'))

    def test_targets_round_trip_through_the_parser(self):
        for target,goal in ((True,False),(True,True)):
            r=v5_row(rule=RULES[0],target=target,goal=goal);sep=r['target']['facts']['separated']=='yes'
            _,_,text=facts(r,points_for(r,(282,263),(203,153) if sep else (300,240)))
            p=parse(text);digit=int(r['target']['digit'])
            self.assertTrue(p['format_valid'] and p['all_steps'] and p['execution_agreement'],text)
            self.assertEqual((p['predicted'],p['program']),(digit,digit));self.assertEqual(p['separated'],r['target']['facts']['separated'])
            self.assertTrue(text.startswith('Line: ') and text.endswith('Verdict: %d'%digit));self.assertEqual(set(p['steps']),set(STEPS))

    def test_true_points_on_the_line_are_rejected(self):
        r=v5_row(rule=RULES[2],target=True,goal=True)
        with self.assertRaisesRegex(ValueError,'on the line'):facts(r,points_for(r,(210,202),(300,240),a=(100,200),b=(320,204)))


class Steps(unittest.TestCase):
    def setUp(self):
        r=v5_row(rule=RULES[0],target=True,goal=False);_,_,self.text=facts(r,points_for(r,(282,263),(203,153)))

    def check(self,text,wrong):
        p=parse(text);self.assertFalse(p['all_steps'])
        self.assertEqual({k for k,v in p['steps'].items() if not v},set(wrong),text);return p

    def test_a_consistent_slip_is_counted_once(self):
        # The product is wrong, but the subtraction uses the written product, so only the product step fails.
        p=self.check(self.text.replace('= 10220 - 2755 = 7465','= 10230 - 2755 = 7475'),{'target_products'})
        self.assertEqual(p['predicted'],0);self.assertTrue(p['execution_agreement'])

    def test_each_step_is_checked(self):
        self.check(self.text.replace('Line vector: [146, 19]','Line vector: [146, 18]'),{'vector','target_products','goal_products'})
        self.check(self.text.replace('Goal offset: [66, -40]','Goal offset: [66, -41]'),{'goal_offset','goal_products'})
        self.check(self.text.replace('= 10220 - 2755 = 7465','= 10220 - 2755 = 7475'),{'target_cross'})
        self.check(self.text.replace('Target side: positive','Target side: negative'),{'target_side','separated'})
        p=self.check(self.text.replace('Verdict: 0','Verdict: 1'),{'verdict'});self.assertFalse(p['execution_agreement'])
        self.check(self.text.replace('Separated: yes\nVerdict: 0','Separated: no\nVerdict: 1'),{'separated'})

    def test_missing_or_repeated_lines(self):
        p=self.check(self.text.replace('Goal offset: [66, -40]\n',''),{'goal_offset','goal_products'});self.assertFalse(p['format_valid'])
        p=parse(self.text.replace('Target side: positive\n','Target side: positive\nTarget side: positive\n'))
        self.assertFalse(p['steps']['target_side']);self.assertFalse(p['format_valid'])
        p=parse('Verdict: 0');self.assertFalse(any(p['steps'][k] for k in STEPS if k!='verdict'));self.assertIsNone(p['program'])

    def test_own_verdict_and_program_can_disagree(self):
        # Right points, a wrong sign word carried through: the program is right, the model's own verdict is not.
        p=parse(self.text.replace('Goal side: negative\nSeparated: yes\nVerdict: 0','Goal side: positive\nSeparated: no\nVerdict: 1'))
        self.assertEqual((p['program'],p['predicted']),(0,1));self.assertFalse(p['steps']['goal_side']);self.assertTrue(p['steps']['separated'])


class Configuration(unittest.TestCase):
    def test_settings_follow_s4(self):
        c=configuration();s4=s4_configuration()
        self.assertEqual(c['s3_configuration'],s4['s3_configuration']);self.assertEqual(c['s4_configuration'],s4)
        self.assertEqual((c['arm'],c['seeds'],c['maximum_new_tokens'],c['primary_reading']),(ARM,[713],256,'model'))
        self.assertEqual(c['optimizer_updates'],448);self.assertEqual(c['api_calls'],0);self.assertFalse(c['hyperparameter_search'])

    def test_reading(self):
        self.assertEqual(s5_reading(.86,[.3,.4]),'self_computes');self.assertEqual(s5_reading(.85,[.01,.4]),'self_computes')
        self.assertEqual(s5_reading(.84,[.2,.3]),'partial');self.assertEqual(s5_reading(.65,[.1,.2]),'partial')
        self.assertEqual(s5_reading(.60,[.05,.2]),'no_self_compute');self.assertEqual(s5_reading(.62,[.05,.2]),'unclear')
        self.assertEqual(s5_reading(.90,[-.01,.2]),'unclear')


class Code(unittest.TestCase):
    def fn(self,path,name):
        return [n for n in ast.parse((ROOT/path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name][0]

    def test_train_p_defaults_are_p_and_hooks_replace_them(self):
        from verifier_training import v5_pos_run as R
        from verifier_training.v5_self_run import s_item,evaluate_s,summary_s
        self.assertEqual(R.arm_hooks(),dict(arm='P',item=R.p_item,evaluate=R.evaluate,summary=R.summary,selection='program'))
        h=R.arm_hooks(dict(arm='S',item=s_item,evaluate=evaluate_s,summary=summary_s,selection='model'))
        self.assertEqual(h,dict(arm='S',item=s_item,evaluate=evaluate_s,summary=summary_s,selection='model'))
        self.assertIn("h = arm_hooks(hooks)",ast.unparse(self.fn('verifier_training/v5_pos_run.py','train_p')))

    def test_launcher_checks_the_freeze_and_waits_for_the_preflight(self):
        text=ast.unparse(self.fn('verifier_training/launch_v5.py','s5'))
        self.assertIn('状态：已冻结',text);self.assertIn("dependency='afterok'",text);self.assertIn('active_gpus() == 0',text)
        self.assertIn("'gpulong'",text);self.assertNotIn('smoke',text)
        self.assertIn('#SBATCH --time=20:00:00',(ROOT/'verifier_training/v5_gpu_long.sbatch').read_text())

    def test_smoke_orders_keep_the_sequence(self):
        from verifier_training.v5_self_run import smoke_orders
        order=[[['a','facts'],['b','facts']],[['c','facts'],['d','facts']],[['e','facts'],['f','facts']]]
        self.assertEqual(smoke_orders(order,{'b','c','f'},2),[[['b','facts'],['c','facts']],[['f','facts']]])


@unittest.skipUnless(HAVE_DATA,'S2 dataset and S4 points not available')
class RealRows(unittest.TestCase):
    def test_real_chains_check_out(self):
        points=json.loads(S4_POINTS.read_text());rows=[r for r in validate(DATASET,include_test=False) if r['split']=='validation'][:120]
        for r in rows:
            _,_,text=facts(r,points[r['metadata']['image_id']]['points']);p=parse(text)
            self.assertTrue(p['all_steps'] and p['format_valid']);self.assertEqual(p['predicted'],int(r['target']['digit']))


if __name__=='__main__':
    unittest.main()
