import ast
from pathlib import Path
import unittest
from verifier_training.v5_det_contract import (configuration,caption,floor_point,centre,inside,aegis_box,image_points,row_program,
    choose_fractions,det_reading,STATIC_FRACTIONS,TARGET_FRACTIONS,AEGIS_BOX_THRESHOLD,REPEAT_QUERIES)
from verifier_training.v5_pos_contract import anchors,SIZE
from verifier_training.tests.test_v5_contract import v5_row,RULES

ROOT=Path(__file__).resolve().parents[2]


def flat(cx,y):
    """A zero-height box whose floor point is (cx, y) pixels for every fraction."""
    return [cx/SIZE,y/SIZE,.02,0.]


class Points(unittest.TestCase):
    def test_captions_are_the_rule_phrases(self):
        self.assertEqual(caption('target'),'chocolate pudding');self.assertEqual(caption('goal'),'basket')
        self.assertEqual(caption('salad_dressing'),'salad dressing bottle');self.assertEqual(caption('red_coffee_mug_obstacle'),'red coffee mug')

    def test_floor_point_and_hit(self):
        box=[.5,.5,.2,.4]
        self.assertEqual(floor_point(box,.5),[224.,224.]);self.assertAlmostEqual(floor_point(box,1.)[1],.7*SIZE)
        self.assertAlmostEqual(floor_point(box,2.)[1],1.1*SIZE);self.assertEqual(centre(box),[224.,224.])
        self.assertTrue(inside([10,10],[12,12,20,20]));self.assertFalse(inside([7,10],[12,12,20,20]));self.assertFalse(inside(None,[0,0,1,1]))

    def test_aegis_box_takes_the_first_candidate_above_threshold(self):
        above=[dict(index=3,score=.4,box=[1,1,1,1]),dict(index=9,score=.9,box=[2,2,2,2])]
        self.assertEqual(aegis_box(dict(above=above)),[1,1,1,1]);self.assertIsNone(aegis_box(dict(above=[])))
        with self.assertRaisesRegex(ValueError,'order'):aegis_box(dict(above=above[::-1]))
        with self.assertRaisesRegex(ValueError,'below'):aegis_box(dict(above=[dict(index=1,score=AEGIS_BOX_THRESHOLD,box=[0,0,0,0])]))


class Program(unittest.TestCase):
    def test_row_program_uses_the_s4_rule(self):
        r=v5_row(rule=RULES[0]);a,b=anchors(r)
        pts={a:[100,100],b:[300,300],'target':[250,100],'goal':[100,250]}
        self.assertEqual(row_program(r,pts),0);self.assertEqual(row_program(r,dict(pts,goal=[260,90])),1)
        self.assertIsNone(row_program(r,dict(pts,goal=None)));self.assertIsNone(row_program(r,dict(pts,**{b:[100,100]})))

    def test_image_points_use_the_target_fraction_for_the_pudding_only(self):
        box=[.5,.5,.2,.4];p=image_points({'target':box,'goal':box,'x':None},.5,1.)
        self.assertEqual(p['goal'],floor_point(box,.5));self.assertEqual(p['target'],floor_point(box,1.));self.assertIsNone(p['x'])

    def test_fractions_are_chosen_on_training_rows_with_the_error_tie_break(self):
        r=v5_row(split='train',rule=RULES[2],target=True,goal=True);a,b=anchors(r)
        # Line y = 200; the goal is below it; the pudding's floor point is too (y = 210) but its box centre is above.
        # The goal's true point is 100 px off, so the median error over the four points moves with the pudding's error.
        boxes={'i0000':{a:flat(0,200),b:flat(400,200),'goal':flat(150,300),'target':[100/SIZE,.4,.05,.1]}}
        truth={'i0000':{'points':{a:[0,200],b:[400,200],'goal':[150,400],'target':[100,210]}}}
        chosen=choose_fractions([r],boxes,truth)
        self.assertEqual((chosen['static'],chosen['target']),(STATIC_FRACTIONS[0],1.25));self.assertEqual(chosen['training_accuracy'],1.)
        self.assertEqual(len(chosen['table']),len(STATIC_FRACTIONS)*len(TARGET_FRACTIONS))
        self.assertEqual({c['accuracy'] for c in chosen['table'] if c['target']==.5},{0.})
        with self.assertRaisesRegex(ValueError,'training rows only'):choose_fractions([v5_row()],boxes,truth)

    def test_reading(self):
        self.assertEqual(det_reading(.86,[-.1,.1]),'detector_close');self.assertEqual(det_reading(.85,[.0,.2]),'detector_close')
        self.assertEqual(det_reading(.70,[.15,.3]),'training_needed');self.assertEqual(det_reading(.75,[.10,.3]),'training_needed')
        self.assertEqual(det_reading(.70,[.05,.3]),'between');self.assertEqual(det_reading(.80,[.12,.2]),'between')


class Configuration(unittest.TestCase):
    def test_configuration(self):
        c=configuration();self.assertEqual(c['version'],'V5-object-det-v1');self.assertFalse(c['training']);self.assertEqual(c['api_calls'],0)
        self.assertEqual(c['seeds_configuration']['all_seeds'],[713,714,715]);self.assertEqual(c['detector']['checkpoint_sha256'][:8],'3b3ca256')
        self.assertEqual(set(c['rules']),{'detector_close','training_needed','between'});self.assertNotIn('test',c['splits'])

    def test_detect_script_is_standalone_python38_with_the_contract_constants(self):
        source=(ROOT/'verifier_training/v5_det_detect.py').read_text();tree=ast.parse(source,feature_version=(3,8))
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):self.assertEqual(node.level,0);self.assertFalse((node.module or '').startswith('verifier'))
            if isinstance(node,ast.Import):self.assertFalse(any(n.name.startswith('verifier') for n in node.names))
        constants={t.id:n.value.value for n in tree.body if isinstance(n,ast.Assign) for t in n.targets
                   if isinstance(t,ast.Name) and isinstance(n.value,ast.Constant)}
        self.assertEqual(constants['AEGIS_BOX_THRESHOLD'],AEGIS_BOX_THRESHOLD);self.assertEqual(constants['REPEAT_QUERIES'],REPEAT_QUERIES)
        self.assertIn("box_threshold=0.0",source);self.assertIn('bert-base-uncased',source)

    def test_launcher_checks_the_freeze_and_waits_for_the_preflight(self):
        tree=ast.parse((ROOT/'verifier_training/launch_v5.py').read_text());fn=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='det'][0]
        text=ast.unparse(fn);self.assertIn('状态：已冻结',text);self.assertIn("dependency='afterok'",text);self.assertIn('active_gpus() == 0',text)
        self.assertNotIn('smoke',text)
        self.assertIn("detector['checkpoint_sha256']",text)
        sbatch=(ROOT/'verifier_training/v5_det_gpu.sbatch').read_text()
        self.assertIn('aegis-py38.sif.sha256',sbatch);self.assertIn('-I -u $CB_V5_ARGS',sbatch);self.assertIn('--gres=gpu:a100:1',sbatch)


if __name__=='__main__':
    unittest.main()
