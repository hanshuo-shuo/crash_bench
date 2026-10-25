import ast
from pathlib import Path
import unittest
from verifier_training.launch_verdict_diagnostic import ASSETS
from verifier_training.v5_contract import configuration as s3_configuration,validate,DATASET
from verifier_training.v5_pos_contract import (configuration,image_points,facts,answer,parse,program_separated,side,s4_reading,
    entity_phrase,point_prompt,parse_point,anchors,FORCED,SIZE)
from verifier_training.tests.test_v5_contract import v5_row,RULES

ROOT=Path(__file__).resolve().parents[2]
HAVE_DATA=(DATASET/'dataset.jsonl').exists()


def points_for(row,target,goal,a=(100,100),b=(300,300)):
    x,y=anchors(row);return {x:a,y:b,'target':target,'goal':goal}


class Program(unittest.TestCase):
    def test_sides_and_separation(self):
        self.assertEqual(side((0,0),(10,0),(5,3)),1);self.assertEqual(side((0,0),(10,0),(5,-3)),-1);self.assertEqual(side((0,0),(10,0),(5,0)),0)
        self.assertTrue(program_separated((0,0),(10,0),(5,3),(5,-3)));self.assertFalse(program_separated((0,0),(10,0),(5,3),(1,8)))
        self.assertFalse(program_separated((0,0),(10,0),(5,0),(5,-3)))

    def test_targets_round_trip_through_the_parser(self):
        for target,goal in ((True,False),(True,True)):
            r=v5_row(target=target,goal=goal);sep=r['target']['facts']['separated']=='yes'
            pts=points_for(r,(250,100),(100,250) if sep else (260,90))
            f,_=facts(r,pts);text=answer(f,r['target']['digit']);p=parse(text)
            self.assertTrue(text.startswith(FORCED));self.assertTrue(p['format_valid']);self.assertEqual(p['program'],int(r['target']['digit']))
            self.assertEqual(p['predicted'],int(r['target']['digit']));self.assertEqual(p['separated'],r['target']['facts']['separated'])
            self.assertEqual(p['points']['target'],[250,100])

    def test_label_disagreement_and_out_of_frame_points_are_rejected(self):
        r=v5_row(target=True,goal=False)
        with self.assertRaisesRegex(ValueError,'disagrees'):facts(r,points_for(r,(250,100),(260,90)))
        with self.assertRaisesRegex(ValueError,'outside'):facts(r,points_for(r,(250,100),(100,SIZE+2)))

    def test_malformed_outputs(self):
        r=v5_row();f,_=facts(r,points_for(r,(250,100),(100,250)));good=answer(f,'0')
        missing=good.replace('Goal point: [100, 250]\n','');extra='Sure.\n'+good;two=good+'\nVerdict: 1'
        self.assertIsNone(parse(missing)['program']);self.assertFalse(parse(missing)['format_valid'])
        self.assertEqual(parse(extra)['program'],0);self.assertFalse(parse(extra)['format_valid']);self.assertIsNone(parse(two)['predicted'])
        same=good.replace('Line points: [100, 100] [300, 300]','Line points: [100, 100] [100, 100]');self.assertIsNone(parse(same)['program'])


class Settings(unittest.TestCase):
    def test_configuration_keeps_s3_settings(self):
        c=configuration();s3=s3_configuration();self.assertEqual(c['s3_configuration'],s3)
        for key in c['unchanged_from_s3']:self.assertIn(key,s3)
        self.assertEqual((c['seeds'],c['primary_reading'],c['view']),([713],'program','overview_a'))
        self.assertEqual(set(c['rules']),{'positions_ground','positions_fail','unclear'});self.assertEqual(c['api_calls'],0)

    def test_reading(self):
        self.assertEqual(s4_reading(.7,.5,[.1,.3]),'positions_ground');self.assertEqual(s4_reading(.7,.5,[-.01,.3]),'unclear')
        self.assertEqual(s4_reading(.58,.5,[.01,.1]),'positions_fail');self.assertEqual(s4_reading(.64,.5,[.05,.2]),'unclear')
        self.assertEqual(s4_reading(.7,.65,[.0,.1]),'unclear')

    def test_pointing_prompts_and_parser(self):
        self.assertEqual(entity_phrase('target'),'the chocolate pudding');self.assertEqual(entity_phrase('goal'),'the basket')
        self.assertEqual(entity_phrase('salad_dressing'),'the salad dressing bottle');self.assertIn('point_2d',point_prompt('the basket'))
        self.assertEqual(parse_point('```json\n[{"point_2d": [123, 45], "label": "basket"}]\n```'),[123.,45.])
        self.assertEqual(parse_point('{"point_2d": [12.5, 400]}'),[12.5,400.]);self.assertEqual(parse_point('at [7, 8]'),[7.,8.])
        self.assertIsNone(parse_point('I cannot see it.'))


@unittest.skipUnless(HAVE_DATA,'S2 dataset not mounted')
class Geometry(unittest.TestCase):
    def test_projected_points_reproduce_every_label_on_a_sample_of_images(self):
        rows=validate();images={}
        for r in rows:images.setdefault(r['metadata']['image_id'],[]).append(r)
        for image in sorted(images)[::40]:
            info=image_points(DATASET,image)
            for r in images[image]:
                f,pts=facts(r,info['points'])
                for k in pts:self.assertTrue(0<=pts[k][0]<SIZE and 0<=pts[k][1]<SIZE)
            for k,box in info['boxes_2d'].items():
                u,v=info['points'][k] if k in info['points'] else (None,None)
                self.assertLess(box[0],box[2]);self.assertLess(box[1],box[3])


@unittest.skipUnless(HAVE_DATA and (ASSETS/'model').exists(),'Staged model or S2 dataset not mounted')
class Encoding(unittest.TestCase):
    def test_p_target_keeps_inputs_and_boundaries(self):
        from verifier_training.fd1000_run import processor_for,vision_inputs,encode,input_digest
        from verifier_training.v5_contract import assistant_target
        processor=processor_for(ASSETS);rows=validate();c=s3_configuration()
        for r in [next(x for x in rows if x['split']==s) for s in ('train','validation','validation_natural')]:
            vision=vision_inputs(processor,r,DATASET);f,_=facts(r,image_points(DATASET,r['metadata']['image_id'])['points'])
            p=encode(processor,r,DATASET,'facts',c['max_length'],vision,verify=True,answer=answer(f,r['target']['digit']),keys=('separated',),forced=FORCED)
            d=encode(processor,r,DATASET,'direct',c['max_length'],vision,answer=assistant_target(r,'direct'),keys=(),forced='Verdict: ')
            self.assertEqual(input_digest(p),input_digest(d));self.assertEqual(int(p['vm'].sum()),1);self.assertEqual(int(p['detail']['key_separated'].sum()),1)
            self.assertLess(p['fields']['input_ids'].shape[1],c['max_length'])


class Isolation(unittest.TestCase):
    def test_s4_stages_never_load_test_rows(self):
        for path,names in (('verifier_training/v5_pos_run.py',('context','preflight','train','report','evaluate')),
                           ('verifier_training/v5_pos_zero_shot.py',('context','main','run_model'))):
            tree=ast.parse((ROOT/path).read_text());functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
            for name in names:
                calls=[n for n in ast.walk(functions[name]) if isinstance(n,ast.Call) and getattr(n.func,'id',None)=='validate']
                self.assertTrue(all([(k.arg,k.value.value) for k in c.keywords]==[('include_test',False)] for c in calls),name)
                literals={n.value for n in ast.walk(functions[name]) if isinstance(n,ast.Constant)}
                self.assertFalse({'test','test_natural','TEST_FREEZE.json'}&literals,name)


if __name__=='__main__':unittest.main()
