import ast
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from verifier_dataset.v5_dataset import facts,facts_text,constraint_text,SCHEMA
from verifier_training.v5_contract import (configuration,validate,check_row,assistant_target,rule_verdict,readings,schedule,check_schedule,
    selection_subset,fit_subset,lookup_predictions,s3_reading,zero_shot_messages,FORCED_PREFIX,FACT_KEYS,DATASET,DATASET_HASHES,SELECT_IMAGES)

ROOT=Path(__file__).resolve().parents[2]
RULES=('line:red_coffee_mug_obstacle|salad_dressing','line:alphabet_soup|orange_juice','line:alphabet_soup|salad_dressing')


def v5_row(sample='v5o_00000',split='validation',image='i0000',rule=RULES[0],target=True,goal=False,time_bin=0):
    f=facts(dict(rule=rule,target_robot_side=target,goal_robot_side=goal,label=int(target!=goal)));digit='0' if target!=goal else '1'
    return dict(schema_version=SCHEMA,sample_id=sample,split=split,
        student=dict(images=[dict(camera=c,path='units/%s/%s.png'%(image,c),sha256=image+c) for c in ('overview_a','overview_b')],
                     task='pick up the chocolate pudding and place it in the basket',constraints=constraint_text(rule)),
        target=dict(verdict='feasible' if digit=='1' else 'infeasible',digit=digit,facts=f,rationale=facts_text(f),scope='conditional_on_stated_rule'),
        metadata=dict(source_id='safelibero_object/II/task1/episode%d'%(int(image[1:])%4),image_id=image,rule=rule,time_bin=time_bin,
                      layout=1,layout_kind='counterfactual',checkpoint=3))


def twins(n):
    rows=[]
    for i in range(n):
        image='i%04d'%i
        rows+=[v5_row('a%04d'%i,image=image,rule=RULES[0],target=True,goal=False),v5_row('b%04d'%i,image=image,rule=RULES[1],target=True,goal=True),
               v5_row('c%04d'%i,image=image,rule=RULES[2],target=False,goal=True)]
    return rows


class Targets(unittest.TestCase):
    def test_targets_follow_the_s2_facts_and_the_program_rule(self):
        for target,goal in ((True,True),(True,False),(False,True),(False,False)):
            r=v5_row(target=target,goal=goal);check_row(r)
            d=str(rule_verdict(r['target']['facts']['target_robot_side'],r['target']['facts']['goal_robot_side']))
            self.assertEqual(d,r['target']['digit']);self.assertEqual(assistant_target(r,'direct'),'Verdict: '+d)
            self.assertEqual(assistant_target(r,'facts'),r['target']['rationale']+'\nVerdict: '+d)
            self.assertTrue(assistant_target(r,'facts').startswith(FORCED_PREFIX['facts']))
        self.assertEqual(set(v5_row()['target']['facts']),set(FACT_KEYS))
        self.assertEqual(readings('A'),{'direct_free':('direct',None)});self.assertEqual(readings('B'),{'facts_free':('facts',None)})

    def test_inconsistent_rows_are_rejected(self):
        bad=v5_row();bad['target']['facts']['separated']='no'
        with self.assertRaisesRegex(ValueError,'Separated'):check_row(bad)
        bad=v5_row();bad['target']['digit']='1'
        with self.assertRaisesRegex(ValueError,'imply'):check_row(bad)
        bad=v5_row();bad['target']['facts']['line']='salad dressing bottle | red coffee mug'
        with self.assertRaisesRegex(ValueError,'Rationale|Line'):check_row(bad)
        bad=v5_row();bad['student']['images'][0]['path']='../x.png'
        with self.assertRaisesRegex(ValueError,'units'):check_row(bad)


class Schedules(unittest.TestCase):
    def test_one_epoch_covers_every_training_row_once_in_the_same_order(self):
        config=configuration();ids=['t%04d'%i for i in range(config['training_rows'])]
        orders=schedule(list(reversed(ids)),config)
        self.assertEqual(list(orders),['713'])
        for arm,order in orders['713'].items():
            check_schedule(order,ids,arm,config);self.assertEqual(len(order),448)
        flat={arm:[s for b in o for s,k in b] for arm,o in orders['713'].items()}
        self.assertEqual(flat['A'],flat['B']);self.assertEqual(Counter(flat['A']),Counter({s:1 for s in ids}))
        self.assertEqual({k for b in orders['713']['B'] for s,k in b},{'facts'})
        broken=[list(b) for b in orders['713']['A']];broken[0]=[broken[1][0]]+broken[0][1:]
        with self.assertRaisesRegex(ValueError,'repeats'):check_schedule(broken,ids,'A',config)
        with self.assertRaisesRegex(ValueError,'kind'):check_schedule(orders['713']['A'],ids,'B',config)

    def test_configuration(self):
        c=configuration()
        self.assertEqual((c['arms'],c['seeds'],c['epochs']),(['A','B'],[713],1))
        self.assertEqual(c['optimizer_updates']*c['effective_batch_size'],c['training_rows']);self.assertEqual(c['evaluation_updates'],[150,300,448])
        self.assertEqual((c['lora_rank'],c['lora_alpha'],c['learning_rate'],c['max_length'],c['image_pixels']),(16,32,1e-4,2048,448*448))
        self.assertEqual(c['maximum_new_tokens'],{'direct':8,'facts':96});self.assertEqual(c['dataset_hashes'],DATASET_HASHES)
        self.assertFalse(c['hyperparameter_search']);self.assertEqual(c['api_calls'],0)


class Subsets(unittest.TestCase):
    def test_selection_subset_is_balanced_and_deterministic(self):
        rows=twins(SELECT_IMAGES+20)+[v5_row('z0',image='i9999',target=True,goal=True)]
        chosen=selection_subset(rows);again=selection_subset(list(reversed(rows)))
        self.assertEqual(chosen,again);self.assertEqual(len(chosen),2*SELECT_IMAGES)
        picked=[r for r in rows if r['sample_id'] in set(chosen)]
        self.assertEqual(Counter(r['target']['digit'] for r in picked),{'0':SELECT_IMAGES,'1':SELECT_IMAGES})
        per=Counter(r['metadata']['image_id'] for r in picked);self.assertEqual(set(per.values()),{2});self.assertNotIn('i9999',per)
        with self.assertRaisesRegex(ValueError,'Too few'):selection_subset(twins(10))

    def test_training_fit_subset(self):
        rows=[v5_row('t%04d'%i,split='train') for i in range(400)]
        fit=fit_subset(rows);self.assertEqual(len(fit),256);self.assertEqual(fit,fit_subset(list(reversed(rows))))
        self.assertLessEqual(set(fit),{r['sample_id'] for r in rows})

    def test_lookup_uses_rule_and_progress(self):
        train=[v5_row('t0',split='train',time_bin=0,target=True,goal=False),v5_row('t1',split='train',time_bin=0,target=True,goal=False),
               v5_row('t2',split='train',time_bin=1,target=True,goal=True)]
        rows=[v5_row('v0',time_bin=0,target=True,goal=True),v5_row('v1',time_bin=1),v5_row('v2',rule=RULES[1])]
        self.assertEqual(lookup_predictions(train,rows),{'v0':0,'v1':1,'v2':0})


class Readings(unittest.TestCase):
    def test_preregistered_reading(self):
        fit={'A':.75,'B':.7}
        self.assertEqual(s3_reading(.5,.7,[.05,.3],fit),'b_grounds');self.assertEqual(s3_reading(.5,.7,[-.01,.3],fit),'unclear')
        self.assertEqual(s3_reading(.7,.75,[-.02,.1],fit),'both_ground');self.assertEqual(s3_reading(.55,.58,[-.1,.1],fit),'both_shortcut')
        self.assertEqual(s3_reading(.55,.58,[-.1,.1],{'A':.65,'B':.6}),'unclear');self.assertEqual(s3_reading(.62,.64,[-.1,.1],fit),'unclear')


class Prompts(unittest.TestCase):
    def test_zero_shot_prompts_share_the_student_input_and_hide_labels(self):
        r=v5_row()
        for mode in ('direct','facts'):
            m=zero_shot_messages(r,mode);text=m[1]['content'][2]['text']
            self.assertEqual([c['type'] for c in m[1]['content']],['image','image','text'])
            self.assertIn(r['student']['constraints'],text);self.assertIn(r['student']['task'],text)
            for hidden in (r['sample_id'],r['metadata']['rule'],r['target']['rationale'],'i0000'):self.assertNotIn(hidden,text)
        self.assertIn('Target on robot side:',zero_shot_messages(r,'facts')[1]['content'][2]['text'])


class TestIsolation(unittest.TestCase):
    def test_pilot_stages_never_load_test_rows(self):
        for path,names in (('verifier_training/v5_run.py',('preflight','train','zero_shot_3b','report','context')),
                           ('verifier_training/v5_zero_shot_32b.py',('main',)),('verifier_training/v5_contract.py',('campaign',))):
            tree=ast.parse((ROOT/path).read_text());functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
            for name in names:
                calls=[n for n in ast.walk(functions[name]) if isinstance(n,ast.Call) and getattr(n.func,'id',None)=='validate']
                self.assertTrue(all([(k.arg,k.value.value) for k in c.keywords]==[('include_test',False)] for c in calls),name)
                literals={n.value for n in ast.walk(functions[name]) if isinstance(n,ast.Constant)}
                self.assertFalse({'test','test_natural','TEST_FREEZE.json'}&literals,name+' must not name the test split')
        tree=ast.parse((ROOT/'verifier_training/v5_contract.py').read_text())
        campaign=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='campaign'][0]
        self.assertEqual(len([n for n in ast.walk(campaign) if isinstance(n,ast.Call) and getattr(n.func,'id',None)=='validate']),1)

    def test_tampered_dataset_fails_before_any_row(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            for f in DATASET_HASHES:(root/f).write_text('tampered')
            with self.assertRaisesRegex(ValueError,'Frozen S2 file changed'):validate(root)

    @unittest.skipUnless((DATASET/'dataset.jsonl').exists(),'S2 dataset not mounted')
    def test_frozen_s2_rows(self):
        rows=validate()
        self.assertEqual(Counter(r['split'] for r in rows),{'train':1792,'validation':1538,'validation_natural':79})
        train=[r for r in rows if r['split']=='train'];validation=[r for r in rows if r['split']=='validation']
        lookup=lookup_predictions(train,train);self.assertAlmostEqual(sum(lookup[r['sample_id']]==int(r['target']['digit']) for r in train)/len(train),.7600446,6)
        lookup=lookup_predictions(train,validation);self.assertEqual(sum(lookup[r['sample_id']]==int(r['target']['digit']) for r in validation)*2,len(validation))
        self.assertEqual(len(selection_subset(validation)),384)


if __name__=='__main__':unittest.main()
