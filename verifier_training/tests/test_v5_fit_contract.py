import ast
from collections import Counter
from pathlib import Path
import unittest
from verifier_training.v5_contract import configuration as s3_configuration,validate,DATASET
from verifier_training.v5_fit_contract import (configuration,pool_images,subsets,orders,classify_run,size_class,decision,fact_lookup,
    SEEDS,REST)
from verifier_training.tests.test_v5_contract import v5_row,RULES

ROOT=Path(__file__).resolve().parents[2]


def train_row(sample,image,rule,target,goal,lift=.01,layout='counterfactual',source=0,time_bin=0):
    r=v5_row(sample,split='train',image=image,rule=rule,target=target,goal=goal,time_bin=time_bin)
    r['metadata'].update(lift=lift,layout_kind=layout,source_id='safelibero_object/II/task1/episode%d'%source);return r


class Configuration(unittest.TestCase):
    def test_schedule_and_unchanged_settings(self):
        c=configuration();s3=s3_configuration()
        self.assertEqual(c['s3_configuration'],s3);self.assertEqual(c['seeds'],list(SEEDS))
        for size,spec in c['sizes'].items():
            self.assertEqual(spec['rows'],int(size));self.assertEqual(spec['rows'],2*spec['images'])
            self.assertEqual(spec['epochs']*spec['rows']//c['effective_batch_size'],spec['optimizer_updates'])
            self.assertEqual(spec['evaluation_updates'][-1],spec['optimizer_updates'])
        for key in c['unchanged_from_s3']:self.assertIn(key,s3)
        self.assertEqual(set(c['rules']['decisions']),{'cannot_fit','fits_16_only','fits_and_transfers','fits_without_transfer','inconclusive'})
        self.assertFalse(c['hyperparameter_search']);self.assertEqual(c['api_calls'],0)


class Selection(unittest.TestCase):
    def test_pool_keeps_resting_counterfactual_twins_only(self):
        rows=[train_row('a0','i0',RULES[0],True,False),train_row('a1','i0',RULES[1],True,True),
              train_row('b0','i1',RULES[0],True,False,layout='natural'),train_row('b1','i1',RULES[1],True,True,layout='natural'),
              train_row('c0','i2',RULES[0],True,False,lift=REST+.01),train_row('c1','i2',RULES[1],True,True),
              train_row('d0','i3',RULES[0],True,False),train_row('d1','i3',RULES[1],True,False)]
        self.assertEqual(set(pool_images(rows)),{'i0'})

    def test_orders_cover_each_row_every_epoch(self):
        c=configuration();ids=['t%02d'%i for i in range(16)];plan=orders(list(reversed(ids)),'16',c)
        self.assertEqual(set(plan),{str(s) for s in SEEDS})
        for order in plan.values():
            self.assertEqual(len(order),160);flat=[x for b in order for x in b]
            for e in range(40):self.assertEqual(Counter(s for s,k in flat[16*e:16*e+16]),Counter({s:1 for s in ids}))
            self.assertEqual({k for s,k in flat},{'facts'})
        with self.assertRaisesRegex(ValueError,'size'):orders(ids[:15],'16',c)

    def test_fact_lookup(self):
        train=[train_row('t0','i0',RULES[0],True,False),train_row('t1','i1',RULES[0],True,True),train_row('t2','i2',RULES[0],True,False)]
        guess=fact_lookup(train,'goal_robot_side')
        self.assertEqual(guess(train_row('v','i9',RULES[0],False,True)),'no');self.assertEqual(guess(train_row('w','i9',RULES[2],False,True)),'no')

    @unittest.skipUnless((DATASET/'dataset.jsonl').exists(),'S2 dataset not mounted')
    def test_real_fit_sets_are_nested_balanced_and_disjoint_from_held_out(self):
        c=configuration();train=[r for r in validate() if r['split']=='train'];parts=subsets(train,c)
        self.assertEqual(parts,subsets(list(reversed(train)),c))
        by_id={r['sample_id']:r for r in train}
        self.assertLessEqual(set(parts['16']['sample_ids']),set(parts['64']['sample_ids']))
        self.assertLessEqual(set(parts['16']['images']),set(parts['64']['images']))
        self.assertFalse(set(parts['64']['images'])&set(parts['held_out']['images']))
        for name,rows,images in (('16',16,8),('64',64,32),('held_out',264,132)):
            part=parts[name];self.assertEqual((len(part['sample_ids']),len(part['images'])),(rows,images),name)
            self.assertEqual(part['labels'],{'0':rows//2,'1':rows//2},name)
            members=[by_id[s] for s in part['sample_ids']]
            self.assertTrue(all(r['metadata']['layout_kind']=='counterfactual' and r['metadata']['lift']<REST for r in members))
            self.assertEqual(set(Counter(r['metadata']['image_id'] for r in members).values()),{2})
        sources=lambda name:Counter(by_id[s]['metadata']['source_id'] for s in parts[name]['sample_ids'])
        self.assertLessEqual(max(sources('16').values()),2);self.assertLessEqual(max(sources('64').values()),8)


class Rules(unittest.TestCase):
    def test_classification_and_decision(self):
        t=configuration()['thresholds']
        self.assertEqual(classify_run(1.,1.,.05,t),'fits');self.assertEqual(classify_run(1.,1.,.2,t),'partial')
        self.assertEqual(classify_run(.9,.6,.05,t),'does_not_fit');self.assertEqual(classify_run(1.,1.,.5,t),'does_not_fit')
        self.assertEqual(classify_run(.9,.8,.2,t),'partial')
        self.assertEqual(size_class(['fits','fits','partial']),'fits');self.assertEqual(size_class(['fits','partial','does_not_fit']),'mixed')
        self.assertEqual(decision('does_not_fit','fits',.3,t),'cannot_fit');self.assertEqual(decision('fits','does_not_fit',.3,t),'fits_16_only')
        self.assertEqual(decision('fits','fits',.10,t),'fits_and_transfers');self.assertEqual(decision('fits','fits',.09,t),'fits_without_transfer')
        self.assertEqual(decision('fits','partial',.3,t),'inconclusive');self.assertEqual(decision('mixed','fits',.3,t),'inconclusive')


class Isolation(unittest.TestCase):
    def test_fit_stages_read_training_rows_only(self):
        tree=ast.parse((ROOT/'verifier_training/v5_fit_run.py').read_text());functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        for name in ('context','fit','fit_run','report','reference','side_readout'):
            calls=[n for n in ast.walk(functions[name]) if isinstance(n,ast.Call) and getattr(n.func,'id',None)=='validate']
            self.assertTrue(all([(k.arg,k.value.value) for k in c.keywords]==[('include_test',False)] for c in calls),name)
            literals={n.value for n in ast.walk(functions[name]) if isinstance(n,ast.Constant)}
            self.assertFalse({'test','test_natural','validation','validation_natural','TEST_FREEZE.json'}&literals,name)


if __name__=='__main__':unittest.main()
