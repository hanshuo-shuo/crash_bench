import ast
from collections import Counter
from pathlib import Path
import unittest
from verifier_training.fd1000_contract import configuration as phase2_configuration
from verifier_training.fit_contract import (configuration,phase2_files,overfit_units,overfit_orders,reading,last_epoch_loss,
    classify_run,arm_class,train_class,decision,SEEDS)

ROOT=Path(__file__).resolve().parents[2]
CELLS=('start_inside_goal_outside__both_inside','start_inside_goal_outside__both_outside',
       'start_outside_goal_inside__both_inside','start_outside_goal_inside__both_outside')


def unit_rows(uid,cell,group,split='train'):
    return [dict(sample_id=f'{uid}_{a}{b}',split=split,metadata=dict(unit_id=uid,construction_cell=cell,ancestry_component=group),
                 target=dict(verdict='feasible' if b=='b' else 'infeasible')) for a in (0,1) for b in 'ab']


def training_rows(shared_group=False):
    rows=[]
    for c,cell in enumerate(CELLS):
        for u in range(6):rows+=unit_rows(f'unit_{c}{u}',cell,'shared' if shared_group else f'group_{(c+u)%5}')
    return rows


class Configuration(unittest.TestCase):
    def test_hyperparameters_are_the_phase2_ones(self):
        config=configuration();base=config['phase2_configuration'];self.assertEqual(base,phase2_configuration('main'))
        o=config['overfit']
        self.assertEqual(o['optimizer_updates'],o['epochs']*o['rows']//o['effective_batch_size'])
        self.assertEqual(o['effective_batch_size'],base['effective_batch_size']);self.assertEqual(o['evaluation_updates'][-1],o['optimizer_updates'])
        for key in o['unchanged_from_phase2']:self.assertIn(key,base)
        self.assertEqual(config['train_evaluation']['checkpoints'],[128,384])
        self.assertTrue(set(config['train_evaluation']['checkpoints'])<=set(base['evaluation_updates']))
        self.assertEqual(set(config['rules']['decisions']),{'implementation_suspect','shared_pipeline_suspect','optimization_stall',
            'fits_training_data','inconclusive'})
        self.assertEqual((reading('A'),reading('B')),('direct_free','facts_free'))

    def test_phase2_evidence_covers_adapters_and_never_the_test(self):
        files=phase2_files()
        self.assertEqual(sum(f.endswith('adapter_model.safetensors') for f in files),6)
        self.assertIn('preflight/ENCODINGS.json',files)
        self.assertFalse([f for f in files if f.startswith('test') or 'TEST_FREEZE' in f or 'selected_adapter' in f])


class Subset(unittest.TestCase):
    def test_one_whole_unit_per_cell_from_distinct_groups(self):
        rows=training_rows();subset=overfit_units(rows)
        self.assertEqual(len(subset['sample_ids']),16);self.assertEqual(sorted(subset['cells']),list(CELLS))
        self.assertEqual(len(subset['source_groups']),4);self.assertEqual(subset['labels'],{'feasible':8,'infeasible':8})
        self.assertEqual(overfit_units(list(reversed(rows))),subset)
        units=Counter(s.rsplit('_',1)[0] for s in subset['sample_ids']);self.assertEqual(set(units.values()),{4})

    def test_refuses_other_splits_missing_cells_and_split_units(self):
        rows=training_rows();rows[0]=dict(rows[0],split='validation')
        with self.assertRaises(ValueError):overfit_units(rows)
        with self.assertRaises(ValueError):overfit_units([r for r in training_rows() if r['metadata']['construction_cell']!=CELLS[0]])
        with self.assertRaises(ValueError):overfit_units(training_rows()[1:])
        with self.assertRaises(ValueError):overfit_units(training_rows(shared_group=True))

    def test_orders_cover_every_row_once_per_epoch_and_match_across_arms(self):
        config=configuration();ids=overfit_units(training_rows())['sample_ids'];orders=overfit_orders(ids,config)
        self.assertEqual(set(orders),{str(s) for s in SEEDS})
        for seed,arms in orders.items():
            self.assertEqual(len({tuple(s for b in o for s,k in b) for o in arms.values()}),1)
            self.assertEqual({k for b in arms['A'] for s,k in b},{'direct'});self.assertEqual({k for b in arms['B'] for s,k in b},{'facts'})
            order=arms['A'];self.assertEqual(len(order),160)
            for e in range(40):self.assertEqual(Counter(s for b in order[4*e:4*e+4] for s,k in b),Counter({s:1 for s in ids}))
        self.assertNotEqual(orders['713']['A'],orders['714']['A'])


class Rules(unittest.TestCase):
    t=configuration()['thresholds']

    def test_run_classes_at_their_boundaries(self):
        self.assertEqual(classify_run(1.,.099,self.t),'fits')
        self.assertEqual(classify_run(1.,.1,self.t),'partial');self.assertEqual(classify_run(15/16,.01,self.t),'partial')
        self.assertEqual(classify_run(11/16,.59,self.t),'partial')
        self.assertEqual(classify_run(10/16,.01,self.t),'does_not_fit');self.assertEqual(classify_run(1.,.6,self.t),'does_not_fit')

    def test_arm_and_training_set_classes(self):
        self.assertEqual(arm_class(['fits','fits','partial']),'fits');self.assertEqual(arm_class(['fits','partial','does_not_fit']),'mixed')
        with self.assertRaises(ValueError):arm_class(['fits','fits'])
        self.assertEqual(train_class([.5,.5,.6],self.t),'at_chance');self.assertEqual(train_class([.5,.9,.95],self.t),'fits')
        self.assertEqual(train_class([.5,.61,.89],self.t),'partial')

    def test_decision_table(self):
        self.assertEqual(decision('does_not_fit','fits','at_chance'),'implementation_suspect')
        self.assertEqual(decision('does_not_fit','does_not_fit','at_chance'),'shared_pipeline_suspect')
        self.assertEqual(decision('fits','fits','at_chance'),'optimization_stall')
        self.assertEqual(decision('fits','fits','fits'),'fits_training_data')
        for a16,b16,a512 in (('fits','fits','partial'),('mixed','fits','at_chance'),('partial','fits','at_chance'),('does_not_fit','mixed','at_chance')):
            self.assertEqual(decision(a16,b16,a512),'inconclusive')

    def test_last_epoch_loss_uses_the_arm_fit_loss(self):
        config=configuration();curve=[dict(direct_verdict=1.,facts_key_values=2.) for _ in range(156)]
        curve+=[dict(direct_verdict=.1*i,facts_key_values=.01*i) for i in range(4)]
        self.assertAlmostEqual(last_epoch_loss(curve,'A',config),.15);self.assertAlmostEqual(last_epoch_loss(curve,'B',config),.015)
        with self.assertRaises(ValueError):last_epoch_loss(curve[1:],'A',config)


class NoTestRows(unittest.TestCase):
    def test_fit_run_never_names_or_loads_the_test_split(self):
        tree=ast.parse((ROOT/'verifier_training/fit_run.py').read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and getattr(n.func,'id',None)=='validate']
        self.assertEqual(len(calls),1);self.assertEqual([(k.arg,k.value.value) for k in calls[0].keywords if k.arg=='include_test'],[('include_test',False)])
        literals={n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
        self.assertFalse({'test','TEST_FREEZE.json','evaluate_test'}&literals)
        self.assertFalse([s for s in literals if s.startswith('test/') or 'selected_adapter' in s])
        names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}|{n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)}
        self.assertNotIn('evaluate_test',names)


if __name__=='__main__':unittest.main()
