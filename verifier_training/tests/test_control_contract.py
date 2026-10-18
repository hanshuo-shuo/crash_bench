import ast
import copy
from collections import Counter
from pathlib import Path
import unittest
from verifier_training.fit_contract import configuration as step1_configuration,overfit_orders
from verifier_training.control_contract import (configuration,ARMS,KIND,PHASE2_FILES,STEP1_FILES,text_bit,image_bit,label,answer,
    relabel,check_labels,orders,pathway_reading,ap_reading)

ROOT=Path(__file__).resolve().parents[2]


def unit(uid,mode,side):
    """One four-row unit built like V4: box a separates target and goal; box b holds both (BOTH) or neither (NEITHER)."""
    rows=[]
    for asg in (0,1):
        colours={'a':'orange','b':'purple'} if asg==0 else {'a':'purple','b':'orange'}
        for boundary in 'ab':
            colour=colours[boundary]
            if boundary=='a':target,goal=('yes','no') if mode=='EXIT' else ('no','yes')
            else:target=goal='yes' if side=='BOTH' else 'no'
            facts=dict(protected=colour,target_inside=target,goal_inside=goal,separated='yes' if target!=goal else 'no')
            rows.append(dict(sample_id=f'{uid}_{asg}{boundary}',split='train',
                metadata=dict(unit_id=uid,appearance_assignment=colours,construction_cell=f'{mode}__{side}',ancestry_component='group_'+uid),
                evidence=dict(assignment=asg,protected_boundary=boundary),
                student=dict(images=[dict(sha256=f'{uid}{asg}-1'),dict(sha256=f'{uid}{asg}-2')],task='Task '+uid,
                    constraints=f'Only the {colour} boundary is protected.'),
                target=dict(verdict='feasible' if target==goal else 'infeasible',facts=facts)))
    return rows


def sixteen():
    return [r for i,(m,s) in enumerate([('EXIT','BOTH'),('EXIT','NEITHER'),('ENTRY','BOTH'),('ENTRY','NEITHER')]) for r in unit(f'u{i}',m,s)]


class Labels(unittest.TestCase):
    def test_verdict_is_the_xor_of_a_text_bit_and_an_image_bit(self):
        rows=sixteen();report=check_labels(rows)
        self.assertEqual(report['balance'],{'t':{'0':8,'1':8},'s':{'0':8,'1':8}})
        self.assertEqual((report['image_groups'],report['text_groups']),(8,8))
        for r in rows:
            self.assertEqual(text_bit(r),int(r['target']['facts']['protected']=='orange'))
            self.assertEqual(image_bit(r),int(r['evidence']['assignment']==0))
            self.assertEqual(label(r,'AP'),int(r['target']['verdict']=='feasible'))
            self.assertEqual(label(r,'AP'),label(r,'s')^label(r,'t'))

    def test_targets_and_relabelled_truth(self):
        r=sixteen()[0]
        self.assertEqual(answer(r,'t'),'Verdict: 1');self.assertEqual(answer(r,'s'),'Verdict: 1')
        self.assertEqual(answer(r,'AP'),'Protected: orange\nVerdict: 0')
        before=copy.deepcopy(r);t=relabel(r,'t')
        self.assertEqual(r,before);self.assertEqual(t['target']['verdict'],'feasible');self.assertEqual(t['target']['control'],'t')
        self.assertEqual(t['student'],r['student']);self.assertEqual(t['target']['facts'],r['target']['facts'])
        self.assertEqual(relabel(r,'AP')['target']['verdict'],r['target']['verdict'])
        with self.assertRaises(ValueError):label(r,'A')

    def test_inconsistent_facts_and_mixed_inputs_are_refused(self):
        rows=sixteen();rows[0]['target']['facts']['separated']='no'
        with self.assertRaises(ValueError):check_labels(rows)
        rows=sixteen();rows[1]['metadata']=dict(rows[1]['metadata'],appearance_assignment={'a':'purple','b':'orange'})
        with self.assertRaises(ValueError):check_labels(rows)
        rows=sixteen()
        for r in rows[:4]:r['student']=dict(r['student'],constraints='Same text for every rule.')
        with self.assertRaises(ValueError):check_labels(rows)
        rows=sixteen()
        for r in rows[:4]:r['student']=dict(r['student'],images=[dict(sha256='shared-1'),dict(sha256='shared-2')])
        with self.assertRaises(ValueError):check_labels(rows)


class Schedule(unittest.TestCase):
    def test_every_arm_follows_the_step1_sample_order(self):
        config=configuration();ids=sorted(r['sample_id'] for r in sixteen());plan=orders(ids,config)
        step1=overfit_orders(ids,step1_configuration())
        self.assertEqual(set(plan),{'713','714','715'})
        for seed,arms in plan.items():
            self.assertEqual(set(arms),set(ARMS))
            for arm,batches in arms.items():
                self.assertEqual([[s for s,k in b] for b in batches],[[s for s,k in b] for b in step1[seed]['A']])
                self.assertEqual({k for b in batches for s,k in b},{KIND[arm]})
                for e in range(40):self.assertEqual(Counter(s for b in batches[4*e:4*e+4] for s,k in b),Counter({s:1 for s in ids}))


class Configuration(unittest.TestCase):
    def test_everything_but_the_target_is_the_step1_setting(self):
        config=configuration();step1=step1_configuration()
        self.assertEqual(config['phase2_configuration'],step1['phase2_configuration']);self.assertEqual(config['thresholds'],step1['thresholds'])
        for key in ('rows','epochs','optimizer_updates','evaluation_updates','effective_batch_size','seeds','units_per_cell'):
            self.assertEqual(config['overfit'][key],step1['overfit'][key])
        self.assertEqual(config['overfit']['arms'],list(ARMS));self.assertEqual(set(config['controls']),set(ARMS))
        self.assertEqual({a:c['kind'] for a,c in config['controls'].items()},KIND)
        self.assertEqual({a:c['loss'] for a,c in config['controls'].items()},{'t':'direct_verdict','s':'direct_verdict','AP':'facts_verdict'})
        self.assertEqual(config['controls']['AP']['key_fields'],['protected'])
        self.assertTrue(set(config['rules']['pathway_readings'])>={pathway_reading(a,b) for a in ('fits','does_not_fit','partial','mixed')
            for b in ('fits','does_not_fit','partial','mixed')})
        self.assertTrue(set(config['rules']['ap_readings'])>={ap_reading(a) for a in ('fits','does_not_fit','partial','mixed')})
        for files in (PHASE2_FILES,STEP1_FILES):self.assertFalse([f for f in files if f.startswith('test') or 'TEST_FREEZE' in f])

    def test_readings(self):
        self.assertEqual(pathway_reading('fits','fits'),'composition_bottleneck')
        self.assertEqual(pathway_reading('fits','does_not_fit'),'image_relation_bottleneck')
        self.assertEqual(pathway_reading('does_not_fit','fits'),'text_pathway_problem')
        self.assertEqual(pathway_reading('does_not_fit','does_not_fit'),'direct_path_problem')
        self.assertEqual(pathway_reading('fits','partial'),'inconclusive');self.assertEqual(pathway_reading('mixed','fits'),'inconclusive')
        self.assertEqual(ap_reading('fits'),'restatement_restores_fit');self.assertEqual(ap_reading('does_not_fit'),'restatement_insufficient')
        self.assertEqual(ap_reading('partial'),'inconclusive')


class NoTestRows(unittest.TestCase):
    def test_control_run_never_names_or_loads_the_test_split(self):
        tree=ast.parse((ROOT/'verifier_training/control_run.py').read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and getattr(n.func,'id',None)=='validate']
        self.assertEqual(len(calls),1);self.assertEqual([(k.arg,k.value.value) for k in calls[0].keywords if k.arg=='include_test'],[('include_test',False)])
        literals={n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
        self.assertFalse({'test','TEST_FREEZE.json','evaluate_test'}&literals)
        self.assertFalse([s for s in literals if s.startswith('test/') or 'selected_adapter' in s])


if __name__=='__main__':unittest.main()
