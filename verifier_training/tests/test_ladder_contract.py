import ast
from pathlib import Path
import unittest
from verifier_training.fd1000_contract import configuration as phase2_configuration
from verifier_training.fit_contract import configuration as step1_configuration,overfit_orders
from verifier_training.control_contract import configuration as controls_configuration
from verifier_training.ladder_contract import (configuration,answer,fit16_orders,comparison_reading,fit16_reading,ARMS16,ARMS512,RUNGS,
    COMPARISONS,PHASE2_FILES,STEP1_FILES,CONTROLS_FILES,FILLER)
from verifier_training.tests.test_control_contract import sixteen

ROOT=Path(__file__).resolve().parents[2]


class Targets(unittest.TestCase):
    def test_rungs_share_the_template_and_end_with_the_original_verdict(self):
        for r in sixteen():
            digit=int(r['target']['verdict']=='feasible')
            self.assertEqual(answer(r,'AF'),f'Protected: {FILLER}\nVerdict: {digit}')
            self.assertEqual(answer(r,'AP'),f'Protected: {r["target"]["facts"]["protected"]}\nVerdict: {digit}')
        self.assertEqual(len({answer(r,'AF').split('\n')[0] for r in sixteen()}),1)
        with self.assertRaises(ValueError):answer(sixteen()[0],'B')


class Schedules(unittest.TestCase):
    def test_fit16_follows_the_step1_sample_order(self):
        config=configuration();ids=sorted(r['sample_id'] for r in sixteen());plan=fit16_orders(ids,config)
        step1=overfit_orders(ids,step1_configuration())
        for seed,arms in plan.items():
            self.assertEqual(set(arms),set(ARMS16))
            for batches in arms.values():
                self.assertEqual([[s for s,k in b] for b in batches],[[s for s,k in b] for b in step1[seed]['A']])
                self.assertEqual({k for b in batches for s,k in b},{'facts'})

    def test_configuration_reuses_the_earlier_settings(self):
        config=configuration();base=phase2_configuration('main');controls=controls_configuration()
        self.assertEqual(config['phase2_configuration'],base);self.assertEqual(config['thresholds'],controls['thresholds'])
        for key in ('rows','epochs','optimizer_updates','evaluation_updates','effective_batch_size','seeds'):
            self.assertEqual(config['fit16'][key],controls['overfit'][key])
        t=config['train512']
        self.assertEqual((t['rows'],t['epochs'],t['optimizer_updates'],t['evaluation_updates'],t['effective_batch_size'],t['seeds']),
            (512,base['epochs'],base['optimizer_updates'],base['evaluation_updates'],base['effective_batch_size'],base['seeds']))
        self.assertEqual((config['fit16']['arms'],t['arms']),(list(ARMS16),list(ARMS512)))
        self.assertEqual(set(config['rungs']),set(RUNGS));self.assertEqual(config['primary_update'],base['evaluation_updates'][-1])
        self.assertEqual({a:config['rungs'][a]['key_fields'] for a in ARMS512},{'AF':[],'AP':['protected']})
        self.assertEqual(config['comparisons'],[list(c) for c in COMPARISONS])
        for treatment,control in COMPARISONS:
            self.assertEqual(set(config['rules']['comparisons'][f'{treatment}-{control}']),{'treatment_higher','treatment_lower','no_detected_difference'})
        self.assertEqual(set(config['rules']['fit16']['readings']),{'extra_tokens_restore_fit','colour_needed_for_fit','inconclusive'})
        for files in (PHASE2_FILES,STEP1_FILES,CONTROLS_FILES):
            self.assertFalse([f for f in files if f.startswith('test') or 'TEST_FREEZE' in f or 'selected_adapter' in f])


class Readings(unittest.TestCase):
    def test_interval_signs_and_fit_reading(self):
        self.assertEqual(comparison_reading([.01,.2]),'treatment_higher');self.assertEqual(comparison_reading([-.2,-.01]),'treatment_lower')
        self.assertEqual(comparison_reading([-.1,.1]),'no_detected_difference');self.assertEqual(comparison_reading([0.,.1]),'no_detected_difference')
        self.assertEqual(fit16_reading('fits'),'extra_tokens_restore_fit');self.assertEqual(fit16_reading('does_not_fit'),'colour_needed_for_fit')
        self.assertEqual(fit16_reading('mixed'),'inconclusive')


class NoTestRows(unittest.TestCase):
    def test_ladder_run_never_names_or_loads_the_test_split(self):
        tree=ast.parse((ROOT/'verifier_training/ladder_run.py').read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and getattr(n.func,'id',None)=='validate']
        self.assertEqual(len(calls),1);self.assertEqual([(k.arg,k.value.value) for k in calls[0].keywords if k.arg=='include_test'],[('include_test',False)])
        literals={n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
        self.assertFalse({'test','TEST_FREEZE.json','evaluate_test'}&literals)
        self.assertFalse([s for s in literals if s.startswith('test/') or 'selected_adapter' in s])


if __name__=='__main__':unittest.main()
