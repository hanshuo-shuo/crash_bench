import ast
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from verifier_training.common import write
from verifier_training.fd1000_contract import (validate,configuration,development_acceptance,check_development_row,facts_text,
    rule_verdict,assistant_target,readings,diagnostic_plan,schedules,check_schedule,target_kinds,zero_shot_messages,
    probe_truth,FORCED_PREFIX,DEVELOPMENT_FILES)

ROOT=Path(__file__).resolve().parents[2]


def facts_row(protected='a',target='no',goal='yes'):
    color={'a':'orange','b':'purple'}[protected];feasible=rule_verdict(target,goal)
    facts=dict(protected=color,target_inside=target,goal_inside=goal,separated='no' if feasible else 'yes')
    return dict(acceptance_mode='development',sample_id='s',split='validation',
        student=dict(task='Task.',constraints=f'Boxes. Only the {color} boundary is protected; the other may be crossed.'),
        metadata=dict(appearance_assignment={'a':'orange','b':'purple'}),evidence=dict(protected_boundary=protected),
        target=dict(verdict='feasible' if feasible else 'infeasible',facts=facts,rationale=facts_text(facts)))


class ScientificDataGate(unittest.TestCase):
    def test_engineering_or_unreviewed_data_cannot_start_training(self):
        for gate in ({'passed':True,'engineering_supervision_only':True},
                     {'passed':True,'scientific_training_allowed':False},
                     {'passed':False,'scientific_training_allowed':True}):
            with tempfile.TemporaryDirectory() as name:
                root=Path(name);write(root/'ACCEPTANCE.json',gate)
                with self.assertRaisesRegex(ValueError,'has not authorized'):validate(root)

    def test_changed_dataset_fails_before_any_model_load(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);(root/'dataset.jsonl').write_text('tampered')
            write(root/'ACCEPTANCE.json',dict(passed=True,scientific_training_allowed=True,dataset_sha256='not-the-file'))
            with self.assertRaisesRegex(ValueError,'evidence changed'):validate(root)

    def test_primary_generation_has_no_gold_rationale_prefix(self):
        for schedule in ('main','trial'):
            config=configuration(schedule)
            self.assertIn('no gold rationale prefix',config['evaluation'])
            self.assertFalse(config['hyperparameter_search'])
            self.assertTrue(config['inference_inputs_identical'])
            self.assertEqual(config['image_pixels'],448*448)
            self.assertEqual(config['max_length'],2048)
            self.assertEqual(config['repetition_penalty'],1.)
            self.assertEqual(config['optimizer_updates'],config['epochs']*512//config['effective_batch_size'])
            self.assertEqual(config['evaluation_updates'][-1],config['optimizer_updates'])


class DevelopmentGate(unittest.TestCase):
    def acceptance(self,root,**changes):
        for name in DEVELOPMENT_FILES:(root/name).write_text(name)
        write(root/'SHORTCUT_AUDIT.json',dict(passed=False,locked_test_scored=False))
        from verifier_training.common import sha
        value=dict(passed=True,acceptance_mode='development',development_training_allowed=True,scientific_training_allowed=False,
            user_decision=dict(date='2026-10-16'),shortcut_gate=dict(passed=False,reported_as_baseline=True,locked_test_scored=False),
            **{key:sha(root/name) for name,key in DEVELOPMENT_FILES.items()})
        value.update(changes);write(root/'DEV_ACCEPTANCE.json',value);return value

    def test_development_mode_requires_disclosed_failed_gate_and_hashes(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);self.acceptance(root);development_acceptance(root)
            (root/'SHORTCUT_FEATURES.npz').write_text('changed')
            with self.assertRaisesRegex(ValueError,'Development evidence changed'):development_acceptance(root)
        for change in (dict(scientific_training_allowed=True),dict(shortcut_gate=dict(passed=True,reported_as_baseline=True,locked_test_scored=False)),
                       dict(shortcut_gate=dict(passed=False,reported_as_baseline=True,locked_test_scored=True)),dict(acceptance_mode='scientific')):
            with tempfile.TemporaryDirectory() as name:
                root=Path(name);self.acceptance(root,**change)
                with self.assertRaises(ValueError):development_acceptance(root)

    def test_development_rows_must_carry_consistent_facts(self):
        check_development_row(facts_row('a','no','yes'));check_development_row(facts_row('b','yes','yes'))
        bad=facts_row('a','no','yes');bad['target']['facts']['separated']='no'
        with self.assertRaisesRegex(ValueError,'imply the label'):check_development_row(bad)
        bad=facts_row('a','no','yes');bad['target']['facts']['protected']='purple'
        with self.assertRaisesRegex(ValueError,'Protected color'):check_development_row(bad)
        bad=facts_row('a','no','yes');bad['student']['constraints']=bad['student']['constraints'].replace('orange','purple')
        with self.assertRaisesRegex(ValueError,'Rule text'):check_development_row(bad)

    def test_development_mode_is_the_only_path_for_rejected_v4(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);self.acceptance(root)
            with self.assertRaises(FileNotFoundError):validate(root)  # scientific mode needs ACCEPTANCE.json


class TargetsAndReadings(unittest.TestCase):
    def test_targets_and_forced_prefixes(self):
        row=facts_row('a','no','yes')
        self.assertEqual(assistant_target(row,'direct'),'Verdict: 0')
        self.assertEqual(assistant_target(row,'facts'),'Protected: orange\nTarget inside orange: no\nGoal inside orange: yes\nSeparated: yes\nVerdict: 0')
        for kind,prefix in FORCED_PREFIX.items():self.assertTrue(assistant_target(row,kind).startswith(prefix))
        self.assertEqual(set(readings('C')),{'direct_forced','facts_forced'})
        self.assertEqual(readings('B')['facts_free'],('facts',None))
        self.assertEqual(diagnostic_plan('A'),[])
        self.assertEqual(rule_verdict('yes','yes'),1);self.assertEqual(rule_verdict('no','yes'),0)

    def test_c_schedule_shares_sample_order_and_balances_targets(self):
        ids=[f's{i:03d}' for i in range(512)];config=configuration('main');orders=schedules(ids,config)
        self.assertEqual(set(orders),{'713','714','715'})
        for seed,arms in orders.items():
            sequences={arm:[s for b in o for s,k in b] for arm,o in arms.items()}
            self.assertEqual(len({tuple(v) for v in sequences.values()}),1)
            for arm,order in arms.items():check_schedule(order,ids,arm,config)
            kinds=Counter((s,k) for b in arms['C'] for s,k in b)
            self.assertEqual(Counter(k for (s,k) in kinds.elements()),Counter(direct=768,facts=768))
            self.assertTrue(all(kinds[(s,'direct')]>=1 and kinds[(s,'facts')]>=1 for s in ids))
            self.assertEqual({k for b in arms['A'] for s,k in b},{'direct'});self.assertEqual({k for b in arms['B'] for s,k in b},{'facts'})
        self.assertNotEqual(orders['713']['A'],orders['714']['A'])
        e0=target_kinds(ids,713,0);e1=target_kinds(ids,713,1)
        self.assertTrue(all(e0[s]!=e1[s] for s in ids))

    def test_trial_schedule_is_one_epoch_of_a_and_b(self):
        config=configuration('trial');orders=schedules([f's{i:03d}' for i in range(512)],config)
        self.assertEqual(set(orders['713']),{'A','B'});self.assertEqual(len(orders['713']['A']),128)

    def test_zero_shot_prompts_share_inputs_and_hide_metadata(self):
        row=facts_row('a','no','yes')
        texts={m:zero_shot_messages(row,m)[1]['content'][-1]['text'] for m in ('direct','facts','oracle')}
        for text in texts.values():
            self.assertIn('Task: Task.',text);self.assertNotIn('unit',text)
        self.assertIn('outside the protected orange box',texts['oracle']);self.assertIn('Separated:',texts['facts'])
        self.assertNotIn('inside the protected',texts['direct'])
        other=facts_row('b','yes','yes');bits=probe_truth([row,other])
        self.assertEqual(bits,{('target','orange'):False,('goal','orange'):True,('target','purple'):True,('goal','purple'):True})


class Preregistration(unittest.TestCase):
    def test_prereg_matches_the_frozen_configuration(self):
        prereg=json.loads((ROOT/'verifier_training/configs/v4_dev_prereg.json').read_text());config=configuration('main')
        self.assertEqual(prereg['training_configuration'],config['version'])
        self.assertEqual(prereg['seeds'],config['seeds']);self.assertEqual(prereg['checkpoints'],config['evaluation_updates'])
        self.assertEqual(prereg['optimizer_updates'],config['optimizer_updates'])
        self.assertEqual(prereg['sampling']['k_values'],config['sampling']['k_values'])
        for arm in config['arms']:
            names=list(readings(arm))+[p[0] for p in diagnostic_plan(arm) if p[3]=='sampled']
            self.assertEqual(prereg['arms'][arm]['readings'],names)
        self.assertEqual(prereg['success']['development_milestone']['test_row_accuracy'],.85)

    def test_train_and_zero_shot_never_load_test_rows(self):
        tree=ast.parse((ROOT/'verifier_training/fd1000_run.py').read_text())
        functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        for name in ('train','zero_shot'):
            calls=[n for n in ast.walk(functions[name]) if isinstance(n,ast.Call) and getattr(n.func,'id',None)=='context']
            self.assertEqual(len(calls),1)
            self.assertEqual([(k.arg,k.value.value) for k in calls[0].keywords],[('include_test',False)])
            literals={n.value for n in ast.walk(functions[name]) if isinstance(n,ast.Constant)}
            self.assertFalse({'test','TEST_FREEZE.json'}&literals,name+' must not name the test split')
        test_call=[n for n in ast.walk(functions['evaluate_test']) if isinstance(n,ast.Call) and getattr(n.func,'id',None)=='context'][0]
        self.assertEqual([(k.arg,k.value.value) for k in test_call.keywords],[('include_test',True)])


if __name__=='__main__':unittest.main()
