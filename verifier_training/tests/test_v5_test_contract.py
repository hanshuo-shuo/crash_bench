import ast
from pathlib import Path
import unittest
from verifier_training.common import read,sha
from verifier_training.v5_contract import validate,DATASET,TEST_SPLITS
from verifier_training.v5_seeds_contract import configuration as seeds_configuration,THRESHOLDS
from verifier_training.v5_test_contract import (configuration,split_target,target_root,evidence_files,freeze,reload_subset,reload_agreement,
    test_reading,TARGETS,LANES,SMOKE_SPLITS,RELOAD_ROWS,RUN_FILES)

ROOT=Path(__file__).resolve().parents[2]
RUNS=Path('/projects/p33100/siosio/crashbench_safelibero/verifier_training/runs')
ROOTS=dict(s3=RUNS/'20261025T055227Z_v5-s3_bdcf277164b6',s4=RUNS/'20261025T180525Z_v5-s4_20913c71c3f3',
    seeds=RUNS/'20261026T001735Z_v5-seeds_ee20b2033465')
HAVE_RUNS=all((r/'RECEIPT.json').exists() for r in ROOTS.values()) and (ROOTS['seeds']/'SEEDS_READING.json').exists()


def rec(sample,text,decision,arm='A',digest=None):
    r=dict(sample_id=sample,generated=text,student_input_sha256=digest or sample)
    r['predicted' if arm=='A' else 'program']=decision;return r


class Settings(unittest.TestCase):
    def test_configuration(self):
        c=configuration()
        self.assertEqual(c['seeds_configuration'],seeds_configuration());self.assertEqual(c['thresholds'],THRESHOLDS)
        self.assertEqual(c['targets'],['A_seed713','A_seed714','A_seed715','P_seed713','P_seed714','P_seed715'])
        self.assertEqual((c['evaluated_splits'],c['primary_split'],c['evaluations_per_adapter']),(['test','test_natural'],'test',1))
        self.assertEqual(set(c['rules']),{'test_confirms','test_fails','unclear','descriptive'})
        self.assertEqual((c['reload_check']['rows'],c['reload_check']['split']),(RELOAD_ROWS,'validation'))
        self.assertFalse(c['hyperparameter_search']);self.assertEqual(c['api_calls'],0);self.assertEqual(SMOKE_SPLITS,('validation_natural',))

    def test_lanes_cover_every_target_once_on_two_gpus(self):
        self.assertEqual(len(LANES),2);flat=[t for lane in LANES for t in lane];self.assertEqual(sorted(flat),sorted(TARGETS))

    def test_targets_live_in_their_runs(self):
        roots=dict(s3='/r/s3',s4='/r/s4',seeds='/r/seeds')
        self.assertEqual(split_target('P_seed715'),('P',715))
        self.assertEqual([str(target_root(t,roots)) for t in TARGETS],['/r/s3','/r/seeds','/r/seeds','/r/s4','/r/seeds','/r/seeds'])
        files=evidence_files(roots)
        self.assertIn('A_seed713/selected_adapter/adapter_model.safetensors',files['s3']);self.assertIn('P_seed713/validation.jsonl',files['s4'])
        self.assertEqual(sum(f.endswith('RESULT.json') and '_seed' in f for v in files.values() for f in v),6)
        self.assertEqual(sum(len(v) for v in files.values()),5+4+3+6*len(RUN_FILES))
        with self.assertRaisesRegex(ValueError,'Unknown target'):split_target('B_seed713')


class Reload(unittest.TestCase):
    def test_subset_is_fixed_and_order_free(self):
        ids=['v%04d'%i for i in range(200)];chosen=reload_subset(ids)
        self.assertEqual(len(chosen),RELOAD_ROWS);self.assertEqual(chosen,reload_subset(list(reversed(ids))));self.assertLessEqual(set(chosen),set(ids))
        self.assertEqual(chosen,sorted(chosen))
        with self.assertRaisesRegex(ValueError,'Too few'):reload_subset(ids[:5])

    def test_agreement(self):
        old=[rec('a','Verdict: 1',1),rec('b','Verdict: 0',0)]
        self.assertTrue(reload_agreement([dict(r) for r in old],old,'A')['passed'])
        flipped=[rec('a','Verdict: 0',0),rec('b','Verdict: 0',0)];result=reload_agreement(flipped,old,'A')
        self.assertFalse(result['passed']);self.assertEqual((result['decisions_equal'],result['texts_equal']),(1,1))
        self.assertEqual(result['mismatches'][0]['sample_id'],'a')
        p_old=[rec('a','Line: x\nVerdict: 1',0,'P')];p_new=[rec('a','Line: y\nVerdict: 1',0,'P')]
        self.assertFalse(reload_agreement(p_new,p_old,'P')['passed']);self.assertTrue(reload_agreement(p_new,p_old,'P',allowed=1)['passed'])
        self.assertFalse(reload_agreement([rec('a','Verdict: 1',1,digest='other')],old,'A')['passed'])
        self.assertFalse(reload_agreement([],old,'A')['passed'])
        with self.assertRaisesRegex(ValueError,'not recorded'):reload_agreement([rec('z','Verdict: 1',1)],old,'A')


class Reading(unittest.TestCase):
    def test_preregistered_reading(self):
        p={'713':.94,'714':.9,'715':.7};d={'713':.44,'714':.4,'715':.2}
        self.assertEqual(test_reading(p,d,[.3,.4]),'test_confirms');self.assertEqual(test_reading(p,d,[0.,.4]),'unclear')
        self.assertEqual(test_reading(dict(p,**{'715':.64}),d,[.3,.4]),'unclear')
        self.assertEqual(test_reading(p,dict(d,**{'715':.09}),[.3,.4]),'unclear')
        self.assertEqual(test_reading(dict(p,**{'714':.60}),d,[.3,.4]),'test_fails')
        self.assertEqual(test_reading(dict(p,**{'714':.65}),d,[.3,.4]),'test_confirms')
        with self.assertRaisesRegex(ValueError,'three seeds'):test_reading({'713':.9},{'713':.4},[.3,.4])


@unittest.skipUnless(HAVE_RUNS,'S3, S4 and seed runs not mounted')
class Frozen(unittest.TestCase):
    def test_freeze_takes_the_selected_adapters_as_recorded(self):
        f=freeze(ROOTS);self.assertEqual(sorted(f),sorted(TARGETS))
        self.assertEqual({t:f[t]['selected_step'] for t in TARGETS},
            dict(A_seed713=448,A_seed714=150,A_seed715=448,P_seed713=448,P_seed714=448,P_seed715=300))
        seeds=read(ROOTS['seeds']/'SEEDS_READING.json')['per_seed']
        for t in TARGETS:
            arm,seed=split_target(t);self.assertEqual(f[t]['adapter_sha256'],seeds[str(seed)][arm]['adapter_sha256'])
            self.assertEqual(f[t]['result_sha256'],seeds[str(seed)][arm]['result_sha256'])

    @unittest.skipUnless((DATASET/'dataset.jsonl').exists(),'S2 dataset not mounted')
    def test_reload_rows_are_recorded_for_every_adapter(self):
        chosen=set(reload_subset([r['sample_id'] for r in validate() if r['split']=='validation']))
        for t in TARGETS:
            path=target_root(t,ROOTS)/t/'validation.jsonl';ids={x.split('"sample_id": "',1)[1].split('"',1)[0] for x in path.read_text().splitlines()}
            self.assertLessEqual(chosen,ids,t)


class Isolation(unittest.TestCase):
    def calls(self,path):
        tree=ast.parse((ROOT/path).read_text());found=[]
        for fn in [n for n in tree.body if isinstance(n,ast.FunctionDef)]:
            found+=[(fn.name,c) for c in ast.walk(fn) if isinstance(c,ast.Call) and getattr(c.func,'id',None)=='validate']
        return found

    def test_only_the_test_context_can_load_test_rows(self):
        for path in sorted((ROOT/'verifier_training').glob('v5_*.py'))+[ROOT/'verifier_training/launch_v5.py']:
            name=str(path.relative_to(ROOT))
            for fn,c in self.calls(name):
                keywords={k.arg:k.value for k in c.keywords}
                if name=='verifier_training/v5_test_run.py':
                    self.assertEqual(fn,'context');test=keywords['include_test']
                    self.assertIsInstance(test,ast.Compare);self.assertEqual(ast.unparse(test),'splits == TEST_SPLITS')
                else:
                    self.assertIn('include_test',keywords,(name,fn));self.assertIs(keywords['include_test'].value,False,(name,fn))
        self.assertEqual(len(self.calls('verifier_training/v5_test_run.py')),1);self.assertEqual(TEST_SPLITS,('test','test_natural'))

    def test_each_adapter_evaluates_once(self):
        tree=ast.parse((ROOT/'verifier_training/v5_test_run.py').read_text())
        evaluate=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate'][0]
        mkdirs=[c for c in ast.walk(evaluate) if isinstance(c,ast.Call) and getattr(c.func,'attr',None)=='mkdir']
        self.assertIn('out.mkdir()',[ast.unparse(c) for c in mkdirs]);self.assertIn("require(check['passed'], 'Reloaded adapter does not reproduce its validation outputs')",
            [ast.unparse(n) for n in ast.walk(evaluate) if isinstance(n,ast.Call)])

    def test_launcher_checks_the_freeze_and_keeps_two_lanes(self):
        tree=ast.parse((ROOT/'verifier_training/launch_v5.py').read_text());fns={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        literals={n.value for n in ast.walk(fns['test']) if isinstance(n,ast.Constant)}
        called={n.func.id for n in ast.walk(fns['test']) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
        self.assertIn('状态：已冻结',literals);self.assertIn('stable',literals)
        self.assertLessEqual({'published_commit','completed','active_gpus','test_root','test_jobs'},called)


if __name__=='__main__':unittest.main()
