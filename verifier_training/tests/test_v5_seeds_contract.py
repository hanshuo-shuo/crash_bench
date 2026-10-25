import ast
import json
from collections import Counter
from pathlib import Path
import unittest
from verifier_training.common import read
from verifier_training.v5_contract import configuration as s3_configuration,schedule,validate,selection_subset,fit_subset,lookup_predictions,DATASET
from verifier_training.v5_pos_contract import configuration as s4_configuration
from verifier_training.v5_seeds_contract import (configuration,orders,check_orders,program_view,seed_average,seeds_reading,
    SEEDS,ALL_SEEDS,REFERENCE_SEED,THRESHOLDS,S3_FILES,S4_FILES)

ROOT=Path(__file__).resolve().parents[2]
RUNS=Path('/projects/p33100/siosio/crashbench_safelibero/verifier_training/runs')
S3=RUNS/'20261025T055227Z_v5-s3_bdcf277164b6'
S4=RUNS/'20261025T180525Z_v5-s4_20913c71c3f3'
HAVE_RUNS=(S3/'ORDERS.json').exists() and (S4/'S4_READING.json').exists()


def rec(sample,image,truth,predicted,source='s0',digest=None,**extra):
    return dict(sample_id=sample,image_id=image,source_id='x/'+source,truth=truth,predicted=predicted,student_input_sha256=digest or sample,
        forced_prefix=None,**extra)


class Settings(unittest.TestCase):
    def test_configuration_embeds_the_frozen_s3_and_s4_settings(self):
        c=configuration()
        self.assertEqual(c['s3_configuration'],s3_configuration());self.assertEqual(c['s4_configuration'],s4_configuration())
        self.assertEqual(c['s4_configuration']['s3_configuration'],c['s3_configuration'])
        self.assertEqual((c['arms'],c['seeds'],c['reference_seed'],c['all_seeds']),(['A','P'],[714,715],713,[713,714,715]))
        self.assertEqual((c['s3_configuration']['seeds'],c['s4_configuration']['seeds']),([713],[713]))
        self.assertEqual(c['thresholds'],dict(p_floor=.65,p_fail=.60,delta=.10));self.assertEqual(set(c['rules']),{'stable','unstable','unclear','per_seed'})
        self.assertEqual(c['aggregation']['bootstrap']['replicates'],1000);self.assertEqual(c['aggregation']['bootstrap']['seed'],1014)
        self.assertFalse(c['hyperparameter_search']);self.assertEqual(c['api_calls'],0)
        self.assertIn('A_seed713/validation.jsonl',S3_FILES);self.assertIn('P_seed713/validation.jsonl',S4_FILES)


class Orders(unittest.TestCase):
    def test_new_seeds_share_one_order_per_seed_and_cover_every_row(self):
        c=configuration();ids=['t%04d'%i for i in range(c['s3_configuration']['training_rows'])];plan=orders(list(reversed(ids)),c)
        self.assertEqual(list(plan),['714','715']);check_orders(plan,ids,c)
        flat=lambda o:[s for b in o for s,k in b]
        for seed,arms in plan.items():
            self.assertEqual(len(arms['A']),448);self.assertEqual(flat(arms['A']),flat(arms['P']))
            self.assertEqual({k for b in arms['A'] for s,k in b},{'direct'});self.assertEqual({k for b in arms['P'] for s,k in b},{'facts'})
            self.assertEqual(Counter(flat(arms['P'])),Counter({s:1 for s in ids}))
        old=schedule(ids,s3_configuration())['713']
        self.assertNotEqual(flat(plan['714']['A']),flat(old['A']));self.assertNotEqual(flat(plan['714']['A']),flat(plan['715']['A']))
        same=schedule(ids,dict(s3_configuration(),seeds=[714]))['714'];self.assertEqual(plan['714'],{'A':same['A'],'P':same['B']})

    def test_broken_orders_are_rejected(self):
        c=configuration();ids=['t%04d'%i for i in range(1792)];plan=orders(ids,c)
        swapped=json.loads(json.dumps(plan));swapped['714']['P']=plan['715']['P']
        with self.assertRaisesRegex(ValueError,'sample order'):check_orders(swapped,ids,c)
        repeated=json.loads(json.dumps(plan));repeated['715']=repeated['714']
        with self.assertRaisesRegex(ValueError,'share'):check_orders(repeated,ids,c)
        kinds=json.loads(json.dumps(plan));kinds['714']['P']=plan['714']['A']
        with self.assertRaisesRegex(ValueError,'kind'):check_orders(kinds,ids,c)

    @unittest.skipUnless(HAVE_RUNS,'S3 and S4 runs not mounted')
    def test_schedule_reproduces_the_s3_and_s4_orders_for_713(self):
        s3=read(S3/'ORDERS.json');ids=sorted(s for b in s3['713']['A'] for s,k in b)
        self.assertEqual(schedule(ids,s3_configuration()),s3);self.assertEqual(read(S4/'ORDERS.json'),{'713':{'P':s3['713']['B']}})


class Reading(unittest.TestCase):
    def test_preregistered_reading(self):
        p={'713':.94,'714':.9,'715':.7};d={'713':.44,'714':.4,'715':.2}
        self.assertEqual(seeds_reading(p,d,[.3,.4]),'stable');self.assertEqual(seeds_reading(p,d,[0.,.4]),'unclear')
        self.assertEqual(seeds_reading(dict(p,**{'715':.64}),d,[.3,.4]),'unclear')
        self.assertEqual(seeds_reading(p,dict(d,**{'715':.09}),[.3,.4]),'unclear')
        self.assertEqual(seeds_reading(dict(p,**{'714':.60}),d,[.3,.4]),'unstable');self.assertEqual(seeds_reading(dict(p,**{'714':.65}),d,[.3,.4]),'stable')
        with self.assertRaisesRegex(ValueError,'three seeds'):seeds_reading({'713':.9,'714':.9},{'713':.4,'714':.4},[.3,.4])
        self.assertEqual(THRESHOLDS,configuration()['thresholds']);self.assertEqual(ALL_SEEDS,(REFERENCE_SEED,)+SEEDS)


class SeedAverage(unittest.TestCase):
    def rows(self,predictions):
        return [rec('r%d'%i,'i%d'%(i//2),i%2,predictions[i],source='s%d'%(i//4)) for i in range(len(predictions))]

    def test_row_means_and_difference(self):
        truth=[0,1,0,1,0,1,0,1]
        a={713:self.rows([0,0,0,0,0,0,0,0]),714:self.rows([1,1,1,1,1,1,1,1]),715:self.rows([0,1,0,0,0,0,0,0])}
        p={713:self.rows(truth),714:self.rows(truth[:7]+[None]),715:self.rows([1-x for x in truth[:2]]+truth[2:])}
        m=seed_average(a,p)
        self.assertEqual((m['rows'],m['images'],m['sources'],m['seeds']),(8,4,2,[713,714,715]))
        self.assertEqual(m['A_by_seed'],{'713':.5,'714':.5,'715':.625});self.assertEqual(m['P_by_seed'],{'713':1.,'714':.875,'715':.75})
        self.assertAlmostEqual(m['A_mean'],(.5+.5+.625)/3);self.assertAlmostEqual(m['P_mean'],(1+.875+.75)/3)
        self.assertAlmostEqual(m['difference'],sum(m['difference_by_seed'].values())/3)
        self.assertEqual(m['rows_correct_every_seed'],dict(A=0,P=5));self.assertEqual(m['rows_wrong_every_seed'],dict(A=0,P=0))
        self.assertEqual(m['same_prediction_every_seed']['P'],5/8)
        low,high=m['image_bootstrap_95pct']['difference'];self.assertLessEqual(low,m['difference']);self.assertGreaterEqual(high,m['difference'])
        self.assertEqual(m,seed_average(a,p))

    def test_mismatched_rows_are_rejected(self):
        good=self.rows([0,1,0,1])
        with self.assertRaisesRegex(ValueError,'different rows'):seed_average({1:good,2:good},{1:good,2:good[:3]})
        other=[dict(r,student_input_sha256='x') if r['sample_id']=='r0' else r for r in good]
        with self.assertRaisesRegex(ValueError,'differ across'):seed_average({1:good,2:good},{1:good,2:other})
        forced=[dict(r,forced_prefix='Verdict: ') for r in good]
        with self.assertRaisesRegex(ValueError,'free generation'):seed_average({1:good,2:forced},{1:good,2:good})
        with self.assertRaisesRegex(ValueError,'same seeds'):seed_average({1:good,2:good},{1:good,3:good})
        with self.assertRaisesRegex(ValueError,'Repeated'):seed_average({1:good+good[:1],2:good},{1:good,2:good})

    def test_program_view(self):
        r=rec('a','i0',0,None,program=0,model=1,p_feasible=.3);v=program_view([r])[0]
        self.assertEqual((v['predicted'],v['p_feasible'],v['reading'],v['model']),(0,None,'program',1))

    @unittest.skipUnless(HAVE_RUNS,'S3 and S4 runs not mounted')
    def test_seed_713_alone_reproduces_the_s4_paired_difference(self):
        lines=lambda p:[json.loads(x) for x in p.read_text().splitlines()]
        a=lines(S3/'A_seed713/validation.jsonl');p=program_view(lines(S4/'P_seed713/validation.jsonl'))
        m=seed_average({713:a,714:a},{713:p,714:p});s4=read(S4/'S4_READING.json')['delta_P_program_minus_A']
        self.assertAlmostEqual(m['difference'],s4['difference'],places=12)
        self.assertEqual(m['image_bootstrap_95pct']['difference'],s4['image_bootstrap_95pct'])
        self.assertEqual(m['source_bootstrap_95pct_reference']['difference'],s4['source_bootstrap_95pct_reference'])


@unittest.skipUnless(HAVE_RUNS and (DATASET/'dataset.jsonl').exists(),'S2 dataset or S3 run not mounted')
class Reuse(unittest.TestCase):
    def test_s3_subsets_and_lookup_are_reproduced(self):
        rows=validate();split=lambda s:[r for r in rows if r['split']==s];c=s3_configuration()
        subsets=dict(selection=selection_subset(split('validation'),c['selection']['images']),training_fit=fit_subset(split('train'),c['training_fit']['rows']))
        self.assertEqual(subsets,read(S3/'SUBSETS.json'))
        lookup={s:lookup_predictions(split('train'),split(s)) for s in ('train','validation','validation_natural')}
        self.assertEqual(lookup,read(S3/'LOOKUP.json')['predictions'])


class Isolation(unittest.TestCase):
    def functions(self,path):
        tree=ast.parse((ROOT/path).read_text());return {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}

    def test_seed_stages_and_reused_training_never_load_test_rows(self):
        for path,names in (('verifier_training/v5_seeds_run.py',('context','preflight','gate','train','report','run_dirs')),
                           ('verifier_training/v5_run.py',('train','train_arm')),('verifier_training/v5_pos_run.py',('train','train_p'))):
            functions=self.functions(path)
            for name in names:
                calls=[n for n in ast.walk(functions[name]) if isinstance(n,ast.Call) and getattr(n.func,'id',None)=='validate']
                self.assertTrue(all([(k.arg,k.value.value) for k in c.keywords]==[('include_test',False)] for c in calls),name)
                literals={n.value for n in ast.walk(functions[name]) if isinstance(n,ast.Constant)}
                self.assertFalse({'test','test_natural','TEST_FREEZE.json'}&literals,name)
        context=self.functions('verifier_training/v5_seeds_run.py')['context']
        self.assertEqual(len([n for n in ast.walk(context) if isinstance(n,ast.Call) and getattr(n.func,'id',None)=='validate']),1)

    def test_launcher_checks_the_freeze_and_keeps_two_lanes(self):
        seeds=self.functions('verifier_training/launch_v5.py')['seeds']
        literals={n.value for n in ast.walk(seeds) if isinstance(n,ast.Constant)}
        called={n.func.id for n in ast.walk(seeds) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
        self.assertIn('状态：已冻结',literals);self.assertLessEqual({'published_commit','completed','active_gpus','submit','archive'},called)
        self.assertTrue((ROOT/'docs/generative_verifier/V5_SEEDS_PROTOCOL.md').exists())


if __name__=='__main__':unittest.main()
