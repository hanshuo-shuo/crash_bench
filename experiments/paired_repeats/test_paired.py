import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parents[1]/'scripts'))
import adapter
import protocol as p
from serve_seeded import SeededPolicy
from api_budget import Budget
from api_worker import fresh_fill, prior_spend
from openrouter_perception import export_request

def rows():
    result=[]
    for s,e,r,m,seed in p.schedule():
        result.append(dict(scenario=p.scenario_name(s),episode=e,repeat=r,method=m,seed=seed,phase='full',
            qpos_sha256='same',active_obstacle='moka_pot_obstacle_1',code_commit='fixed',
            modification_threshold=1e-6,modified_steps=1,collided=False,max_obstacle_l1_m=0.,success=True,
            exited=False,exit_reason='success',vlm_correct=True if m=='aegis' else None,filter_status='enabled'))
    return result

class ProtocolTests(unittest.TestCase):
    def test_matrix_pairing_and_unique_seeds(self):
        schedule=list(p.schedule())
        self.assertEqual(len(schedule),600)
        self.assertEqual(len({x[4] for x in schedule}),300)
        for i in range(0,600,2):
            a,b=schedule[i:i+2]
            self.assertEqual(a[:3],b[:3]);self.assertEqual(a[4],b[4]);self.assertNotEqual(a[3],b[3])

    def test_stability_white_intervention_and_threshold(self):
        data=rows()
        # State 0 nominal stable collision (4/5), AEGIS stable safety (5/5).
        for row in data:
            if row['scenario']==p.scenario_name(p.SCENARIOS[0]) and row['episode']==0 and row['method']=='nominal' and row['repeat']<4:
                row.update(collided=True,max_obstacle_l1_m=.005,success=True)
            # State 1 nominal 3 vs 2 => unstable, excluded from matrix.
            if row['scenario']==p.scenario_name(p.SCENARIOS[0]) and row['episode']==1 and row['method']=='nominal' and row['repeat']<2:
                row.update(success=False)
        p.validate_rows(data)
        groups=p.tables(data,.001)['groups']['合并']
        self.assertEqual(groups['table'],[[0,1,0],[0,58,0],[0,0,0]])
        self.assertEqual(groups['碰巧'],1);self.assertEqual(groups['白干预'],58)
        relaxed=p.tables(data,.01)['groups']['合并']
        self.assertEqual(relaxed['table'][1][1],59)

    def test_exact_strict_collision_boundary_and_exits(self):
        row=rows()[0];row.update(max_obstacle_l1_m=.001,success=False,exited=True)
        self.assertEqual(p.category(row),'安全但没完成')
        row['max_obstacle_l1_m']=.001000001
        self.assertEqual(p.category(row),'撞了')
        row.update(max_obstacle_l1_m=0,success=True)
        self.assertEqual(p.category(row),'安全完成')

    def test_refuse_partial_duplicate_state_seed_or_threshold_mismatch(self):
        original=rows()
        cases=[original[:-1],original+[original[0]]]
        for field,value in [('qpos_sha256','different'),('seed',0),('modification_threshold',None),('collided',True),('active_obstacle','other')]:
            data=copy.deepcopy(original);data[0][field]=value;cases.append(data)
        for data in cases:
            with self.assertRaises(ValueError):p.validate_rows(data)

    def test_complete_summary_has_four_primary_tables_and_sensitivity(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=p.summarize(rows(),tmp)
            self.assertEqual(len(result['thresholds']),2)
            self.assertEqual(result['thresholds'][0]['groups']['合并']['table'][1][1],60)
            self.assertEqual(len(result['thresholds'][0]['safe_success_differences']),60)
            self.assertEqual(len((Path(tmp)/'raw.csv').read_text().splitlines()),601)

class AdapterTests(unittest.TestCase):
    def test_compiles_and_preserves_full_qp_and_bug(self):
        source=(HERE.parents[1]/'tests/fixtures/main_aegis_upstream.txt').read_text()
        for method in ['nominal','aegis']:
            patched=adapter.adapt(source,method);ast.parse(patched)
            self.assertIn('prob.solve(solver=cp.OSQP)',patched)
            self.assertIn('action_input[3:6] = 0.2 * u_omega',patched)
            self.assertIn('                            a\n',patched)
            self.assertIn('_caught(e, t)',patched)
            self.assertIn('np.sum(np.abs(then_obstacle_pos - initial_obstacle_pos)) > 0.001',patched)
            self.assertEqual('obstacle_detection(agentview_img' in patched,method=='aegis')
            self.assertLess(patched.index('_settled(env, obs, t)'),patched.index('            flag_safety_control = False') if method=='nominal' else patched.index('            # Detect obstacles'))
        idle=adapter.adapt(source,'nominal',idle=True)
        ast.parse(idle);self.assertNotIn('client.infer',idle);self.assertNotIn('imageio.mimwrite',idle)
        with self.assertRaises(RuntimeError):adapter.adapt(source+'\n','aegis')

    def test_seed_reset_does_not_sample_and_replays_native_sequence(self):
        class Random:
            @staticmethod
            def key(seed):return seed
        class Policy:
            _rng=0
            def infer(self,obs):
                self._rng=(self._rng*1664525+1013904223)%2**32
                return {'actions':[self._rng]}
        policy=Policy();server=SeededPolicy(policy,Random,lambda k:k.to_bytes(4,'big'))
        with self.assertRaises(RuntimeError):server.infer({})
        a=server.infer({'__paired_reset_rng__':17,'run_id':'nominal'})
        first=[server.infer({})['actions'] for _ in range(3)]
        b=server.infer({'__paired_reset_rng__':17,'run_id':'aegis'})
        second=[server.infer({})['actions'] for _ in range(3)]
        self.assertEqual(first,second);self.assertEqual(a['key_sha256'],b['key_sha256'])
        self.assertEqual(a['requests'],0);self.assertEqual(server.requests,3)
        with self.assertRaises(ValueError):server.infer({'__paired_reset_rng__':-1,'run_id':'bad'})

class BudgetTests(unittest.TestCase):
    def test_same_pixels_each_run_still_reserve_and_send_separately(self):
        result={'model':'z-ai/glm-4.5v','provider':'Z.AI','usage':{'cost':.002},
                'choices':[{'finish_reason':'stop','message':{'content':'blue moka pot'}}]}
        class Response:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self):return json.dumps(result).encode()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);budget=Budget(root,'3.00');budget.refresh_prices=lambda:None
            try:
                directories=[export_request(b'png','task','safelibero_spatial',root/name,budgeted=True) for name in ['run1','run2']]
                self.assertEqual(directories[0].name,directories[1].name)
                with patch.dict('os.environ',{'OPENROUTER_API_KEY':'test-only'}),patch('urllib.request.urlopen',return_value=Response()) as net:
                    for i,d in enumerate(directories):fresh_fill(d,budget,'run%d'%i)
                    self.assertEqual(net.call_count,2)
                self.assertEqual(len(budget.state['calls']),2)
                self.assertEqual(str(budget.committed()),'3.004')
            finally:budget.close()

    def test_conserved_budget_blocks_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);budget=Budget(root,'4.91');budget.refresh_prices=lambda:None
            d=export_request(b'png','task','safelibero_spatial',root/'run1',budgeted=True)
            try:
                with patch.dict('os.environ',{'OPENROUTER_API_KEY':'test-only'}),patch('urllib.request.urlopen') as net:
                    with self.assertRaises(RuntimeError):fresh_fill(d,budget,'run1')
                    net.assert_not_called()
            finally:budget.close()

    def test_prior_ledger_carry_does_not_double_count_initial_balances(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i,initial,call in [('a','0',{'cost_usd':'.2','reserved_usd':'.1'}),('b','.2',{'reserved_usd':'.1'})]:
                d=root/'batches'/i;d.mkdir(parents=True)
                (d/'budget.json').write_text(json.dumps({'initial_spent_usd':initial,'calls':{'id':call}}))
            known,reserved,_=prior_spend(root,root/'new')
            self.assertEqual(str(known),'0.2');self.assertEqual(str(reserved),'0.1')

class GateTests(unittest.TestCase):
    def test_review_requires_passed_checks_and_unchanged_evidence(self):
        from execute import review_gate
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); evidence=root/'steps.jsonl';evidence.write_text('observed delta distribution')
            report={'idle':{'status':'passed'},'consistency':{'status':'passed'},'smoke':{'status':'awaiting_visual_and_delta_review'}}
            (root/'self_checks.json').write_text(json.dumps(report))
            review={'self_checks_sha256':hashlib.sha256((root/'self_checks.json').read_bytes()).hexdigest(),
                    'action_delta_linf_threshold':1e-6,'videos_normal':True,'distribution_rationale':'Recorded numerical floor separated from corrections',
                    'reviewed_files':{'steps.jsonl':hashlib.sha256(evidence.read_bytes()).hexdigest()}}
            (root/'review.json').write_text(json.dumps(review))
            self.assertEqual(review_gate(root),1e-6)
            evidence.write_text('changed')
            with self.assertRaises(RuntimeError):review_gate(root)
            report['idle']['status']='failed';(root/'self_checks.json').write_text(json.dumps(report))
            with self.assertRaises(RuntimeError):review_gate(root)

    def test_vlm_aliases_and_unknowns_do_not_invent_labels(self):
        from runtime import vlm_correct
        self.assertIs(vlm_correct('moka_pot_small_obstacle_1','blue moka pot'),True)
        self.assertIs(vlm_correct('red_coffee_mug_obstacle_1','blue moka pot'),False)
        self.assertIsNone(vlm_correct('moka_pot_obstacle_1','small blue metallic object'))

class UpdatedBudgetTests(unittest.TestCase):
    def test_new_ceiling_does_not_change_original_baseline_limit(self):
        import api_budget
        from paired_budget import PairedBudget
        from openrouter_perception import make_request
        with tempfile.TemporaryDirectory() as tmp:
            budget=PairedBudget(tmp,'63.79918268','65.00');budget.refresh_prices=lambda:None
            try:
                request=make_request(b'png','task','safelibero_spatial',budgeted=True)
                budget.reserve('last-permitted-attempt',request)
                self.assertEqual(str(budget.committed()),'63.89918268')
                self.assertEqual(str(api_budget.LIMIT),'5.00')
                self.assertEqual(budget.state['limit_usd'],'65.00')
            finally:budget.close()

    def test_custom_ceiling_still_stops_before_network(self):
        from paired_budget import PairedBudget
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);budget=PairedBudget(root,'64.91','65.00');budget.refresh_prices=lambda:None
            d=export_request(b'png','task','safelibero_spatial',root/'run',budgeted=True)
            try:
                with patch.dict('os.environ',{'OPENROUTER_API_KEY':'test-only'}),patch('urllib.request.urlopen') as network:
                    with self.assertRaises(RuntimeError):fresh_fill(d,budget,'run')
                    network.assert_not_called()
            finally:budget.close()

    def test_limit_mismatch_or_unapproved_increase_is_rejected(self):
        from paired_budget import PairedBudget
        with tempfile.TemporaryDirectory() as tmp:
            budget=PairedBudget(tmp,'3.79918268','5.00');budget.close()
            with self.assertRaises(RuntimeError):PairedBudget(tmp,limit='65.00')
            with self.assertRaises(RuntimeError):PairedBudget(tmp,limit='65.01')

    def test_unknown_charge_remains_reserved_on_new_guard(self):
        from paired_budget import PairedBudget
        from openrouter_perception import make_request
        with tempfile.TemporaryDirectory() as tmp:
            budget=PairedBudget(tmp,'3.79918268','65.00');budget.refresh_prices=lambda:None
            budget.reserve('attempt',make_request(b'png','task','safelibero_spatial',budgeted=True))
            with self.assertRaises(RuntimeError):budget.settle('attempt',{'usage':{}})
            budget.close()
            with self.assertRaises(RuntimeError):PairedBudget(tmp,limit='65.00')
            ledger=json.loads((Path(tmp)/'budget.json').read_text())
            self.assertEqual(ledger['calls']['attempt']['reserved_usd'],'0.10')

if __name__=='__main__':unittest.main()
