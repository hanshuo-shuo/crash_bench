import ast
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
import common
from budget import CampaignBudget
from adapter import adapt


class MetricTests(unittest.TestCase):
    def test_frozen_schedule_pairing_and_no_full_stage(self):
        plans=list(common.schedule(common.config()))
        self.assertEqual(len(plans),9)
        self.assertFalse(plans[0]['diagnostics'])
        self.assertEqual(sum(x['method']=='aegis' for x in plans),4)
        for i in range(1,9,2):
            self.assertEqual(common.seed_for(plans[i]['scene'],plans[i]['repeat']),
                             common.seed_for(plans[i+1]['scene'],plans[i+1]['repeat']))
        self.assertEqual(common.seed_for(plans[0]['scene'],0),2628542810)

    def test_sample_std_not_variance_and_valid_axes(self):
        chunks=[[[float(i)]*7 for _ in range(10)] for i in range(8)]
        self.assertAlmostEqual(common.disagreement(chunks,[2.]*7),math.sqrt(6)/2.)
        self.assertEqual(common.disagreement([chunks[0]]*8,[1.]*7),0.)

    def test_churn_uses_shifted_overlap(self):
        previous=[[float(t)]*7 for t in range(10)]
        current=[[float(t+5)]*7 for t in range(10)]
        self.assertEqual(common.churn(current,previous,[1.]*7),0.)
        self.assertIsNone(common.churn(current,None,[1.]*7))
        self.assertEqual(common.norm([0.]*6+[1.],[1.]*7),1.)
        self.assertEqual(common.norm([0.]*6+[1.],[1.]*7,6),0.)

    def test_collision_clock_and_outcome(self):
        self.assertEqual([common.time_to_crash(x,3) for x in [1,2,3,4]],[2,1,0,None])
        self.assertEqual(common.time_to_crash(300,None),-1)
        self.assertEqual(common.outcome(True,3),'crash')
        self.assertEqual(common.outcome(False,None),'safe_incomplete')
        self.assertEqual(common.outcome(True,None),'safe_success')

    def test_sampling_seed_is_method_independent_and_distinct(self):
        values=[common.noise_seed('scene',7,2,i) for i in range(8)]
        self.assertEqual(len(set(values)),8)
        self.assertNotEqual(values,[common.noise_seed('scene',7,3,i) for i in range(8)])

    def test_full_identity_shards_and_scene_split(self):
        cfg=common.config('full'); whole=list(common.full_schedule(cfg))
        parts=[list(common.full_schedule(cfg,i)) for i in [0,1]]
        self.assertEqual([len(x) for x in parts],[600,600])
        self.assertEqual(len({x['name'] for x in whole}),1200)
        self.assertEqual({x['name'] for x in whole},set(x['name'] for p in parts for x in p))
        self.assertFalse(set(x['name'] for x in parts[0])&set(x['name'] for x in parts[1]))
        for p in parts:
            for a,b in zip(p[::2],p[1::2]):
                self.assertEqual(common.seed_for(a['scene'],a['repeat']),common.seed_for(b['scene'],b['repeat']))
                self.assertEqual({a['method'],b['method']},{'nominal','aegis'})
        split=json.loads((Path(__file__).parent/'split.json').read_text())
        self.assertEqual((len(split['train']),len(split['test'])),(42,18))
        self.assertFalse(set(split['train'])&set(split['test']))
        self.assertTrue(set(split['exposed_smoke_assigned_train'])<=set(split['train']))

    def test_repair_allocations_share_cumulative_ceiling(self):
        receipts=[dict(state='FAILED',elapsed_seconds=361),dict(state='FAILED',elapsed_seconds=362)]
        minutes,used=common.remaining_minutes(receipts)
        self.assertEqual((minutes,used),(17,723))
        self.assertLessEqual(used+minutes*60,1800)
        with self.assertRaisesRegex(RuntimeError,'unknown or active'):
            common.remaining_minutes([dict(state='RUNNING',elapsed_seconds=1)])
        with self.assertRaisesRegex(RuntimeError,'exhausted'):
            common.remaining_minutes([dict(state='TIMEOUT',elapsed_seconds=1800)])


class BudgetTests(unittest.TestCase):
    def request(self):
        from api_budget import MAX_PRICE
        return dict(model='z-ai/glm-4.5v',max_tokens=16384,
                    provider={'only':['z-ai'],'allow_fallbacks':False,'require_parameters':True,'max_price':MAX_PRICE})

    def test_unknown_reservations_and_retries_hit_exact_three_dollars(self):
        with tempfile.TemporaryDirectory() as t:
            b=CampaignBudget(t);b.refresh_prices=lambda: None
            try:
                for i in range(30):b.reserve(str(i),self.request())
                self.assertEqual(str(b.committed()),'3.00')
                with self.assertRaisesRegex(RuntimeError,'boundary'): b.reserve('31',self.request())
            finally:b.close()
            with self.assertRaisesRegex(RuntimeError,'unknown charge'): CampaignBudget(t)

    def test_actual_usage_settlement_and_unknown_cost(self):
        with tempfile.TemporaryDirectory() as t:
            b=CampaignBudget(t);b.refresh_prices=lambda: None
            try:
                b.reserve('a',self.request())
                with self.assertRaisesRegex(RuntimeError,'Unknown API charge'):b.settle('a',{})
                self.assertEqual(str(b.committed()),'0.10')
                b.settle('a',dict(usage={'cost':.002,'prompt_tokens':100,'completion_tokens':50},model='z-ai/glm-4.5v',provider='Z.AI'))
                self.assertEqual(str(b.committed()),'0.002')
                self.assertEqual(b.state['calls']['a']['usage']['prompt_tokens'],100)
            finally:b.close()

    def test_full_authorization_preserves_ledger_and_removes_dollar_cap(self):
        with tempfile.TemporaryDirectory() as t:
            b=CampaignBudget(t);b.refresh_prices=lambda:None
            b.reserve('old',self.request());b.settle('old',dict(usage={'cost':.002},model='z-ai/glm-4.5v',provider='Z.AI'));b.close()
            b=CampaignBudget(t,common.config('full'));b.refresh_prices=lambda:None
            try:
                self.assertIsNone(b.limit);self.assertEqual(str(b.committed()),'0.002')
                for i in range(31):b.reserve('full'+str(i),self.request())
                receipt=json.loads((Path(t)/'API_AUTHORIZATION_CHANGE_20261009.json').read_text())
                self.assertEqual(receipt['previous_ledger']['calls']['old']['status'],'settled')
                self.assertEqual(receipt['previous_limit_usd'],'3.00')
            finally:b.close()


class AdapterTests(unittest.TestCase):
    def test_pinned_scoring_and_qp_stay_byte_identical(self):
        source=Path('/tmp/crashbench_uncertainty_main_aegis.py')
        if not source.exists():self.skipTest('Pinned evaluator supplied on Quest or during local verification')
        original=source.read_text()
        for method in ['nominal','aegis']:
            text=adapt(original,method);ast.parse(text)
            for start,end in [('                        v_ref =','                        action_input ='),
                              ('                    if collide_flag == False:','                    eef_pos ='),
                              ('                    if done:','                except Exception as e:')]:
                a=original[original.index(start):original.index(end,original.index(start))]
                b=text[text.index(start):text.index(end,text.index(start))]
                self.assertEqual(a,b)
            self.assertEqual(text.count('observer.after(obs, done, t)'),2)
            self.assertIn('action_plan.extend(action_chunk[: args.replan_steps])',text)


class FullWorkerTests(unittest.TestCase):
    def fake_budget(self,*args):
        class Fake:
            path=Path('test_budget.json');state={'calls':{}}
            def refresh_prices(self):pass
            def committed(self):return 0
            def close(self):pass
        return Fake()

    def root(self,t):
        root=Path(t)
        (root/'plan.json').write_text(json.dumps(dict(configuration=common.config('full'),
            jobs=['10001','10002'],campaign_root=t)))
        for i in [0,1]:(root/'shards'/str(i)).mkdir(parents=True)
        return root

    def test_invalid_request_cancels_only_two_recorded_jobs_before_any_call(self):
        import full_worker
        with tempfile.TemporaryDirectory() as t:
            root=self.root(t);p=root/'shards/0/requests/not_authorized/hash';p.mkdir(parents=True)
            (p/'request.json').write_text('{}')
            with patch.object(full_worker,'CampaignBudget',side_effect=self.fake_budget),patch.object(full_worker,'fill') as paid,patch.object(full_worker.subprocess,'run') as command:
                with self.assertRaisesRegex(RuntimeError,'outside frozen'):full_worker.serve(root)
                paid.assert_not_called()
                self.assertEqual([x.args[0] for x in command.call_args_list],[['scancel','10001'],['scancel','10002']])

    def test_second_release_requires_retained_eight_case_smoke(self):
        import full_worker
        with tempfile.TemporaryDirectory() as t:
            root=self.root(t)
            (root/'shards/0/FULL_SMOKE_PASS.json').write_text(json.dumps(dict(passed=True,runs=8)))
            for i in [0,1]:(root/'shards'/str(i)/'COMPLETE.json').write_text('{}')
            with patch.object(full_worker,'CampaignBudget',side_effect=self.fake_budget),patch.object(full_worker.subprocess,'run') as command,patch.object(full_worker,'read_job_state',return_value={'state':'COMPLETED'}):
                full_worker.serve(root)
                self.assertEqual([x.args[0] for x in command.call_args_list],[['scontrol','release','10002']])
                self.assertTrue((root/'COLLECTION_COMPLETE.json').exists())


class TerminalEvidenceTests(unittest.TestCase):
    def test_delivery_rejects_traversal_and_symlink_descendants(self):
        from receive_evidence import checked_paths
        with self.assertRaisesRegex(ValueError,'Unsafe'):checked_paths([dict(path='../escape',kind='file')],1)
        with self.assertRaisesRegex(ValueError,'beneath'):checked_paths([dict(path='link',kind='symlink'),dict(path='link/file',kind='file')],1)
        with self.assertRaisesRegex(ValueError,'Duplicate'):checked_paths([dict(path='a',kind='file')]*2,1)

    def test_delivery_checks_all_regular_bytes_and_preserves_inert_links(self):
        from receive_evidence import receive
        import hashlib,io,tarfile
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);data=b'complete evidence';link='/gpfs/projects/p33100/siosio/crashbench_safelibero/nonexistent_test_asset'
            manifest=[dict(path='file',kind='file',bytes=len(data),sha256=hashlib.sha256(data).hexdigest()),dict(path='asset',kind='symlink',target=link)]
            (p/'FILES.json').write_text(json.dumps(manifest))
            with tarfile.open(p/'evidence.tar','w') as tar:
                item=tarfile.TarInfo('shard_1/file');item.size=len(data);tar.addfile(item,io.BytesIO(data))
                item=tarfile.TarInfo('shard_1/asset');item.type=tarfile.SYMTYPE;item.linkname=link;tar.addfile(item)
            report=dict(passed=True,shard=1,runs=600,actions=1,diagnostic_inferences=1,collection_commit='original',audit_commit='audit',manifest_sha256=common.sha(p/'FILES.json'),archive_sha256=common.sha(p/'evidence.tar'))
            (p/'SHARD_AUDIT.json').write_text(json.dumps(report));proof=receive(p/'evidence.tar',p/'FILES.json',p/'SHARD_AUDIT.json',p/'delivered')
            self.assertEqual(proof['regular_files'],1);self.assertEqual((p/'delivered/shard_1/file').read_bytes(),data)
            self.assertEqual((p/'delivered/shard_1/asset').readlink(),Path(link))
            with (p/'evidence.tar').open('ab') as stream:stream.write(b'corruption')
            with self.assertRaisesRegex(ValueError,'hash differs'):receive(p/'evidence.tar',p/'FILES.json',p/'SHARD_AUDIT.json',p/'bad')
            self.assertFalse((p/'bad').exists())

    def test_audit_refuses_running_or_failed_accounting(self):
        from audit_shard import terminal_accounting
        from types import SimpleNamespace
        for state,exit_code in [('RUNNING','0:0'),('FAILED','1:0'),('COMPLETED','1:0')]:
            with patch('audit_shard.subprocess.run',return_value=SimpleNamespace(stdout='10001|%s|%s|12\n'%(state,exit_code))):
                with self.assertRaisesRegex(RuntimeError,'not terminal'):terminal_accounting('10001')
        with patch('audit_shard.subprocess.run',return_value=SimpleNamespace(stdout='10001|COMPLETED|0:0|12\n')):
            self.assertEqual(terminal_accounting('10001')['elapsed_seconds'],12)

    def test_audit_requires_full_frozen_case_order_and_commit(self):
        from audit_shard import validate_receipt
        import copy
        specs=list(common.full_schedule(common.config('full'),1))
        receipt=dict(passed=True,runs=600,complete_matrix_shard=True,source_commit='frozen',results=[dict(spec=s,status='complete') for s in specs])
        validate_receipt(receipt,specs,'frozen')
        bad=copy.deepcopy(receipt);bad['results']=bad['results'][:-1]
        with self.assertRaisesRegex(RuntimeError,'identities'):validate_receipt(bad,specs,'frozen')
        bad=copy.deepcopy(receipt);bad['results'][0],bad['results'][1]=bad['results'][1],bad['results'][0]
        with self.assertRaisesRegex(RuntimeError,'identities'):validate_receipt(bad,specs,'frozen')
        with self.assertRaisesRegex(RuntimeError,'receipt'):validate_receipt(receipt,specs,'different')


class StatisticalRuleTests(unittest.TestCase):
    def test_never_crash_postaction_collision_and_censoring(self):
        from stat_rules import collision_target
        self.assertEqual(collision_target(10,None,300,False,5),0)
        self.assertEqual(collision_target(10,10,300,False,5),1)
        self.assertIsNone(collision_target(11,10,300,False,5))
        self.assertEqual(collision_target(10,14,300,False,5),1)
        self.assertEqual(collision_target(10,15,300,False,5),0)
        self.assertIsNone(collision_target(291,None,300,False,40))
        self.assertEqual(collision_target(141,None,152,True,40),0)

    def test_landmark_rejects_observed_early_crash(self):
        from stat_rules import first_five_eligible
        infos=[{'step':x} for x in [1,6,11,16,21]]
        self.assertFalse(first_five_eligible(infos,16,300))
        self.assertTrue(first_five_eligible(infos,21,300))
        self.assertFalse(first_five_eligible(infos[:4],None,18))

    def test_equal_state_and_rollout_weight_not_action_weight(self):
        from stat_rules import grouped_weights,weighted_auc,weighted_quantile
        scenes=['A']*3+['B'];runs=['A0','A0','A1','B0']
        self.assertEqual(grouped_weights(scenes,runs),[.125,.125,.25,.5])
        self.assertEqual(weighted_auc([0,1],[0.,0.],[.5,.5]),.5)
        self.assertEqual(weighted_auc([0,1],[0.,1.],[.8,.2]),1.)
        self.assertIsNone(weighted_auc([0,0],[0.,1.],[.5,.5]))
        self.assertEqual(weighted_quantile([0,1,2],[.5,.25,.25],.9),2.)

    def test_bootstrap_retains_duplicated_cluster_mass(self):
        from stat_rules import grouped_weights,bootstrap_weights
        scenes=['A','A','B'];runs=['A0','A0','B0']
        original=grouped_weights(scenes,runs)
        self.assertEqual(bootstrap_weights(scenes,original,['A','A','B']),[1/3,1/3,1/3])
        self.assertEqual(bootstrap_weights(scenes,original,['A','A','A']),[.5,.5,0.])
        self.assertIsNone(bootstrap_weights(scenes,original,['C']))

    def test_failure_window_fraction_is_per_rollout_and_action_based(self):
        from stat_rules import failure_window_fraction
        infos=[{'step':x,'disagreement':u} for x,u in [(1,.1),(6,.2),(11,.3),(16,.1)]]
        self.assertEqual(failure_window_fraction(infos,12,10,.2),dict(fraction=.5,low=1,boundaries=2,partial=False))
        self.assertEqual(failure_window_fraction(infos,11,5,.2)['fraction'],0.)
        self.assertTrue(failure_window_fraction(infos,2,10,.2)['partial'])
        self.assertIsNone(failure_window_fraction(infos,5,1,.2)['fraction'])

    def test_requested_confident_failure_mean_excludes_event_and_keeps_duration(self):
        from stat_rules import pre_event_action_mean,confident_failure_indicator
        rows=[{'step':t,'disagreement':float(1+(t-1)//5)} for t in range(1,16)]
        value=pre_event_action_mean(rows,9,5)
        self.assertTrue(value['eligible']);self.assertEqual((value['start'],value['end']),(4,8))
        self.assertEqual(value['mean'],1.6)
        self.assertEqual(confident_failure_indicator(1.6,1.7),1)
        self.assertEqual(confident_failure_indicator(1.6,1.6),0)
        self.assertIsNone(confident_failure_indicator(1.6,None))

    def test_requested_confident_failure_rejects_short_or_missing_windows(self):
        from stat_rules import pre_event_action_mean
        rows=[{'step':t,'disagreement':.2} for t in range(1,8)]
        self.assertEqual(pre_event_action_mean(rows,5,5)['reason'],'incomplete_window')
        self.assertEqual(pre_event_action_mean(rows[:6],8,5)['reason'],'missing_action_rows')
        rows[4]['disagreement']=None
        self.assertEqual(pre_event_action_mean(rows,8,5)['reason'],'missing_proxy')


if __name__=='__main__':unittest.main()
