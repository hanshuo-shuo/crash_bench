import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from api_budget import Budget,validate_endpoint,atomic_json
import openrouter_perception as api
from safelibero_batch import matrix,collect,recovery_selection
import batch_api_worker

class BudgetSafety(unittest.TestCase):
    def request(self):return api.make_request(b'png','task','safelibero_spatial',budgeted=True)

    def test_insufficient_budget_prevents_paid_network_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);budget=Budget(root,'4.91')
            try:
                directory=api.export_request(b'png','task','safelibero_spatial',root/'cache',budgeted=True)
                with patch.dict('os.environ',{'OPENROUTER_API_KEY':'test-placeholder'}),patch.object(budget,'refresh_prices'),patch('urllib.request.urlopen') as network:
                    with self.assertRaisesRegex(RuntimeError,'budget boundary'):api.fill(directory,budget)
                    network.assert_not_called()
                self.assertEqual(budget.state['calls'],{})
            finally:budget.close()

    def test_unknown_charge_retains_hold_and_prevents_automatic_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget=Budget(tmp,'0.00279368')
            with patch.object(budget,'refresh_prices'):budget.reserve('request',self.request())
            with self.assertRaisesRegex(RuntimeError,'Unknown API charge'):budget.settle('request',{})
            self.assertEqual(str(budget.committed()),'0.10279368')
            budget.close()
            with self.assertRaisesRegex(RuntimeError,'Unresolved'):Budget(tmp)

    def test_settled_cost_replaces_reservation_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget=Budget(tmp,'0.00279368')
            with patch.object(budget,'refresh_prices'):budget.reserve('request',self.request())
            budget.settle('request',{'usage':{'cost':0.0018}});budget.close()
            budget=Budget(tmp)
            try:
                self.assertEqual(str(budget.committed()),'0.00459368')
                with self.assertRaisesRegex(RuntimeError,'already'):budget.reserve('request',self.request())
                with self.assertRaises(BlockingIOError):Budget(tmp)
            finally:budget.close()

    def test_changed_provider_bounds_fail_closed(self):
        endpoint={'provider_name':'Z.AI','context_length':65536,'max_completion_tokens':16384,'pricing':{'prompt':'0.0000006','completion':'0.0000018'}}
        data={'data':{'endpoints':[endpoint]}};validate_endpoint(data)
        endpoint['context_length']=131072
        with self.assertRaises(RuntimeError):validate_endpoint(data)
        endpoint['context_length']=65536;endpoint['pricing']['completion']='0.000002'
        with self.assertRaises(RuntimeError):validate_endpoint(data)

    def test_truncated_response_is_charged_but_never_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);budget=Budget(root)
            directory=api.export_request(b'png','task','safelibero_spatial',root/'cache',budgeted=True)
            reply={'model':api.MODEL,'provider':'Z.AI','choices':[{'finish_reason':'length','message':{'content':'blue'}}],'usage':{'cost':0.002}}
            import io
            with patch.dict('os.environ',{'OPENROUTER_API_KEY':'test-placeholder'}),patch.object(budget,'refresh_prices'),patch('urllib.request.urlopen',return_value=io.StringIO(json.dumps(reply))):
                with self.assertRaises(RuntimeError):api.fill(directory,budget)
            self.assertFalse((directory/'response.json').exists())
            self.assertTrue((directory/'failed_response.json').exists())
            self.assertEqual(str(budget.committed()),'0.002');budget.close()

class MatrixSafety(unittest.TestCase):
    def test_exact_official_coverage_and_pairing(self):
        rows=matrix();self.assertEqual(len(rows),64)
        identities={(r['mode'],r['suite'],r['level'],r['task'],e) for r in rows for e in r['episodes']}
        self.assertEqual(len(identities),3200)
        for first,second in zip(rows[:32],rows[32:]):
            self.assertEqual({k:v for k,v in first.items() if k not in ['index','mode']},{k:v for k,v in second.items() if k not in ['index','mode']})

    def test_partial_results_not_presented_as_complete_or_mixed_commits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);atomic_json(root/'batch.json',{'cells':matrix(),'code_commit':'frozen'})
            out=root/'runs/00_frozen_1/evaluation';out.mkdir(parents=True)
            m={'mode':'nominal','suite':'safelibero_spatial','level':'I','task':0,'code_commit':'frozen','status':'started'}
            atomic_json(out/'manifest.json',m)
            atomic_json(out/'episodes.json',[{'episode':0,'success':False,'collision':True,'safe_success':False,'steps':300}])
            status=collect(root);self.assertTrue(status['partial']);self.assertEqual(status['completed_episodes'],1)
            m['status']='complete';atomic_json(out/'manifest.json',m)
            with self.assertRaisesRegex(RuntimeError,'missing official'):collect(root)
            m.update(status='started',code_commit='other');atomic_json(out/'manifest.json',m)
            with self.assertRaisesRegex(RuntimeError,'Mixed code'):collect(root)

    def test_worker_budget_failure_cancels_only_recorded_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);atomic_json(root/'batch.json',{'slurm_arrays':['123','124'],'prior_api_spend_usd':'0'})
            directory=root/'perception_cache/request';directory.mkdir(parents=True);(directory/'request.json').write_text('{}')
            with patch.object(batch_api_worker,'Budget') as budget,patch.object(batch_api_worker,'collect',return_value={'failed_cells':[],'complete_cells':[]}),patch.object(batch_api_worker,'reuse_cache',return_value=False),patch.object(batch_api_worker,'fill',side_effect=RuntimeError('budget stop')),patch.object(batch_api_worker.subprocess,'run') as run:
                run.return_value.stdout='';budget.return_value.committed.return_value='4.95'
                with self.assertRaisesRegex(RuntimeError,'budget stop'):batch_api_worker.main(root)
                self.assertTrue((root/'STOP.json').exists())
                self.assertEqual(run.call_args.args[0],['scancel','123','124'])

class RecoverySafety(unittest.TestCase):
    def test_recovery_inherits_complete_cells_and_excludes_partial_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'old';root.mkdir()
            atomic_json(root/'batch.json',{'cells':matrix(),'code_commit':'old'})
            atomic_json(root/'STOP.json',{'reason':'HTTP 520'})
            for i,count in [(0,50),(1,7)]:
                out=root/('runs/%02d_old_1/evaluation'%i);out.mkdir(parents=True)
                cell=matrix()[i]
                atomic_json(out/'manifest.json',{**{k:cell[k] for k in ['mode','suite','level','task']},'code_commit':'old','status':'complete' if count==50 else 'started'})
                atomic_json(out/'episodes.json',[{'episode':e,'success':True,'collision':False,'safe_success':True,'steps':100} for e in range(count)])
            inherited,partial,pending=recovery_selection(root)
            self.assertEqual(list(inherited),['0']);self.assertEqual(pending,list(range(1,64)))
            self.assertEqual(partial['1']['retained_but_excluded_episode_indices'],list(range(7)))
            new=Path(tmp)/'new';new.mkdir()
            atomic_json(new/'batch.json',{'cells':matrix(),'code_commit':'new','inherited_cells':inherited})
            status=collect(new);self.assertEqual(status['completed_episodes'],50)
            self.assertTrue(status['partial'])
            old=Path(inherited['0']['evaluation_dir'])/'episodes.json';old.write_text('[]')
            with self.assertRaisesRegex(RuntimeError,'Inherited result changed'):collect(new)

    def test_exact_cache_reuse_needs_no_paid_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            old=api.export_request(b'png','task','safelibero_spatial',root/'old',budgeted=True)
            new=api.export_request(b'png','task','safelibero_spatial',root/'new',budgeted=True)
            response={'request_sha256':old.name,'response':{'model':api.MODEL,'provider':'Z.AI','choices':[{'finish_reason':'stop','message':{'content':'blue moka pot'}}]}}
            atomic_json(old/'response.json',response)
            with patch('urllib.request.urlopen') as network:
                self.assertTrue(batch_api_worker.reuse_cache(new,[root/'old']))
                network.assert_not_called()
            self.assertEqual((old/'response.json').read_bytes(),(new/'response.json').read_bytes())
            self.assertEqual(api.read_response(new),'blue moka pot')
            changed=api.export_request(b'other','task','safelibero_spatial',root/'new',budgeted=True)
            self.assertFalse(batch_api_worker.reuse_cache(changed,[root/'old']))

    def test_http_retry_keeps_unknown_charge_and_reserves_again(self):
        import io,urllib.error
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);budget=Budget(root,'1.70775168')
            directory=api.export_request(b'png','task','safelibero_spatial',root/'cache',budgeted=True)
            reply={'model':api.MODEL,'provider':'Z.AI','choices':[{'finish_reason':'stop','message':{'content':'blue moka pot'}}],'usage':{'cost':0.002}}
            error=urllib.error.HTTPError('https://openrouter.ai/api/v1/chat/completions',520,'test',None,None)
            with patch.dict('os.environ',{'OPENROUTER_API_KEY':'test-placeholder'}),patch.object(budget,'refresh_prices'),patch('urllib.request.urlopen',side_effect=[error,io.StringIO(json.dumps(reply))]) as network,patch.object(batch_api_worker.time,'sleep'):
                self.assertEqual(batch_api_worker.fill_with_transport_retry(directory,budget),'blue moka pot')
                self.assertEqual(network.call_count,2)
            self.assertEqual(str(budget.committed()),'1.80975168')
            self.assertEqual(budget.state['calls'][directory.name]['status'],'reserved')
            self.assertEqual(budget.state['calls'][directory.name]['http_error'],520)
            budget.close()

    def test_retry_stops_at_budget_without_second_paid_request(self):
        import urllib.error
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);budget=Budget(root,'4.85')
            directory=api.export_request(b'png','task','safelibero_spatial',root/'cache',budgeted=True)
            error=urllib.error.HTTPError('https://openrouter.ai/api/v1/chat/completions',520,'test',None,None)
            with patch.dict('os.environ',{'OPENROUTER_API_KEY':'test-placeholder'}),patch.object(budget,'refresh_prices'),patch('urllib.request.urlopen',side_effect=error) as network,patch.object(batch_api_worker.time,'sleep'):
                with self.assertRaisesRegex(RuntimeError,'budget boundary'):batch_api_worker.fill_with_transport_retry(directory,budget)
                self.assertEqual(network.call_count,1)
            self.assertEqual(str(budget.committed()),'4.95');budget.close()

    def test_http_errors_are_bounded_to_two_attempts(self):
        import urllib.error
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);budget=Budget(root)
            directory=api.export_request(b'png','task','safelibero_spatial',root/'cache',budgeted=True)
            error=urllib.error.HTTPError('https://openrouter.ai/api/v1/chat/completions',520,'test',None,None)
            with patch.dict('os.environ',{'OPENROUTER_API_KEY':'test-placeholder'}),patch.object(budget,'refresh_prices'),patch('urllib.request.urlopen',side_effect=error) as network,patch.object(batch_api_worker.time,'sleep'):
                with self.assertRaises(urllib.error.HTTPError):batch_api_worker.fill_with_transport_retry(directory,budget)
                self.assertEqual(network.call_count,2)
            self.assertEqual(str(budget.committed()),'0.20');budget.close()

if __name__=='__main__':unittest.main()
