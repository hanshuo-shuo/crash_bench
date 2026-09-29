import copy
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from recovery import select,load_inherited,scheduled,EVIDENCE,first_chunk_audit
from slurm_monitor import read_job_state

def reply(stdout='',code=0,stderr=''):
    return SimpleNamespace(stdout=stdout,returncode=code,stderr=stderr)

class MonitorTests(unittest.TestCase):
    def test_running_job_does_not_query_accounting(self):
        with patch('slurm_monitor.subprocess.run',return_value=reply('42|RUNNING\n')) as run:
            self.assertEqual(read_job_state('42')['state'],'RUNNING')
            self.assertEqual(run.call_count,1)
            self.assertEqual(run.call_args.args[0][0],'squeue')

    def test_transient_accounting_failure_is_unknown_not_failure(self):
        with patch('slurm_monitor.subprocess.run',side_effect=[reply(),reply(code=1,stderr='database timeout')]):
            state=read_job_state('42')
            self.assertEqual(state['state'],'UNKNOWN')
            self.assertIn('timeout',state['query_errors'][0])

    def test_timeouts_on_both_services_are_nonfatal(self):
        with patch('slurm_monitor.subprocess.run',side_effect=subprocess.TimeoutExpired('query',8)):
            self.assertEqual(read_job_state('42')['state'],'UNKNOWN')

    def test_only_matching_job_terminal_state_is_accepted(self):
        with patch('slurm_monitor.subprocess.run',side_effect=[reply(),reply('99|FAILED\n42|CANCELLED by 5775\n')]):
            self.assertEqual(read_job_state('42')['state'],'CANCELLED')
        with patch('slurm_monitor.subprocess.run',side_effect=[reply('99|FAILED\n'),reply('99|FAILED\n')]):
            self.assertEqual(read_job_state('42')['state'],'UNKNOWN')

class RecoveryTests(unittest.TestCase):
    def parent(self,path,n):
        (path/'full/runs').mkdir(parents=True)
        (path/'plan.json').write_text(json.dumps({'code_commit':'science','jobs':{'full':'42'}}))
        (path/'STOP.json').write_text('{"reason":"accounting timeout"}')
        for i,item in enumerate(scheduled()[:n]):
            row=dict(item,phase='full',code_commit='science',collided=False,max_obstacle_l1_m=0.,
                modification_threshold=1e-6,qpos_sha256='q',active_obstacle='obstacle',
                policy_reset_key='rng',first_action_chunk_sha256='chunk')
            d=path/'full/runs'/row['run_id'];d.mkdir()
            (d/'row.json').write_text(json.dumps(row));(d/'manifest.json').write_text(json.dumps(dict(row,status='complete')))
            for name in EVIDENCE[2:]:(d/name).write_bytes(b'evidence')
        if n<600:
            d=path/'full/runs'/scheduled()[n]['run_id'];d.mkdir()
            (d/'manifest.json').write_text('{"status":"started"}')

    def test_exact_153_missing_runs_and_447_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.parent(root,447)
            before=(root/'STOP.json').read_bytes()
            inherited,pending,unfinished=select(root)
            self.assertEqual(len(inherited),447);self.assertEqual(len(pending),153)
            self.assertEqual(sum(r['method']=='aegis' for r in pending),76)
            self.assertEqual(pending[0]['run_id'],'full_s2_e04_r3_nominal')
            self.assertEqual(len(unfinished),1)
            self.assertEqual((root/'STOP.json').read_bytes(),before)
            self.assertEqual(len(load_inherited({'inherited_records':inherited})),447)

    def test_tampered_inherited_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.parent(root,2)
            inherited,_,_=select(root)
            Path(inherited[0]['directory'],'settled_qpos.npy').write_bytes(b'changed')
            with self.assertRaises(RuntimeError):load_inherited({'inherited_records':inherited})

    def test_missing_middle_record_cannot_silently_change_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.parent(root,3)
            (root/'full/runs'/scheduled()[1]['run_id']/'row.json').unlink()
            with self.assertRaises(RuntimeError):select(root)

    def test_quality_warning_retains_action_mismatches(self):
        rows=[dict(scenario='s',episode=0,repeat=0,seed=1,qpos_sha256='q',policy_reset_key='r',
                   first_action_chunk_sha256=chunk) for chunk in ['a','b']]
        self.assertEqual(first_chunk_audit(rows)['first_chunk_mismatch_count'],1)
        rows[1]['qpos_sha256']='different'
        with self.assertRaises(RuntimeError):first_chunk_audit(rows)

if __name__=='__main__':unittest.main()
