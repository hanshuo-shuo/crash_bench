import json
from pathlib import Path
import tempfile
import unittest
from verifier_dataset.collection_evidence import replay_candidates


def write(path,value):path.write_text(json.dumps(value))


class CollectionInheritance(unittest.TestCase):
    def fixture(self,root,success=True):
        scenes=[dict(suite='safelibero_goal',level='I',task=i,episode=0) for i in range(2)]
        write(root/'RECEIPT.json',dict(stage='collect-cached-smoke',api_mode='cache_only',code_commit='abc',job='42'))
        write(root/'API_FINAL.json',dict(paid_calls=0,worker_started=False))
        write(root/'COLLECTION_SOURCES.json',[dict(scene=s) for s in scenes])
        result=dict(spec=dict(name='fd_goal00_aegis',scene=scenes[0],method='aegis',repeat=0,diagnostics=False),
                    status='complete',success=success,collision_step=None,exited=None,seed=17,actions=30,
                    outcome='safe_success' if success else 'safe_incomplete')
        path=root/'runs/fd_goal00_aegis';path.mkdir(parents=True)
        write(path/'RESULT.json',result)
        write(path/'manifest.json',dict(status='complete',code_commit='abc',slurm_job='42'))
        (path/'steps.jsonl').write_text('{}\n');(path/'settled_state.npz').write_bytes(b'fixture')
        deferred=dict(source='safelibero_goal/I/task1/episode0',run_name='fd_goal01_aegis',reason='exact_cache_missing',
                      feasibility_label='unknown',paid_request_sent=False)
        write(root/'COLLECTION_COMPLETE.json',dict(passed=True,runs=1,rows=[result],deferred=[deferred]))

    def test_cached_success_is_unknown_until_replayed_and_miss_is_preserved(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);self.fixture(root)
            rows,evidence=replay_candidates(root)
            self.assertEqual(len(rows),1);self.assertEqual(rows[0]['feasibility_label'],'unknown')
            self.assertEqual(evidence['unresolved'][0]['reason'],'exact_cache_missing')

    def test_policy_failure_cannot_become_a_replay_positive_or_negative(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);self.fixture(root,success=False)
            with self.assertRaisesRegex(RuntimeError,'No completed successful'):replay_candidates(root)

    def test_result_disagreement_fails_before_submission(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);self.fixture(root)
            write(root/'runs/fd_goal00_aegis/RESULT.json',{})
            with self.assertRaisesRegex(RuntimeError,'results disagree'):replay_candidates(root)


if __name__=='__main__':unittest.main()
