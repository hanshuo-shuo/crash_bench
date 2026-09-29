import copy
import unittest
from analyze_partial import audit
from protocol import schedule, scenario_name

def sample(n=21):
    return [dict(scenario=scenario_name(s),episode=e,repeat=r,method=m,seed=seed,
        phase='full',collided=False,max_obstacle_l1_m=0,modification_threshold=1e-6,
        qpos_sha256='a'*64,policy_reset_key='b'*64,first_action_chunk_sha256='c'*64,
        active_obstacle='test_obstacle',code_commit='fixed',upstream_commit='upstream',
        slurm_job='job',run_id='run%d'%i)
        for i,(s,e,r,m,seed) in enumerate(list(schedule())[:n])]

class PartialAuditTests(unittest.TestCase):
    def test_missing_repeat_is_excluded_not_counted_as_failure(self):
        complete,states,result=audit(sample())
        self.assertEqual(len(complete),20)
        self.assertEqual(len(states),3)
        self.assertEqual(result['completed_pair_checks'],10)
        self.assertTrue(result['exact_first_chunk_pairs_passed'])

    def test_action_mismatch_is_exposed_without_silent_exclusion(self):
        rows=sample();rows[0]['first_action_chunk_sha256']='d'*64
        complete,_,result=audit(rows)
        self.assertEqual(len(complete),20)
        self.assertFalse(result['exact_first_chunk_pairs_passed'])
        self.assertEqual(len(result['first_chunk_mismatch_pairs']),1)
        self.assertEqual(result['seed_and_qpos_mismatches'],0)

    def test_bad_seed_duplicate_or_state_is_rejected(self):
        for key,value in [('seed',0),('qpos_sha256','d'*64),('policy_reset_key','e'*64),('collided',True)]:
            rows=sample();rows[0][key]=value
            with self.assertRaises(ValueError):audit(rows)
        rows=sample();rows[1]=copy.deepcopy(rows[0])
        with self.assertRaises(ValueError):audit(rows)

if __name__=='__main__':unittest.main()
