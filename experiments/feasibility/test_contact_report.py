import unittest
from contact_report import summarize


class ContactReportTests(unittest.TestCase):
    def test_proxy_success_is_not_no_contact_success_and_samples_not_episodes(self):
        item = {'replay_verified': True, 'exact_physics_verified_checkpoints': [0,27],
                'run_id': 'screen', 'state': 'spatial_09', 'source_success': True,
                'source_safe_success': True, 'source_proxy_max_l1_m': .00065, 'slurm_job': '123',
                'events': [{'category': 'robot', 'action': 146, 'distance_m': -.001,
                            'body_pair': ['bottle','link5']}] * 3}
        result = summarize(item)
        self.assertTrue(result['official_proxy_safe_complete'])
        self.assertFalse(result['complete_without_robot_target_protected_contact'])
        self.assertEqual(result['robot_contact_actions'], 1)
        item['events'] = []
        self.assertTrue(summarize(item)['complete_without_robot_target_protected_contact'])
        item['replay_verified'] = False
        with self.assertRaises(RuntimeError): summarize(item)


if __name__ == '__main__': unittest.main()
