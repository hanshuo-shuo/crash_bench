import unittest

import numpy as np

from verifier_dataset import hil_teacher as T
from verifier_dataset.paired_policy import policy_input


def obs(quat):
    return dict(robot0_eef_pos=np.array([.1, -.2, 1.1]), robot0_eef_quat=np.array(quat), robot0_gripper_qpos=np.array([.03, -.03]),
                agentview_image=np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3),
                robot0_eye_in_hand_image=np.zeros((256, 256, 3), np.uint8))


class Label(unittest.TestCase):
    def test_arm_is_clipped_and_gripper_is_a_sign(self):
        self.assertEqual(T.label([1.5, -2., .25, 0., -.5, 1., .97]), [1., -1., .25, 0., -.5, 1., 1.])
        self.assertEqual(T.label([0.] * 6 + [-1.02])[6], -1.)
        self.assertEqual(T.label([0.] * 6 + [0.])[6], 0.)


class Frame(unittest.TestCase):
    def test_state_is_the_policy_input_state(self):
        for quat in ([.6, .1, -.3, .73], [0., 0., 0., 1.], [1., 0., 0., 0.]):
            o = obs(quat)
            self.assertTrue(np.array_equal(T.state(o), policy_input(o, 'x')['observation/state']))

    def test_images_are_rotated_half_a_turn(self):
        o = obs([0., 0., 0., 1.]); f = T.frame(o)
        self.assertEqual(f['image'].shape, (256, 256, 3))
        self.assertTrue(np.array_equal(f['image'][0, 0], o['agentview_image'][255, 255]))
        self.assertTrue(np.array_equal(f['image'][10, 3], o['agentview_image'][245, 252]))

    def test_input_check_reports_equal_and_changed_inputs(self):
        o = obs([.6, .1, -.3, .73]); stored = {k: v for k, v in policy_input(o, 'x').items() if k != 'prompt'}
        same = T.input_check(o, stored)
        self.assertTrue(same['state_equal'] and same['state_matches_frame'])
        self.assertEqual((same['image']['linf'], same['wrist_image']['linf']), (0, 0))
        stored['observation/wrist_image'] = stored['observation/wrist_image'] + 3
        self.assertEqual(T.input_check(o, stored)['wrist_image'], dict(linf=3, changed=1.))


class Deal(unittest.TestCase):
    def test_every_episode_goes_to_exactly_one_process(self):
        for shards, workers in ((1, 2), (2, 6)):
            for index in range(50):
                owners = [(s, w) for s in range(shards) for w in range(workers) if T.mine(index, s, shards, w, workers)]
                self.assertEqual(len(owners), 1)


if __name__ == '__main__':
    unittest.main()
