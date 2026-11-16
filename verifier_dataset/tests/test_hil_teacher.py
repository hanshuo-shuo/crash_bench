import unittest

import numpy as np

from verifier_dataset import hil_suite as S
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
        self.assertEqual(same['image'], dict(match=True, linf=0, changed=0)); self.assertEqual(same['wrist_image']['linf'], 0)
        noise = stored['observation/wrist_image'].copy(); noise[0, 0, 0] += 2
        self.assertEqual(T.input_check(o, dict(stored, **{'observation/wrist_image': noise}))['wrist_image'], dict(match=True, linf=2, changed=1))
        other = stored['observation/wrist_image'] + 3
        self.assertEqual(T.input_check(o, dict(stored, **{'observation/wrist_image': other}))['wrist_image'], dict(match=False, linf=3, changed=224 * 224 * 3))


def plant(condition, held, lift=.1, floor=0.):
    """An ideal arm: each step it covers half of the commanded position error; a held object follows until let go and cannot
    go below floor (something under it)."""
    eef = [.1, .2, .3]; q = [0., 0., 0., 1.]; script = S.Instruct(condition, [0., 0., .5], q, eef, q, [.1, .2, .28], 1. if held else -1., held)
    lift = lift if held else None; stages = []; actions = []
    while True:
        action, stage = script.step(eef, q, lift)
        if action is None:
            return script, stage, stages, actions, eef
        move = [x * S.RETREAT_PARAMS['pos_gain'] * .5 for x in action[:3]]
        if held and action[6] > 0:
            move[2] = max(move[2], floor - lift); lift += move[2]
        eef = [e + m for e, m in zip(eef, move)]; stages.append(stage); actions.append(action)


class Scripts(unittest.TestCase):
    def test_stop_is_zero_commands_with_the_gripper_left_as_it_was(self):
        _, status, stages, actions, eef = plant('STOP', True)
        self.assertEqual((status, len(actions), eef), ('still', S.INSTRUCT_PARAMS['stop_steps'], [.1, .2, .3]))
        self.assertTrue(all(a == [0.] * 6 + [1.] for a in actions))
        self.assertTrue(all(a[6] == -1. for a in plant('STOP', False)[3]))

    def test_return_is_the_home_retreat_then_zero_commands(self):
        _, status, stages, actions, eef = plant('RETURN', True)
        n = S.INSTRUCT_PARAMS['after_steps']
        self.assertEqual(status, 'reached'); self.assertEqual(stages[-n:], ['still'] * n); self.assertNotIn('still', stages[:-n])
        self.assertTrue(all(a == [0.] * 6 + [-1.] for a in actions[-n:]))
        self.assertLess(sum((a - b) ** 2 for a, b in zip(eef, [0., 0., .5])) ** .5, S.RETREAT_PARAMS['pos_tolerance'])

    def test_cancel_lowers_what_it_holds_before_letting_go(self):
        script, status, stages, actions, _ = plant('CANCEL', True)
        self.assertEqual(status, 'lowered'); self.assertLessEqual(script.release_lift, S.RESTING)
        self.assertEqual([k for k in dict.fromkeys(stages)], ['lower', 'open', 'rise', 'still'])
        self.assertTrue(all(a[6] == 1. for a, k in zip(actions, stages) if k == 'lower'))
        self.assertTrue(all(abs(a[2]) <= S.INSTRUCT_PARAMS['lower_action'] for a, k in zip(actions, stages) if k == 'lower'))

    def test_cancel_lets_go_where_the_object_stops_descending(self):
        script, status, _, _, _ = plant('CANCEL', True, lift=.2, floor=.12)
        self.assertEqual(status, 'stalled'); self.assertAlmostEqual(script.release_lift, .12)

    def test_cancel_with_nothing_held_opens_and_stays(self):
        _, status, stages, _, _ = plant('CANCEL', False)
        self.assertEqual((status, list(dict.fromkeys(stages))), ('nothing_held', ['open', 'still']))

    def test_valid(self):
        info = dict(status='reached', arm_path=.001, held='bowl', drop=.01, object_path=0.)
        self.assertTrue(S.instruct_valid('STOP', dict(stopped=True), info)); self.assertFalse(S.instruct_valid('STOP', dict(stopped=False), info))
        self.assertTrue(S.instruct_valid('RETURN', dict(home=True), info))
        self.assertFalse(S.instruct_valid('RETURN', dict(home=True), dict(info, status='cap')))
        self.assertTrue(S.instruct_valid('CANCEL', dict(a_done=False), info))
        self.assertFalse(S.instruct_valid('CANCEL', dict(a_done=True), info))                    # putting it down finished the old task
        self.assertFalse(S.instruct_valid('CANCEL', dict(a_done=False), dict(info, drop=.09)))   # dropped, not put down
        self.assertTrue(S.instruct_valid('CANCEL', dict(a_done=False), dict(info, held=None, drop=None, object_path=None)))


class Deal(unittest.TestCase):
    def test_every_episode_goes_to_exactly_one_process(self):
        for shards, workers in ((1, 2), (2, 6)):
            for index in range(50):
                owners = [(s, w) for s in range(shards) for w in range(workers) if T.mine(index, s, shards, w, workers)]
                self.assertEqual(len(owners), 1)


if __name__ == '__main__':
    unittest.main()
