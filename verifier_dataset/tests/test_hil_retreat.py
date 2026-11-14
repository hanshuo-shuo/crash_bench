"""Scripted retreat of the HIL runner (docs/generative_verifier/HIL_RETREAT_PROTOCOL.md): the pure pose-error-to-action
functions, the outcome read on the policy's part, the runner's scripted loop on a fake environment, and a guard that the
stage 3 PUSH script is unchanged."""
import math
import unittest

import numpy as np

from verifier_dataset import hil_suite as S
from verifier_dataset import hil_suite_run as R

P = S.RETREAT_PARAMS
IDENTITY = [0., 0., 0., 1.]


def about(axis, angle):
    n = math.sqrt(sum(x * x for x in axis)); s = math.sin(angle / 2) / n
    return [axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2)]


def times(a, b):
    """Quaternion product a * b (x, y, z, w): rotation b, then a."""
    x1, y1, z1, w1 = a; x2, y2, z2, w2 = b
    return [w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2]


class Rotation(unittest.TestCase):
    def near(self, a, b):
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, places=9)

    def test_identity(self):
        self.near(S.rotation_error(IDENTITY, IDENTITY), [0., 0., 0.])
        q = about([1, 2, 3], .7); self.near(S.rotation_error(q, q), [0., 0., 0.])

    def test_world_frame_delta(self):
        # target = R(delta) @ current, as the controller composes it: the error is delta itself, whatever current is.
        current = about([1, -1, 2], 1.1)
        for axis, angle in (([0, 0, 1], .3), ([1, 0, 0], -.8), ([1, 1, 0], 1.5)):
            delta = [angle * x / math.sqrt(sum(a * a for a in axis)) for x in axis]
            self.near(S.rotation_error(times(about(axis, angle), current), current), delta)

    def test_short_way_round(self):
        error = S.rotation_error(about([0, 0, 1], 3.5), IDENTITY)          # 3.5 rad one way is 2 pi - 3.5 the other
        self.near(error, [0., 0., 3.5 - 2 * math.pi])
        self.near(S.rotation_error([-x for x in about([0, 0, 1], .4)], IDENTITY), [0., 0., .4])    # q and -q are one rotation


class Action(unittest.TestCase):
    home = [0., 0., 1.2]
    anchor = [.3, .2, 1.]

    def script(self, condition, opened=True, quat=IDENTITY, params=None):
        """A retreat script whose switch step is at self.anchor with orientation quat, by default past its open steps."""
        script = S.Retreat(condition, self.home, IDENTITY, self.anchor, quat, params=params)
        for _ in range((params or P)['open_steps'] if opened else 0):
            self.assertEqual(script.step(self.anchor, quat), ([0.] * 6 + [-1.], 'open'))
        return script

    def test_opens_first_and_holds_the_switch_pose(self):
        for condition in S.RETREAT:
            self.assertNotEqual(self.script(condition).step(self.anchor, IDENTITY)[1], 'open')
        s = self.script('HOME', opened=False)
        action, status = s.step([.31, .2, .98], about([0, 0, 1], .1))                                           # drifted off the switch pose
        self.assertEqual(status, 'open')
        for got, want in zip(action, [-.2, 0., .4, 0., 0., -.2, -1.]):                                          # pulled back to it, gripper opening
            self.assertAlmostEqual(got, want)
        # Even at the start pose the gripper is opened first.
        self.assertEqual(S.Retreat('HOME', self.home, IDENTITY, self.home, IDENTITY).step(self.home, IDENTITY), ([0.] * 6 + [-1.], 'open'))

    def test_backs_off_the_way_it_came_while_opening(self):
        for condition in S.RETREAT:
            s = S.Retreat(condition, self.home, IDENTITY, self.anchor, IDENTITY, [.26, .23, .98])
            action, status = s.step(self.anchor, IDENTITY)
            self.assertEqual(status, 'open')
            for got, want in zip(action, [-.8, .6, -.4, 0., 0., 0., -1.]):
                self.assertAlmostEqual(got, want)
            for _ in range(P['open_steps'] - 1):
                self.assertEqual(s.step([.26, .23, .98], IDENTITY), ([0.] * 6 + [-1.], 'open'))                   # there: wait for the gripper
            self.assertEqual(s.step([.26, .23, .98], IDENTITY)[1], 'lift')
        # RELEASE still measures its rise from the switch height, not from where it backed off to.
        s = S.Retreat('RELEASE', self.home, IDENTITY, self.anchor, IDENTITY, [.26, .23, .98], dict(P, open_steps=0))
        self.assertAlmostEqual(s.step([.26, .23, .98], IDENTITY)[0][2], 1.)
        self.assertEqual(s.step([.26, .23, 1.095], IDENTITY), (None, 'reached'))

    def test_release_rises_then_stops(self):
        s = self.script('RELEASE')
        self.assertEqual(s.step([.3, .2, 1.], IDENTITY), ([0., 0., 1., 0., 0., 0., -1.], 'lift'))               # 10 cm to go: capped at 1
        action, status = s.step([.3, .2, 1.08], about([1, 0, 0], .05))
        self.assertEqual(status, 'lift'); self.assertAlmostEqual(action[2], .4); self.assertEqual(action[:2], [0., 0.])
        self.assertAlmostEqual(action[3], -.1); self.assertEqual(action[6], -1.)                                # the switch-step orientation is held
        self.assertEqual(s.step([.3, .2, 1.0899], IDENTITY)[1], 'lift')
        self.assertEqual(s.step([.3, .2, 1.0901], IDENTITY), (None, 'reached'))
        s = self.script('RELEASE')
        for _ in range(P['release_steps'] - P['open_steps']):
            self.assertEqual(s.step([.3, .2, 1.05], IDENTITY)[1], 'lift')                                       # stuck 5 cm up
        self.assertEqual(s.step([.3, .2, 1.05], IDENTITY), (None, 'cap'))

    def test_home_lifts_travels_then_settles(self):
        turned = about([0, 0, 1], .5); s = self.script('HOME', quat=turned)
        self.assertEqual(s.step([.3, .2, 1.], turned), ([0., 0., 1., 0., 0., 0., -1.], 'lift'))                 # no sideways motion, no rotation
        action, status = s.step([.3, .2, 1.19], about([0, 0, 1], .6))                                           # within the lift tolerance
        self.assertEqual(status, 'travel'); self.assertEqual(action[2:5] + action[6:], [0., 0., 0., -1.])       # level
        self.assertAlmostEqual(action[5], -.2)                                                                  # back to the switch-step orientation, not home's
        self.assertAlmostEqual(action[0] / action[1], 1.5); self.assertAlmostEqual(max(abs(x) for x in action[:2]), 1.)   # direction kept
        action, status = s.step([.01, -.01, 1.19], turned)                                                      # above the start position
        self.assertEqual(status, 'settle')
        for got, want in zip(action, [-.2, .2, .2, 0., 0., -1., -1.]):                                          # now to the start orientation
            self.assertAlmostEqual(got, want)
        self.assertEqual(s.step([.3, .2, 1.], turned)[1], 'settle')                                             # stages never go back
        self.assertEqual(s.stalled, [])

    def test_home_above_the_start_height_travels_level_first(self):
        s = S.Retreat('HOME', self.home, IDENTITY, [.3, 0., 1.35], IDENTITY, params=dict(P, open_steps=0))
        self.assertEqual(s.step([.3, 0., 1.35], IDENTITY), ([-1., 0., 0., 0., 0., 0., -1.], 'travel'))          # no descent until it is back

    def test_a_stalled_stage_hands_over(self):
        s = self.script('HOME'); w = P['stall_window']
        for _ in range(w):
            self.assertEqual(s.step([.3, .2, 1.], IDENTITY)[1], 'lift')                                         # caught: no height gained
        self.assertEqual(s.step([.3, .2, 1.004], IDENTITY)[1], 'travel'); self.assertEqual(s.stalled, ['lift'])  # under 5 mm in five steps
        for k in range(w):
            self.assertEqual(s.step([.3 - .002 * k, .2, 1.004], IDENTITY)[1], 'travel')
        self.assertEqual(s.step([.3, .2, 1.004], IDENTITY)[1], 'settle'); self.assertEqual(s.stalled, ['lift', 'travel'])
        moving = self.script('HOME')
        for k in range(30):
            self.assertEqual(moving.step([.3, .2, 1. + .002 * k], IDENTITY)[1], 'lift')                         # slow but moving: 1 cm in five steps
        self.assertEqual(moving.stalled, [])

    def test_home_reached_needs_both_tolerances(self):
        self.assertEqual(self.script('HOME').step([.01, 0., 1.2], IDENTITY), (None, 'reached'))
        self.assertEqual(self.script('HOME').step([.01, 0., 1.2], about([0, 0, 1], .2))[1], 'settle')            # orientation still off
        self.assertEqual(self.script('HOME').step([.03, 0., 1.2], IDENTITY)[1], 'travel')                        # position still off
        late = dict(P, home_steps=P['open_steps'])
        self.assertEqual(self.script('HOME', params=late).step([.3, 0., 1.2], IDENTITY), (None, 'cap'))
        self.assertEqual(self.script('HOME', params=late).step([.01, 0., 1.2], IDENTITY), (None, 'reached'))     # reached wins at the cap

    def test_actions_stay_in_range_and_follow_the_parameters(self):
        for eef in ([.6, -.5, .9], [-.4, .4, 1.6], [.02, .0, 1.19]):
            for quat in (IDENTITY, about([1, 1, 1], 2.5)):
                for condition in S.RETREAT:
                    s = S.Retreat(condition, self.home, IDENTITY, self.anchor, IDENTITY)
                    for _ in range(P['open_steps'] + 12):
                        action, _ = s.step(eef, quat)
                        if action is not None:
                            self.assertTrue(all(abs(x) <= 1 for x in action)); self.assertEqual(len(action), 7); self.assertEqual(action[6], -1.)
        slow = dict(P, max_action=.5, open_steps=2)
        action, status = self.script('HOME', params=slow).step([.3, .2, 1.19], IDENTITY)
        self.assertEqual(status, 'travel'); self.assertAlmostEqual(max(abs(x) for x in action[:3]), .5)

    def test_not_a_stage_3_perturbation(self):
        self.assertFalse(set(S.RETREAT) & set(S.PERTURB))


def step(t, eef=(0., 0., 1.), a=False, b=False, scripted=False):
    out = dict(step=t, eef=list(eef), a=a, others={'1': b}, done=a)
    if scripted:
        out['scripted'] = True
    return out


class Outcome(unittest.TestCase):
    unit = dict(other=1)
    start = [0., 0., 1.2]

    def test_reads_the_policy_part_only(self):
        trace = [step(50)] + [step(51 + i, scripted=True) for i in range(20)] + [step(71 + i) for i in range(30)] + [step(101, b=True)]
        o = S.retreat_outcome(trace, self.unit, self.start)
        self.assertEqual((o['scripted_steps'], o['b_done'], o['b_after'], o['steps']), (20, True, 31, 31))
        self.assertEqual((o['b_at_checkpoint'], o['b_at_switch'], o['b_in_script'], o['a_done'], o['a_at_handover']), (False, False, False, False, False))

    def test_b_at_the_handover_is_flagged(self):
        trace = [step(50)] + [step(51, scripted=True), step(52, scripted=True, b=True)] + [step(53, b=True)]
        o = S.retreat_outcome(trace, self.unit, self.start)
        self.assertTrue(o['b_at_checkpoint'] and o['b_in_script'] and o['b_done']); self.assertEqual(o['scripted_steps'], 2)

    def test_b_only_during_the_script_is_not_the_policys(self):
        trace = [step(50)] + [step(51, scripted=True, b=True), step(52, scripted=True)] + [step(53 + i) for i in range(5)]
        o = S.retreat_outcome(trace, self.unit, self.start)
        self.assertEqual((o['b_done'], o['b_at_checkpoint'], o['b_in_script']), (False, False, True))

    def test_a_finished_after_the_handover(self):
        trace = [step(50)] + [step(51, scripted=True, a=True)] + [step(52, a=True)]
        o = S.retreat_outcome(trace, self.unit, self.start)
        self.assertTrue(o['a_at_handover'] and o['a_done'])

    def test_without_a_script_it_is_the_plain_outcome(self):
        trace = [step(50)] + [step(51 + i) for i in range(5)] + [step(56, b=True)]
        o = S.retreat_outcome(trace, self.unit, self.start)
        self.assertEqual(o['scripted_steps'], 0)
        self.assertEqual({k: o[k] for k in S.outcome(trace, self.unit, self.start)}, S.outcome(trace, self.unit, self.start))


class Sim:
    def __init__(self, bodies):
        self.data = type('D', (), {})(); self.data.body_xpos = bodies; self.model = None


class Inner:
    """What the runner reads off env.env."""
    action_spec = ([-1.] * 7, [1.] * 7)
    parsed_problem = dict(goal_state=[['on', 'bowl_1', 'plate_1']])
    fixtures_dict = {}

    def __init__(self):
        self.objects_dict = dict(bowl_1=None, plate_1=None); self.obj_body_id = dict(bowl_1=0, plate_1=1); self.success = False

    def _check_success(self):
        return self.success


class Env:
    """A kinematic stand-in for the OSC_POSE controller: each step the end effector covers `rate` of the commanded delta
    (5 cm or 0.5 rad per unit of action), the rotation delta applied in the world frame, as the controller does."""
    def __init__(self, eef, quat, rate=.5, caught=None):
        self.eef = np.asarray(eef, float); self.quat = list(quat); self.rate = rate; self.env = Inner(); self.actions = []; self.caught = caught
        self.sim = Sim(np.array([[.1, .1, .9], [.3, .0, .9]])); self.dropped = False

    def obs(self):
        return dict(robot0_eef_pos=self.eef.copy(), robot0_eef_quat=np.asarray(self.quat), robot0_gripper_qpos=np.array([.04, -.04]))

    def step(self, action):
        action = np.asarray(action, float); self.actions.append(action.tolist())
        move = self.rate * .05 * action[:3]
        if self.caught is not None and self.eef[0] > self.caught:    # caught on something: no rising until it has backed off in x
            move[2] = 0.
        self.eef = self.eef + move
        delta = self.rate * .5 * action[3:6]; angle = float(np.linalg.norm(delta))
        if angle > 0:
            self.quat = times(about(delta.tolist(), angle), self.quat)
        if action[6] < 0 and not self.dropped:                       # opening the gripper drops the bowl 4 cm
            self.sim.data.body_xpos[0] = self.sim.data.body_xpos[0] + [0., 0., -.04]; self.dropped = True
        return self.obs(), 0., False, {}


HASH = lambda env: 'h%d' % len(env.actions)
# A prefix tape whose arm came along +x: 2 cm a step, ending at the switch position of the tests below.
TAPE = [dict(step=t, eef=[.1 - .02 * (60 - t), .25, .95]) for t in range(61)]


class Runner(unittest.TestCase):
    home = dict(eef=[-.2, 0., 1.17], quat=about([1, 0, 0], math.pi))

    def test_home_script_returns_to_the_start_pose(self):
        env = Env([.1, .25, .95], times(about([0, 0, 1], .9), self.home['quat']))
        unit = dict(condition='HOME', checkpoint=60, other=1)
        obs, info, scripted = R.retreat(env, unit, env.obs(), self.home, TAPE, HASH, {1: lambda: False})
        self.assertTrue(info['reached']); self.assertEqual(info['status'], 'reached')
        self.assertLess(info['position_error'], P['pos_tolerance']); self.assertLess(info['orientation_error'], P['ori_tolerance'])
        self.assertEqual(info['steps'], len(scripted)); self.assertLess(len(scripted), P['home_steps'])
        self.assertEqual(info['stalled'], []); self.assertEqual(sum(info['stage_steps'].values()), len(scripted))
        self.assertEqual(info['stage_steps']['open'], P['open_steps']); self.assertTrue(all(info['stage_steps'][n] > 0 for n in ('lift', 'travel', 'settle')))
        self.assertEqual([x['step'] for x in scripted], list(range(61, 61 + len(scripted))))
        self.assertTrue(all(x['scripted'] and x['others'] == {'1': False} and x['action'][6] == -1. for x in scripted))
        self.assertTrue(all(abs(v) <= 1 for x in scripted for v in x['action']))
        self.assertEqual(info['eef_back'], [.1 - .02 * P['back_steps'], .25, .95])
        self.assertEqual(scripted[0]['action'], [-1., 0., 0., 0., 0., 0., -1.])                              # back the way it came, opening
        self.assertTrue(all(x['action'][1:6] == [0.] * 5 and x['action'][0] <= 0 for x in scripted[:P['open_steps']]))
        self.assertTrue(all(x['action'][3:6] == [0., 0., 0.] for x in scripted[:info['steps'] - info['stage_steps']['settle']]))   # no turning before it settles
        self.assertAlmostEqual(info['moved']['bowl_1'], .04); self.assertEqual(info['moved']['plate_1'], 0.)
        self.assertEqual(info['eef_from'], [.1, .25, .95]); np.testing.assert_allclose(obs['robot0_eef_pos'], info['eef_to'])

    def test_home_script_backs_off_when_the_lift_is_caught(self):
        env = Env([.1, .25, .95], self.home['quat'], caught=0.)    # as with the gripper around a drawer handle
        obs, info, scripted = R.retreat(env, dict(condition='HOME', checkpoint=60, other=1), env.obs(), self.home, TAPE, HASH, {1: lambda: False})
        self.assertTrue(info['reached']); self.assertEqual(info['stalled'], ['lift'])
        self.assertEqual(info['stage_steps']['lift'], P['stall_window'])

    def test_home_script_stops_at_the_cap(self):
        env = Env([.1, .25, .95], self.home['quat'], rate=.02)     # far too slow to arrive
        obs, info, scripted = R.retreat(env, dict(condition='HOME', checkpoint=60, other=1), env.obs(), self.home, TAPE, HASH, {1: lambda: False})
        self.assertEqual((info['status'], info['reached'], len(scripted)), ('cap', False, P['home_steps']))

    def test_release_script_rises_ten_centimetres(self):
        env = Env([.1, .25, .95], self.home['quat'])
        obs, info, scripted = R.retreat(env, dict(condition='RELEASE', checkpoint=60, other=1), env.obs(), self.home, TAPE, HASH, {1: lambda: True})
        self.assertTrue(info['reached']); self.assertGreaterEqual(info['lifted'], P['release_lift'] - P['release_tolerance']); self.assertLess(len(scripted), P['release_steps'])
        self.assertTrue(all(x['action'][:2] == [0., 0.] and x['action'][3:6] == [0., 0., 0.] for x in scripted[P['open_steps']:]))
        self.assertTrue(scripted[-1]['others']['1'])               # the goal checks ride on every scripted record
        self.assertAlmostEqual(obs['robot0_eef_pos'][0], .1 - .02 * P['back_steps'], places=3)                 # it rose from where it backed off to

    def test_parameters_can_be_overridden_for_the_dry_run(self):
        env = Env([.1, .25, .95], self.home['quat'])
        _, info, scripted = R.retreat(env, dict(condition='HOME', checkpoint=60, other=1), env.obs(), self.home, TAPE, HASH, {1: lambda: False},
                                      dict(P, open_steps=3, max_action=.5))
        self.assertEqual(sum(x['action'][1:3] == [0., 0.] and x['action'][0] < 0 for x in scripted[:4]), 3)   # three open steps, then the lift
        self.assertLessEqual(max(abs(v) for x in scripted for v in x['action'][:6]), .5 + 1e-12)

    def test_stage_3_push_script_is_unchanged(self):
        """The PUSH script as it stood before the retreat was added: 15 steps toward eef + PUSH_OFFSET, the error over
        PUSH_GAIN clipped to the action range, the gripper command carried over, records without goal checks."""
        env = Env([.1, .25, .95], self.home['quat'], rate=.4); last = [0.] * 6 + [.7]
        obs, info, scripted = R.perturb(env, dict(condition='PUSH', checkpoint=60), env.obs(), {}, last, HASH)
        eef = np.array([.1, .25, .95]); target = eef + np.asarray(S.PUSH_OFFSET); want = []
        for _ in range(S.PUSH_STEPS):
            action = np.clip(np.concatenate([(target - eef) / S.PUSH_GAIN, np.zeros(3), [.7]]), -1., 1.); want.append(action.tolist())
            eef = eef + .4 * .05 * action[:3]
        self.assertEqual([x['action'] for x in scripted], want)
        self.assertEqual(len(scripted), 15); self.assertTrue(all(x['scripted'] and x['others'] == {} for x in scripted))
        self.assertEqual([x['step'] for x in scripted], list(range(61, 76)))
        self.assertTrue(info['applied']); self.assertAlmostEqual(info['displacement'], float(np.linalg.norm(eef - np.array([.1, .25, .95]))))


if __name__ == '__main__':
    unittest.main()
