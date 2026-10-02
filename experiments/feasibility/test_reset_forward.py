import unittest
from reset_forward import replay_renderer_reset_forward


class ResetForwardTests(unittest.TestCase):
    def base(self):
        class Base:
            has_renderer = False
            has_offscreen_renderer = False

            def __init__(self):
                self.events = []
                self.sim = type('Sim', (), {'forward': lambda _: self.events.append('forward')})()

            def _reset_internal(self):
                self.events.append('base_reset')
                return 'original_result'
        return Base

    def test_forward_occurs_at_original_base_reset_point_without_a_step_or_state_write(self):
        base = self.base()
        record = replay_renderer_reset_forward(base)
        env = base()
        self.assertEqual(env._reset_internal(), 'original_result')
        self.assertEqual(env.events, ['forward', 'base_reset'])
        self.assertEqual(record['calls'], 1)
        with self.assertRaises(RuntimeError):
            replay_renderer_reset_forward(base)

    def test_rendered_environment_cannot_receive_duplicate_constructor_forward(self):
        base = self.base()
        replay_renderer_reset_forward(base)
        env = base(); env.has_offscreen_renderer = True
        with self.assertRaises(RuntimeError):
            env._reset_internal()
        self.assertEqual(env.events, [])


if __name__ == '__main__':
    unittest.main()
