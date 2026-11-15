import unittest

import numpy as np

import teacher as T
import train_switch as W


def unit(sw, task, other, init, phase=.5):
    return dict(sw=sw, task=task, other=other, init=init, phase=phase, prefix='p%d_%d' % (task, init), suite='libero_goal', checkpoint=70,
                prompt='prompt %d' % other, pair='%d->%d' % (task, other), covariates=dict(same_object=False))


class Select(unittest.TestCase):
    def setUp(self):
        self.units = [unit('a', 0, 3, 0), unit('b', 0, 3, 1), unit('c', 0, 3, 2), unit('d', 0, 3, 3), unit('e', 4, 1, 0), unit('f', 1, 4, 9)]
        self.done = dict(home=dict(a=True, b=False, c=False, d=False, e=True, f=True), native=dict(a=True, b=True, c=False, d=False, e=False, f=False),
                         release=dict(a=True, b=True, c=True, d=False, e=False, f=False))
        self.folders = dict(home='/r/branch', release='/r/branch', native='/c/branch')

    def test_split(self):
        self.assertEqual([T.split(4, 0), T.split(7, 9), T.split(1, 9), T.split(1, 8)], ['test', 'test', 'dev', 'train'])

    def test_the_retreat_first_then_a_stored_success_without_it(self):
        rows = {r['sw']: r for r in T.select(self.units, self.done, self.folders)}
        self.assertEqual([rows[k]['arm'] for k in 'abcd'], ['home', 'native', 'release', None])
        self.assertEqual(rows['a']['source'], '/r/branch/a__home'); self.assertEqual(rows['b']['source'], '/c/branch/b__native')
        self.assertIsNone(rows['d']['source'])
        self.assertEqual([rows[k]['split'] for k in 'aef'], ['train', 'test', 'dev'])
        self.assertEqual(rows['a']['prompt'], 'prompt 3')

    def test_counts(self):
        c = T.counts(T.select(self.units, self.done, self.folders))
        self.assertEqual(c['train'], dict(states=4, home=1, native=1, release=1, none=1))
        self.assertEqual(c['test'], dict(states=1, home=1)); self.assertEqual(c['dev'], dict(states=1, home=1))


class Export(unittest.TestCase):
    def meta(self, name, linf=0, passed=True):
        check = dict(image=dict(linf=linf, changed=0.), wrist_image=dict(linf=0, changed=0.))
        return dict(id=name, passed=passed, frames=10, scripted=4, clipped=1, input_check=check) if passed else dict(id=name, passed=False, error='x')

    def test_reading(self):
        plan = dict(episodes=[dict(id='a'), dict(id='b')])
        ok = T.export_reading(plan, dict(a=self.meta('a'), b=self.meta('b')))
        self.assertTrue(ok['passed']); self.assertEqual((ok['frames'], ok['scripted_frames'], ok['clipped_steps']), (20, 8, 2))
        self.assertFalse(T.export_reading(plan, dict(a=self.meta('a')))['passed'])                       # one missing
        self.assertFalse(T.export_reading(plan, dict(a=self.meta('a'), b=self.meta('b', passed=False)))['passed'])
        changed = T.export_reading(plan, dict(a=self.meta('a'), b=self.meta('b', linf=2)))
        self.assertFalse(changed['passed']); self.assertEqual((changed['image_linf_max'], changed['image_unequal']), (2, 1))

    def test_features_are_those_of_the_libero_dataset(self):
        f = T.features()
        self.assertEqual({k: (v['dtype'], tuple(v['shape'])) for k, v in f.items()},
                         dict(image=('image', (256, 256, 3)), wrist_image=('image', (256, 256, 3)), state=('float32', (8,)), actions=('float32', (7,))))


class Mix(unittest.TestCase):
    def test_pick_is_deterministic_and_follows_the_shares(self):
        picks = [W.pick(i, [100, 10], [.25, .75], 0) for i in range(4000)]
        self.assertEqual(picks, [W.pick(i, [100, 10], [.25, .75], 0) for i in range(4000)])
        self.assertNotEqual(picks, [W.pick(i, [100, 10], [.25, .75], 1) for i in range(4000)])
        share = np.mean([k for k, _ in picks])
        self.assertTrue(.72 < share < .78)
        self.assertTrue(all(0 <= j < (100, 10)[k] for k, j in picks))
        self.assertEqual({j for k, j in picks if k == 1}, set(range(10)))

    def test_mix_draws_from_its_parts(self):
        m = W.Mix([['a0', 'a1'], ['b0']], [.5, .5], 6, 3)
        self.assertEqual(len(m), 6)
        self.assertEqual([m[i] for i in range(6)], [(['a0', 'a1'], ['b0'])[k][j] for k, j in (W.pick(i, [2, 1], [.5, .5], 3) for i in range(6))])
        with self.assertRaises(ValueError):
            W.Mix([[1]], [.5], 1, 0)

    def test_settings(self):
        s = dict(name='x', base_checkpoint='/c', datasets=[dict(repo_id='a', root='/a', share=.25), dict(repo_id='b', root='/b', share=.75)],
                 batch_size=32, steps=10, peak_lr=1e-5, warmup_steps=1, ema_decay=None, save_interval=5, seed=0, num_workers=2)
        self.assertEqual(W.settings_ok(s), s)
        with self.assertRaises(ValueError):
            W.settings_ok(dict(s, datasets=[dict(repo_id='a', root='/a', share=.5)]))
        with self.assertRaises(ValueError):
            W.settings_ok({k: v for k, v in s.items() if k != 'seed'})


if __name__ == '__main__':
    unittest.main()
