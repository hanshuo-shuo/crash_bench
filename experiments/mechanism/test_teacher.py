import unittest

import numpy as np

import launch_train as L
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
        self.assertEqual(rows['a']['prompt'], 'prompt 3'); self.assertEqual({r['kind'] for r in rows.values()}, {'switch'})

    def test_keep_episodes_are_the_training_prefixes(self):
        prefixes = [dict(id='p%d_%d' % (t, i), suite='libero_goal', task=t, init=i, prompt='do %d' % t) for t in (1, 4) for i in (0, 9)]
        keep = T.keep_episodes(prefixes, '/prefix_root', 10)
        self.assertEqual([e['id'] for e in keep], ['p1_0_KEEP'])
        e = keep[0]
        self.assertEqual((e['kind'], e['source'], e['prompt'], e['checkpoint'], e['other'], e['split']),
                         ('keep', '/prefix_root/prefix/p1_0', 'do 1', 10, None, 'train'))

    def test_counts(self):
        c = T.counts(T.select(self.units, self.done, self.folders))
        self.assertEqual(c['train'], dict(states=4, home=1, native=1, release=1, none=1))
        self.assertEqual(c['test'], dict(states=1, home=1)); self.assertEqual(c['dev'], dict(states=1, home=1))


class Export(unittest.TestCase):
    def meta(self, name, linf=0, passed=True):
        check = dict(image=dict(match=True, linf=linf, changed=int(linf > 0)), wrist_image=dict(match=True, linf=0, changed=0))
        return dict(id=name, passed=passed, frames=10, scripted=4, clipped=1, input_check=check) if passed else dict(id=name, passed=False, error='x')

    def test_reading(self):
        plan = dict(episodes=[dict(id='a'), dict(id='b')])
        ok = T.export_reading(plan, dict(a=self.meta('a'), b=self.meta('b')))
        self.assertTrue(ok['passed']); self.assertEqual((ok['frames'], ok['scripted_frames'], ok['clipped_steps']), (20, 8, 2))
        self.assertFalse(T.export_reading(plan, dict(a=self.meta('a')))['passed'])                       # one missing
        self.assertFalse(T.export_reading(plan, dict(a=self.meta('a'), b=self.meta('b', passed=False)))['passed'])
        noise = T.export_reading(plan, dict(a=self.meta('a'), b=self.meta('b', linf=2)))                 # rendering noise is reported
        self.assertTrue(noise['passed']); self.assertEqual((noise['image_linf_max'], noise['image_not_identical']), (2, 1))

    def test_an_invalid_scripted_run_is_left_out_and_is_not_a_failure(self):
        plan = dict(episodes=[dict(id='a'), dict(id='b')])
        script = lambda name, valid: dict(id=name, passed=True, valid=valid, frames=80, scripted=80, clipped=0)
        r = T.export_reading(plan, dict(a=script('a', True), b=script('b', False)))
        self.assertTrue(r['passed']); self.assertEqual((r['exported'], r['invalid'], r['frames'], r['image_linf_max']), (1, 1, 80, None))

    def test_features_are_those_of_the_libero_dataset(self):
        f = T.features()
        self.assertEqual({k: (v['dtype'], tuple(v['shape'])) for k, v in f.items()},
                         dict(image=('image', (256, 256, 3)), wrist_image=('image', (256, 256, 3)), state=('float32', (8,)), actions=('float32', (7,))))


class Instructions(unittest.TestCase):
    def test_one_episode_per_switch_step_and_instruction(self):
        units = [unit('p1_0_s070_SW_b%d' % b, 1, b, 0) for b in (2, 3)] + [unit('p4_0_s070_SW_b1', 4, 1, 0), unit('p1_9_s070_SW_b2', 1, 2, 9)]
        text = lambda c, a: '%s/%s' % (c, a)
        train = T.instruct_episodes(units, {1: 'one', 4: 'four'}, '/r', ('train',), ('STOP', 'RETURN'), text)
        self.assertEqual([e['id'] for e in train], ['p1_0_s070_STOP', 'p1_0_s070_RETURN'])           # two B's of one state: one episode each
        self.assertEqual([e['prompt'] for e in train], ['STOP/one', 'RETURN/one'])
        self.assertEqual([e['home_source'] for e in train], [None, '/r/branch/p1_0_s070_SW_b2__home'])
        other = T.instruct_episodes(units, {1: 'one', 4: 'four'}, '/r', ('dev', 'test'), ('STOP',), text)
        self.assertEqual([(e['prefix'], e['split']) for e in other], [('p1_9', 'dev'), ('p4_0', 'test')])

    def test_baseline_table(self):
        episodes = [dict(id=k, split='test', condition=c) for k, c in (('a', 'STOP'), ('b', 'STOP'), ('c', 'CANCEL'))]
        o = lambda stopped, a_done: dict(stopped=stopped, home=False, a_done=a_done)
        metas = dict(a=dict(valid=True, outcome=o(True, False), script=dict(steps=80)), b=dict(valid=False, outcome=o(False, False), script=dict(steps=80)),
                     c=dict(valid=False, outcome=o(False, False), script=dict(steps=50)))
        t = T.baseline_table(episodes, metas)['test']
        self.assertEqual(t['STOP'], dict(runs=2, valid=1, follow=1, steps_median=80))
        self.assertEqual(t['CANCEL'], dict(runs=1, valid=0, follow=1, steps_median=50))               # not finishing A is the suite's rule


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
                 batch_sizes=[64, 32], steps=10, peak_lr=1e-5, warmup_steps=1, ema_decay=None, save_interval=5, seed=0, num_workers=2)
        self.assertEqual(W.settings_ok(s), s)
        with self.assertRaises(ValueError):
            W.settings_ok(dict(s, datasets=[dict(repo_id='a', root='/a', share=.5)]))
        with self.assertRaises(ValueError):
            W.settings_ok({k: v for k, v in s.items() if k != 'seed'})
        with self.assertRaises(ValueError):
            W.settings_ok(dict(s, batch_sizes=[]))

    def test_only_an_out_of_memory_failure_moves_to_the_next_batch_size(self):
        self.assertTrue(W.out_of_memory('XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 1.2GiB'))
        self.assertFalse(W.out_of_memory('FileNotFoundError: no such dataset'))


class Launcher(unittest.TestCase):
    def test_every_training_run_is_well_formed(self):
        self.assertEqual(set(L.SETTINGS), set(L.DATA)); self.assertEqual(set(L.SETTINGS), set(L.KINDS))
        for name, s in L.SETTINGS.items():
            found = {kind: dict(repo_id=kind, root='/' + kind) for kind in L.DATA[name]}
            body = dict(s, name=name, base_checkpoint='/c', datasets=L.datasets(name, found))
            self.assertEqual(W.settings_ok(body), body)
            self.assertEqual(body['datasets'][0]['share'], .25)                                  # LIBERO's own demonstrations
            self.assertLessEqual(set(L.DATA[name]), set(L.TEACHER_MODE))
        self.assertEqual([d['share'] for d in L.datasets('t2', dict(switch={}, instruct={}))], [.25, .5, .25])
        self.assertEqual((L.per_state('t1'), L.per_state('t2')), (['keep'], ['keep', 'stop', 'return', 'cancel']))

    def test_checkpoints_are_where_openpi_saves_them(self):
        from pathlib import Path
        saved = L.checkpoints(Path('/run'), 't1')
        self.assertEqual(list(saved), ['1000', '1999'])
        self.assertEqual(str(saved['1999']), '/run/checkpoints/pi05_libero/t1/1999')

    def test_dependencies(self):
        self.assertEqual(L.after(), [])
        self.assertEqual(L.after('1', '2'), ['--dependency=afterok:1,afterok:2', '--kill-on-invalid-dep=yes'])
        self.assertEqual(L.after('1', ended=['1', '3']), ['--dependency=afterok:1,afterany:3', '--kill-on-invalid-dep=yes'])
        self.assertEqual(L.after(ended=['3']), ['--dependency=afterany:3', '--kill-on-invalid-dep=yes'])

    def test_longer_runs_differ_in_length_alone(self):
        from pathlib import Path
        self.assertEqual(list(L.checkpoints(Path('/run'), 't1b')), ['2000', '4000', '5999'])
        self.assertEqual(L.SETTINGS['t2'], L.SETTINGS['t1b'])
        same = {k for k in L.T1 if k not in ('steps', 'save_interval', 'batch_sizes')}
        self.assertTrue(all(L.SETTINGS['t1b'][k] == L.T1[k] for k in same))
        self.assertEqual((L.DATA['t1b'], L.KINDS['t1b']), (L.DATA['t1'], L.KINDS['t1']))


if __name__ == '__main__':
    unittest.main()
