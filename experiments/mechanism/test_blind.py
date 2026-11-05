"""Tests for blind.py (P4 blind explanation test).

  /projects/p33100/siosio/envs/openpi/bin/python -m unittest test_blind
"""
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blind as B  # noqa: E402
import run_neurons as RN  # noqa: E402


def fake_root(tmp):
    """Two neurons of pair 1->8; neuron 0 is high exactly when the instruction mentions 'plate', neuron 1 is noise."""
    root = Path(tmp)/'open'; root.mkdir()
    states = []
    for init in list(range(0, 2)) + list(range(10, 14)):
        for phase in (0, .5):
            states.append(dict(id='s%d_%s' % (init, phase), pair='1->8', init=init, phase=phase, prompt_a='put the bowl on the stove',
                               prompt_b='put the bowl on the plate'))
    (root/'STATES.json').write_text(json.dumps(dict(states=states)))
    RN.write(root/'PLAN.json', dict(state_ids=[s['id'] for s in states], states_sha256=RN.sha(root/'STATES.json'), shards=1))
    width = 100
    RN.write(root/'NEURONS.json', dict(width=width, union=[3 * width + 7, 5 * width + 1],
                                       pairs={'1->8': dict(top=[[3, 7, .5], [5, 1, .1]] + [[6, n, 0.] for n in range(20)])}))
    rng = np.random.default_rng(0); profiles = {}
    for s in states:
        prof = np.zeros((14, 2, 3), np.float32)
        for j, (name, text) in enumerate(RN.prompts(s['init'])):
            prof[j, 0, 0] = (2. if 'plate' in text else 0.) + .01 * rng.standard_normal()
            prof[j, 1, 0] = rng.standard_normal()
        RN.save(root/'states'/s['id'], dict(id=s['id']), dict(PROFILE=prof))
    for flat in (3 * width + 7, 5 * width + 1):
        profiles[str(flat)] = dict(mean={'1->8 f=0': {'goal8': 2., 'goal1': 0.}})
    RN.write(root/'PROFILES_DISCOVERY.json', profiles)
    return root


class Blind(unittest.TestCase):
    def test_wilson_and_reading(self):
        self.assertAlmostEqual(B.wilson_lower(20, 24), .64147, places=4)   # by hand: (0.91336 - 0.16922) / 1.16006
        self.assertEqual(B.wilson_lower(0, 0), 0.)
        per = {k: dict(explained=e) for k, e in zip('abcde', (1, 1, 1, 0, 0))}
        self.assertEqual(B.reading(per)[0], 'interpretable')
        self.assertEqual(B.reading({k: dict(explained=e) for k, e in zip('abcde', (1, 0, 0, 0, 0))})[0], 'partly_interpretable')
        self.assertEqual(B.reading({k: dict(explained=0) for k in 'abcde'})[0], 'opaque')

    def test_baseline_prefers_words_of_b(self):
        rng = random.Random(0)
        a = dict(prompt='put the bowl on the plate', prompt_b='put the bowl on the plate')
        b = dict(prompt='turn on the stove', prompt_b='put the bowl on the plate')
        self.assertEqual(B.baseline_pick(a, b, rng), 1); self.assertEqual(B.baseline_pick(b, a, rng), 2)

    def test_questions_and_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fake_root(tmp)
            expl = Path(tmp)/'EXPLANATIONS.json'
            RN.write(expl, {'307': 'High when the instruction mentions the plate.', '501': 'No pattern.'})
            n = B.questions(root, expl, Path(tmp)/'blind')
            q = RN.read(Path(tmp)/'blind'/'QUESTIONS.json')
            self.assertEqual(n, len(q['questions'])); self.assertEqual(set(q['explanations']), {'N01', 'N02'})
            self.assertNotIn('value', json.dumps(q)); self.assertTrue(all('s0_' not in x['item1'] for x in q['questions']))
            # answers from the plate rule: right on N01 (and its trivial baseline is also right: the margin decides)
            ans = {x['id']: (1 if 'plate' in x['item1'].split('instruction')[1] else 2) for x in q['questions']}
            truth = RN.read(Path(tmp)/'blind'/'TRUTH.json')['truth']
            RN.write(Path(tmp)/'ANSWERS.json', ans)
            out = B.score(Path(tmp)/'blind', Path(tmp)/'ANSWERS.json')
            self.assertEqual(out['per_neuron']['N01']['accuracy'], 1.)
            self.assertLess(out['per_neuron']['N02']['accuracy'], .9)
            self.assertEqual(sum(truth[k] == ans[k] for k in truth if k.startswith('N01')), out['per_neuron']['N01']['correct'])
            RN.write(Path(tmp)/'BAD.json', dict(list(ans.items())[:-1]))
            with self.assertRaises(RuntimeError):
                B.score(Path(tmp)/'blind', Path(tmp)/'BAD.json')
            with self.assertRaises(RuntimeError):
                RN.write(Path(tmp)/'E2.json', {'307': 'x'}); B.questions(root, Path(tmp)/'E2.json', Path(tmp)/'blind2')


if __name__ == '__main__':
    unittest.main()
