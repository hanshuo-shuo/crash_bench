"""P4 blind explanation test (docs/generative_verifier/HIL_MECHANISM_NEURONS.md 4.3 and N6), CPU only.

  blind.py questions OPEN_ROOT EXPLANATIONS OUT_DIR    QUESTIONS.json (what the blind answerer sees), TRUTH.json, BASELINE.json
  blind.py score OUT_DIR ANSWERS                       SCORE.json with the per-neuron results and the N6 reading

The questions come from the reading inits (10-19) of the formal open-loop root; the explanations were written from the
discovery inits only (PROFILES_DISCOVERY.json). An item is "the robot was doing task A, at progress f, and now receives
instruction X"; a question pairs two items of one neuron and asks which one gives the higher mean activation on the image rows.
"""
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import neuron_readings as NR  # noqa: E402
import run_neurons as RN  # noqa: E402

QUESTIONS = 24          # per neuron
TIE_FRACTION = .1       # pairs closer than this fraction of the neuron's discovery range are not asked
ACCURACY, MARGIN = .75, .10
SHARE_ALL, SHARE_PART = .5, .2


def words(text):
    return set(text.lower().split())


def wilson_lower(k, n, z=1.959963984540054):
    if n == 0:
        return 0.
    p = k / n; d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def describe(state, prompt):
    progress = 'at the very start, before moving' if state['phase'] == 0 else 'about %d%% of the way through it' % round(100 * state['phase'])
    return 'The robot was doing "%s", %s; it now receives the instruction "%s".' % (state['prompt_a'], progress, prompt)


def items(root, plan, states, neurons, flat):
    """(description, value, state, prompt) for every reading-init state of the neuron's owner pairs and every profile prompt."""
    width = neurons['width']; l, n = divmod(flat, width)
    owners = [p for p, info in neurons['pairs'].items() if [l, n] in [x[:2] for x in info['top'][:NR.TOP_PROFILE]]]
    j = neurons['union'].index(flat); out = []
    for sid in plan['state_ids']:
        s = states[sid]
        if s['pair'] not in owners or s['init'] not in NR.READING_INITS:
            continue
        row = RN.verified(Path(root)/'states'/sid); prof = np.load(Path(root)/'states'/sid/'PROFILE.npy')[:, j, 0]
        for (name, text), value in zip(RN.prompts(s['init']), prof):
            if name == 'random':
                text = '<a random string of the letters A-F and +>'
            out.append(dict(text=describe(s, text), value=float(value), pair=s['pair'], prompt=text, prompt_b=s['prompt_b'], state=sid))
        assert row['id'] == sid
    return owners, out


def discovery_range(profiles, flat):
    values = [v for c in profiles[str(flat)]['mean'].values() for v in c.values()]
    return max(values) - min(values)


def baseline_pick(a, b, rng):
    """Higher shared-word count with the state's B instruction; then the item whose instruction is B; then a coin."""
    sa, sb = len(words(a['prompt']) & words(a['prompt_b'])), len(words(b['prompt']) & words(b['prompt_b']))
    if sa != sb:
        return 1 if sa > sb else 2
    ea, eb = a['prompt'] == a['prompt_b'], b['prompt'] == b['prompt_b']
    if ea != eb:
        return 1 if ea else 2
    return rng.choice((1, 2))


def questions(open_root, explanations_path, out_dir):
    root, plan, states = RN.load(open_root)
    neurons = RN.read(root/'NEURONS.json'); profiles = RN.read(root/'PROFILES_DISCOVERY.json')
    explanations = RN.read(explanations_path)
    if sorted(explanations) != sorted(str(f) for f in neurons['union']):
        raise RuntimeError('One explanation per neuron of the union is required')
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=False)
    asked, truth, base = [], {}, {}
    for k, flat in enumerate(neurons['union']):
        label = 'N%02d' % (k + 1)
        owners, pool = items(root, plan, states, neurons, flat)
        rng = random.Random(NR.seed_of('blind:%d' % flat)); spread = discovery_range(profiles, flat); got = 0; tries = 0
        while got < QUESTIONS and tries < 20000:
            tries += 1
            a, b = rng.sample(pool, 2)
            if abs(a['value'] - b['value']) < TIE_FRACTION * spread or a['text'] == b['text']:
                continue
            got += 1; qid = '%s-%02d' % (label, got)
            asked.append(dict(id=qid, neuron=label, item1=a['text'], item2=b['text']))
            truth[qid] = 1 if a['value'] > b['value'] else 2
            base[qid] = baseline_pick(a, b, rng)
    labels = {('N%02d' % (k + 1)): str(f) for k, f in enumerate(neurons['union'])}
    RN.write(out_dir/'QUESTIONS.json', dict(explanations={lab: explanations[f] for lab, f in labels.items()}, questions=asked))
    RN.write(out_dir/'TRUTH.json', dict(truth=truth, labels=labels, open_root=str(root), explanations_sha256=RN.sha(explanations_path)))
    RN.write(out_dir/'BASELINE.json', base)
    return len(asked)


def reading(per):
    explained = [k for k, v in per.items() if v['explained']]
    share = len(explained) / len(per) if per else 0.
    return ('interpretable' if share >= SHARE_ALL else 'partly_interpretable' if share >= SHARE_PART else 'opaque'), explained


def score(out_dir, answers_path):
    out_dir = Path(out_dir); truth = RN.read(out_dir/'TRUTH.json'); base = RN.read(out_dir/'BASELINE.json'); answers = RN.read(answers_path)
    if set(answers) != set(truth['truth']) or any(v not in (1, 2) for v in answers.values()):
        raise RuntimeError('Answers must give 1 or 2 for every question')
    per = {}
    for label in truth['labels']:
        ids = [q for q in truth['truth'] if q.startswith(label + '-')]
        right = sum(answers[q] == truth['truth'][q] for q in ids); trivial = sum(base[q] == truth['truth'][q] for q in ids)
        acc = right / len(ids) if ids else 0.; bacc = trivial / len(ids) if ids else 0.
        per[label] = dict(flat=truth['labels'][label], questions=len(ids), correct=right, accuracy=acc, wilson_lower=wilson_lower(right, len(ids)),
                          baseline_accuracy=bacc, explained=bool(ids) and acc >= ACCURACY and wilson_lower(right, len(ids)) > .5 and acc - bacc >= MARGIN)
    result, explained = reading(per)
    out = dict(reading=result, explained=explained, per_neuron=per, answers_sha256=RN.sha(answers_path))
    RN.write(out_dir/'SCORE.json', out)
    return out


if __name__ == '__main__':
    if sys.argv[1] == 'questions':
        print(questions(*sys.argv[2:5]), 'questions')
    else:
        print(json.dumps({k: v for k, v in score(*sys.argv[2:4]).items() if k != 'per_neuron'}, indent=1))
