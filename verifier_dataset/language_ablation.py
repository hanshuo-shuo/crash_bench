"""From-start language ablation on unchanged standard LIBERO tasks; pure plan and statistics."""
from collections import defaultdict
import hashlib
import random
from . import hil_suite as S

CONDITIONS = ('C0', 'EMPTY', 'RANDOM')
ALPHABET = tuple('ABCDEF+')
SYMBOLS = 12
RANDOM_SALT = 'language-ablation-20261103'
INITS = tuple(range(10))
SMOKE_INIT = 40
BOOTSTRAP = 10000
BOOTSTRAP_SEED = 1103


def random_prompt(init):
    """The same task-independent prompt bank for every task, with a separate seed from the policy."""
    seed = int(hashlib.sha256(('%s|%d' % (RANDOM_SALT, init)).encode()).hexdigest(), 16)
    rng = random.Random(seed)
    return ' '.join(rng.choice(ALPHABET) for _ in range(SYMBOLS))


def plan(names, mode):
    if mode not in ('smoke', 'formal') or any(len(names[s]) != 10 for s in S.SUITES):
        raise ValueError('Expected smoke/formal and ten tasks per suite')
    units = []
    for suite in S.SUITES:
        for task in (range(1) if mode == 'smoke' else range(10)):
            for init in ((SMOKE_INIT,) if mode == 'smoke' else INITS):
                original = S.language(names[suite][task]); group = S.prefix_id(suite, task, init)
                prompts = dict(C0=original, C0R=original, EMPTY='', RANDOM=random_prompt(init))
                for condition in CONDITIONS + (('C0R',) if mode == 'smoke' or init == INITS[0] else ()):
                    units.append(dict(id=group + '_' + condition, group=group, suite=suite, task=task, init=init,
                                      condition=condition, prompt=prompts[condition], original_prompt=original,
                                      name=names[suite][task], seed=S.policy_seed(suite, task, init), max_steps=S.SUITES[suite]))
    return units


def shard_indices(units, shard, shards):
    """All conditions and the repeat for a state stay on one GPU."""
    groups = {g: i for i, g in enumerate(dict.fromkeys(u['group'] for u in units))}
    return [i for i, u in enumerate(units) if groups[u['group']] % shards == shard]


def paired_rows(units, rows):
    if set(rows) != {u['id'] for u in units} or len(rows) != len(units):
        raise RuntimeError('Missing, extra or duplicate runs')
    grouped = defaultdict(dict)
    for u in units:
        row = rows[u['id']]
        if any(row[k] != v for k, v in u.items()) or not isinstance(row['success'], bool):
            raise RuntimeError('Recorded unit differs from the plan: ' + u['id'])
        if u['condition'] in grouped[u['group']]:
            raise RuntimeError('Duplicate condition: ' + u['group'])
        grouped[u['group']][u['condition']] = row['success']
    out = []
    for group, conditions in grouped.items():
        if not set(CONDITIONS) <= set(conditions):
            raise RuntimeError('Incomplete paired state: ' + group)
        u = next(u for u in units if u['group'] == group)
        out.append(dict(group=group, suite=u['suite'], task=u['task'], init=u['init'], **conditions))
    return out


def statistics(pairs, replicates=BOOTSTRAP):
    """Paired bootstrap, resampling initial states within each fixed task; equal task weights."""
    import numpy as np
    tasks = defaultdict(list)
    for p in pairs:
        tasks[(p['suite'], p['task'])].append([int(p[c]) for c in CONDITIONS])
    if not tasks:
        raise ValueError('No paired states')
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    point = np.zeros(3); draws = np.zeros((replicates, 3))
    for key in sorted(tasks):
        a = np.asarray(tasks[key], dtype=float); point += a.mean(axis=0) / len(tasks)
        indices = rng.integers(len(a), size=(replicates, len(a)))
        draws += a[indices].mean(axis=1) / len(tasks)

    def interval(value, samples):
        valid = samples[np.isfinite(samples)]
        return dict(value=None if not np.isfinite(value) else float(value),
                    ci=None if not len(valid) else np.percentile(valid, [2.5, 97.5]).tolist(), valid_bootstraps=len(valid))

    result = dict(states=len(pairs), tasks=len(tasks), success={}, contrasts={})
    for k, c in enumerate(CONDITIONS):
        result['success'][c] = dict(interval(point[k], draws[:, k]), count=sum(p[c] for p in pairs), n=len(pairs))
        if k:
            ratio = np.divide(draws[:, k], draws[:, 0], out=np.full(replicates, np.nan), where=draws[:, 0] > 0)
            retained = [p[c] for p in pairs if p['C0']]
            result['contrasts'][c] = dict(difference=interval(point[k] - point[0], draws[:, k] - draws[:, 0]),
                                          ratio=interval(point[k] / point[0] if point[0] else np.nan, ratio),
                                          baseline_only=sum(p['C0'] and not p[c] for p in pairs),
                                          ablation_only=sum(p[c] and not p['C0'] for p in pairs),
                                          retained_baseline_successes=sum(retained), baseline_successes=len(retained))
    return result
