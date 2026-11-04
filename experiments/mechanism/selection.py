"""P0 of the mechanism pilot: the switch states, read from the frozen HIL suite stage 1 run (no new execution).

A state is one (pair, init, phase) switch point of LIBERO-Goal group 1a. The SW branch (prompt B) and its A twin start from
the same replayed prefix with the same seed, so both directories hold the same policy input (first_policy_input.npz, the
prompt excepted) and the native first chunk under their own prompt (POLICY.jsonl, request 1). The A twin is the C0 branch
at f = .25/.5/.75 and the prefix run itself at f = 0 (the switch at checkpoint WAIT = the prefix's first request).
Every unit of the chosen pairs is kept; nothing is selected on outcome.

Usage: python selection.py BRANCH_ROOT OUT_JSON
"""
import collections
import gzip
import json
import sys
from pathlib import Path

# LIBERO-Goal task ids (A, B), chosen from the stage 1 table and the stage 5 pair-level re-read (HIL_MECHANISM_IDEA.md 3):
# 1->8 the positive control (bowl, destination only, B 1.00/1.00/1.00/0.95, A-B gap largest while carrying),
# 7->8 cross-object with a decaying A-B gap (B 1.00/0.75/0.15/0.40), 2->9 same object with a weak gap (0.90/0.20/0.30/0.00).
PAIRS = ((1, 8), (7, 8), (2, 9))
PHASES = (0, .25, .5, .75)
SMOKE_INITS = (0, 1)


def read(path):
    with open(path) as stream:
        return json.load(stream)


def first_request(folder):
    """The native first chunk of a run: request 1 of POLICY.jsonl, issued right after the RNG reset."""
    with open(Path(folder)/'POLICY.jsonl') as stream:
        line = json.loads(stream.readline())
    if line['request_index'] != 1:
        raise ValueError('First logged request is not request 1: %s' % folder)
    return {k: line[k] for k in ('step', 'input_sha256', 'rng_before', 'native_chunk_sha256', 'actions')}


def select(plan, runs, pairs=PAIRS, inits=None):
    """State rows for every 1a SW unit of `pairs` (optionally only `inits`), each with its A twin; raises on any mismatch."""
    rows = {r['id']: r for r in runs}
    twins = {(u['prefix'], u['checkpoint']): u for u in plan['branches'] if u['group'] == '1a' and u['condition'] == 'C0'}
    states = []
    for u in plan['branches']:
        if u['group'] != '1a' or u['condition'] != 'SW' or (u['task'], u['other']) not in pairs:
            continue
        if inits is not None and u['init'] not in inits:
            continue
        if u['phase'] not in PHASES:
            raise ValueError('Unexpected phase %r in %s' % (u['phase'], u['id']))
        if u['phase'] == 0:
            twin = None
        else:
            twin = twins[(u['prefix'], u['checkpoint'])]
            if twin['seed'] != u['seed'] or twin['task'] != u['task']:
                raise ValueError('Twin mismatch for %s' % u['id'])
        run = rows[u['id']]
        states.append(dict(
            id='%s_f%03d' % (u['id'], round(100 * u['phase'])), pair='%d->%d' % (u['task'], u['other']),
            task_a=u['task'], task_b=u['other'], init=u['init'], phase=u['phase'], checkpoint=u['checkpoint'], seed=u['seed'],
            prefix=u['prefix'], sw=u['id'], twin=None if twin is None else twin['id'],
            prompt_b=u['prompt'], tokens_b=plan['prompt_tokens'][u['prompt']],
            b_done=bool(run['b_done']), twin_a_done=None if twin is None else bool(rows[twin['id']]['a_done'])))
    return states


def attach_files(states, branch_root, prefix_root):
    """Adds each state's A prompt, input and native chunk records (A twin and SW), checking the shared policy input."""
    for s in states:
        sw = Path(branch_root)/'branch'/s['sw']
        a_dir = Path(prefix_root)/'prefix'/s['prefix'] if s['twin'] is None else Path(branch_root)/'branch'/s['twin']
        a_row = read(a_dir/'ROW.json'); sw_row = read(sw/'ROW.json')
        if a_row['seed'] != s['seed'] or sw_row['seed'] != s['seed']:
            raise ValueError('Seed mismatch for %s' % s['id'])
        a_first, b_first = first_request(a_dir), first_request(sw)
        if a_first['step'] != s['checkpoint'] or b_first['step'] != s['checkpoint']:
            raise ValueError('First request is not at the switch step for %s' % s['id'])
        if a_first['rng_before'] != b_first['rng_before']:
            raise ValueError('A twin and SW start from different RNG states for %s' % s['id'])
        # The twins share the physics hash at the checkpoint, but the rendered agent-view image can differ by one intensity
        # level in a few pixels (EGL; 62 of the first 120 1a twin pairs). Both prompts are therefore fed the SW input; the
        # runner compares the A twin's stored input array and reports the difference with the A fidelity check.
        same = a_row.get('policy_input_state_sha256')
        s.update(prompt_a=a_row['prompt'], a_dir=str(a_dir), b_dir=str(sw), a_first=a_first, b_first=b_first,
                 input_npz=str(sw/'first_policy_input.npz'), a_input_npz=str(a_dir/'first_policy_input.npz'),
                 twin_state_sha_equal=None if same is None else same == sw_row['policy_input_state_sha256'])
    return states


def summary(states):
    """Counts per (pair, phase): states and switch successes, for the plan's table."""
    cells = collections.defaultdict(lambda: [0, 0])
    for s in states:
        cell = cells['%s f=%s' % (s['pair'], s['phase'])]; cell[0] += 1; cell[1] += s['b_done']
    return {k: dict(states=v[0], b_done=v[1]) for k, v in sorted(cells.items())}


def main(branch_root, out_json, smoke=False):
    branch_root = Path(branch_root); plan = read(branch_root/'PLAN.json')
    with gzip.open(branch_root/'RUNS.jsonl.gz', 'rt') as stream:
        runs = [json.loads(x) for x in stream]
    states = attach_files(select(plan, runs, inits=SMOKE_INITS if smoke else None), branch_root, plan['prefix_root'])
    tokens = {s['prompt_a'] for s in states}
    for s in states:
        s['tokens_a'] = plan['prompt_tokens'].get(s['prompt_a'])
    out = dict(branch_root=str(branch_root), prefix_root=plan['prefix_root'], pairs=[list(p) for p in PAIRS],
               phases=list(PHASES), smoke=smoke, states=states, summary=summary(states), prompts_a=sorted(tokens))
    Path(out_json).write_text(json.dumps(out, indent=1))
    return out


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a != '--smoke']
    result = main(args[0], args[1], smoke='--smoke' in sys.argv)
    print(json.dumps(result['summary'], indent=1))
