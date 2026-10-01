"""Evidence gates and paired, count-only diagnostic comparisons (no probabilities)."""
import hashlib
import json
from collections import defaultdict
from protocol import STATES, CONDITIONS, CANDIDATES, REPEATS, VALIDATION_REPEATS, seed_for, BRANCH_CONDITIONS, BRANCH_BASELINE, INTERVENTION_CANDIDATES

INITIAL_COMMON_FIELDS = [
    'sim_state_sha256', 'qpos_sha256', 'qvel_sha256', 'ctrl_sha256',
    'warmstart_sha256', 'applied_force_sha256', 'external_force_sha256',
    'mocap_pos_sha256', 'mocap_quat_sha256', 'act_sha256',
    'marker_position', 'marker_quaternion', 'controller', 'action_queue',
    'policy_rng', 'python_rng', 'numpy_rng', 'observation',
]

def execution_digest(payload, include_aegis=False):
    fields = INITIAL_COMMON_FIELDS + (['aegis'] if include_aegis else [])
    selected = {key: payload[key] for key in fields}  # Missing evidence fails closed.
    return hashlib.sha256(json.dumps(selected, sort_keys=True, allow_nan=False).encode()).hexdigest()

def validate_rows(rows, require_initial_complete=False):
    by_id = {s['id']: s for s in STATES}
    if len({r['run_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate run identity')
    if len({r['code_commit'] for r in rows}) > 1:
        raise ValueError('Mixed scientific source commits')
    if any(r['upstream_commit'] != '2457feed5968ae803926e178c8ce8243b9ecdcf9' for r in rows):
        raise ValueError('Unexpected scientific upstream')
    paired = defaultdict(list)
    initial = {}
    branches = defaultdict(list)
    for row in rows:
        state = by_id[row['state']]
        if row['seed'] != seed_for(state, row['repeat'], row['validation']):
            raise ValueError('Unexpected policy seed')
        if row['safe_success'] != (row['success'] and not row['collided']):
            raise ValueError('Incorrect safe completion label')
        if row['collided'] != (row['max_obstacle_l1_m'] > .001):
            raise ValueError('Incorrect official collision label')
        if row['end_step'] < 0 or row['end_step'] > 300 + row['extra_budget']:
            raise ValueError('Incorrect execution budget')
        ack = row['rng_reset']
        if ack['seed'] != row['seed'] or ack['run_id'] != row['run_id'] or ack['requests'] != 0:
            raise ValueError('RNG reset stream mismatch')
        paired[(row['state'], row['seed'])].append(row)
        if row['branch_step'] is None and row['condition'] in CONDITIONS:
            key = (row['state'], row['repeat'], row['condition'])
            if key in initial:
                raise ValueError('Duplicate initial condition')
            initial[key] = row
        if row['branch_step'] is not None:
            if row.get('prefix_condition') != BRANCH_BASELINE:
                raise ValueError('Continuation uses uncontrolled perception prefix')
            if row['condition'] not in BRANCH_CONDITIONS:
                raise ValueError('Unexpected continuation candidate')
            if row['end_step'] < row['branch_step']:
                raise ValueError('Continuation ends before its claimed checkpoint')
            if not row.get('prefix_verified') or not row.get('checkpoint_fingerprint'):
                raise ValueError('Unverified continuation prefix')
            branches[(row['state'], row['repeat'], row['branch_step'])].append(row)
    for group in paired.values():
        for key in ['initial_policy_input_sha256', 'first_chunk_sha256', 'initial_execution_fingerprint']:
            values = [r.get(key) for r in group]
            if any(not value for value in values) or len(set(values)) != 1:
                raise ValueError('Incomplete or mismatched initial execution: ' + key)
    for group in branches.values():
        if len({r['checkpoint_fingerprint'] for r in group}) != 1:
            raise ValueError('Candidates/budgets use different physical prefixes')
        keys = [(r['extra_budget'], r['condition']) for r in group]
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicate continuation candidate')
    if require_initial_complete:
        expected = {(s['id'], repeat, c) for s in STATES for repeat in range(REPEATS) for c in CONDITIONS}
        if set(initial) != expected:
            raise ValueError('Initial perception diagnostic is incomplete')
        for state in STATES:
            validation = [r for r in rows if r['state'] == state['id'] and r['condition'] == 'reference'
                          and r['branch_step'] is None and r['validation']]
            if {r['repeat'] for r in validation} != set(range(VALIDATION_REPEATS)) or len(validation) != VALIDATION_REPEATS:
                raise ValueError('Fresh reference validation is incomplete')
            screening = [r for r in rows if r['state'] == state['id'] and r['condition'] == 'reference'
                         and r['branch_step'] is None and not r['validation']]
            if sorted(r['variant'] for r in screening) != ['center', 'rim', 'side']:
                raise ValueError('Prespecified reference screening is incomplete')
    return {'paired_groups': len(paired), 'verified_initial_runs': len(rows),
            'continuation_prefix_groups': len(branches), 'initial_complete': require_initial_complete}

def paired_effect(rows, state_id, before, after):
    groups = {c: {r['repeat']: r for r in rows if r['state'] == state_id
                  and r['branch_step'] is None and r['condition'] == c} for c in [before, after]}
    repeats = sorted(set(groups[before]) & set(groups[after]))
    wins = losses = 0
    for repeat in repeats:
        a, b = groups[before][repeat], groups[after][repeat]
        if a['seed'] != b['seed']:
            raise ValueError('Unpaired perception contrast')
        wins += not a['safe_success'] and b['safe_success']
        losses += a['safe_success'] and not b['safe_success']
    return {'before': before, 'after': after, 'paired_n': len(repeats),
            'safe_completion_gains': wins, 'safe_completion_losses': losses, 'net_count': wins - losses}

def continuation_comparisons(rows, state_id):
    """Match surviving prefixes/seed cohorts; missing continuations are never failures."""
    groups = defaultdict(dict)
    for r in rows:
        if r['state'] != state_id or r['branch_step'] is None:
            continue
        step = r['branch_step']
        key = (step, r['extra_budget'], r['repeat'])
        groups[key][r['condition']] = r
    complete = {key: value for key, value in groups.items()
                if set(value) == set(BRANCH_CONDITIONS)}
    records = []
    for step in sorted({key[0] for key in complete}):
        if step == 0:
            continue
        # Equal 300-action suffix at early and late checkpoints, using the same repeats.
        repeats = sorted({r for s, b, r in complete if s == 0 and b == 0}
                         & {r for s, b, r in complete if s == step and b == step})
        early = late = gains = losses = 0
        for repeat in repeats:
            a, b = complete[(0, 0, repeat)], complete[(step, step, repeat)]
            x = any(a[c]['safe_success'] for c in INTERVENTION_CANDIDATES)
            y = any(b[c]['safe_success'] for c in INTERVENTION_CANDIDATES)
            early += x; late += y; gains += not x and y; losses += x and not y
        remaining = sorted({r for s, b, r in complete if s == step and b == 0}
                           & {r for s, b, r in complete if s == step and b == step})
        budget_gains = budget_losses = 0
        for repeat in remaining:
            a, b = complete[(step, 0, repeat)], complete[(step, step, repeat)]
            x = any(a[c]['safe_success'] for c in INTERVENTION_CANDIDATES)
            y = any(b[c]['safe_success'] for c in INTERVENTION_CANDIDATES)
            budget_gains += not x and y; budget_losses += x and not y
        reference_early=sum(complete[(0,0,r)]['reference']['safe_success'] for r in repeats)
        reference_late=sum(complete[(step,step,r)]['reference']['safe_success'] for r in repeats)
        records.append({'reference_early_witness':reference_early,'reference_late_witness':reference_late,'checkpoint': step, 'same_surviving_prefix_repeats': repeats,
                        'equal_suffix_actions': 300, 'early_any_candidate_witness': early,
                        'late_any_candidate_witness': late, 'opportunity_losses': losses,
                        'opportunity_gains': gains, 'budget_matched_repeats': remaining,
                        'extra_budget_gains': budget_gains, 'extra_budget_losses': budget_losses,
                        'scope': 'ordinary intervention candidates exclude privileged reference; conditional on noncollided common prefixes, not true infeasibility'})
    return records
