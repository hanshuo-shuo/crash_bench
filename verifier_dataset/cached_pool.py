"""A bounded, source-diverse continuation of the successful offline smoke."""
from .inventory import identity


GROUPS = [('safelibero_goal', t) for t in range(4)] + [
    ('safelibero_spatial', t) for t in (0, 2, 3)] + [
    ('safelibero_long', t) for t in (2, 3)]


def select_pool(outcomes):
    """Freeze five historical proxy successes per task, excluding the Goal smoke.

    Success is used only to route collection. No row gains a feasibility label.
    Interleave groups so each of two shards covers several task compositions.
    """
    groups = []
    for suite, task in GROUPS:
        eligible = sorted([r for r in outcomes if r['scene']['suite'] == suite
                           and r['scene']['level'] == 'I' and r['scene']['task'] == task
                           and r['method'] == 'aegis' and r['proxy_safe_success']],
                          key=lambda r: r['scene']['episode'])[:5]
        if len(eligible) != 5:
            raise RuntimeError('Insufficient frozen source coverage')
        groups.append(eligible[1:] if suite == 'safelibero_goal' else eligible)
    selected = [group[i] for i in range(5) for group in groups if i < len(group)]
    ids = [identity(r['scene']) for r in selected]
    if len(selected) != 41 or len(set(ids)) != 41:
        raise RuntimeError('Expected 41 distinct additional source identities')
    if any(r['feasibility_label'] != 'unknown' for r in selected):
        raise RuntimeError('A proxy outcome must remain unlabeled')
    return selected


def run_spec(stage, index, entry):
    prefix = 'fd_pool' if stage == 'collect-cached-batch' else 'fd_goal'
    return dict(name=prefix + '%02d_aegis' % index, scene=entry['scene'],
                repeat=0, method='aegis', diagnostics=False)
