"""Keep official-proxy eligibility separate from prior protected contacts."""
from pathlib import Path
from protocol import BRANCH_BASELINE, CHECKPOINTS, REPEATS, INTERVENTION_CANDIDATES, STATES, VALIDATION_REPEATS


def merge_replays(records):
    """Repeated audits of one trajectory are evidence, not extra trials."""
    merged = {}
    for record in records:
        key = (str(Path(record['source_root']).resolve()), record['run_id'])
        facts = {k: v for k, v in record.items() if k not in ['slurm_job', 'replay_jobs']}
        facts['source_root'] = key[0]
        if key in merged:
            previous = {k: v for k, v in merged[key].items() if k not in ['slurm_job', 'replay_jobs']}
            if facts != previous:
                raise RuntimeError('Contact audits disagree: ' + record['run_id'])
            merged[key]['replay_jobs'] = sorted(set(merged[key]['replay_jobs'] + [record['slurm_job']]))
        else:
            merged[key] = dict(facts, slurm_job=record['slurm_job'], replay_jobs=[record['slurm_job']])
    return list(merged.values())


def prefix_eligibility(rows, contacts, target, states, repeats=range(REPEATS)):
    target = str(Path(target).resolve())
    audits = {c['run_id']: c for c in contacts if str(Path(c['source_root']).resolve()) == target}
    bases = {(r['state'], r['repeat']): r for r in rows
             if r['state'] in states and r['condition'] == BRANCH_BASELINE and r['branch_step'] is None}
    expected = {(s, r) for s in states for r in repeats}
    if set(bases) != expected:
        raise RuntimeError('Corrected baseline coverage missing for contact gate')
    first_contacts = {}
    for key, row in bases.items():
        if row['run_id'] not in audits:
            raise RuntimeError('Corrected baseline contact audit missing: ' + row['run_id'])
        item = audits[row['run_id']]
        actions = item['robot_contact_actions'] + item['target_contact_actions']
        first_contacts[key] = min(actions) if actions else None
    result = []
    for state in states:
        for step in CHECKPOINTS:
            accepted = sorted({r['repeat'] for r in rows if r['state'] == state and r['branch_step'] == step})
            touched = [r for r in accepted if first_contacts[(state, r)] is not None
                       and first_contacts[(state, r)] <= step]
            result.append({'state': state, 'checkpoint': step,
                           'official_proxy_eligible_repeats': accepted,
                           'prior_protected_contact_repeats': touched,
                           'no_prior_protected_contact_repeats': [r for r in accepted if r not in touched],
                           'first_protected_contact_by_repeat': {str(r): first_contacts[(state, r)] for r in repeats},
                           'scope': 'contacts during executed actions after official initialization and 20 settling actions; no rows discarded'})
    return result


def continuation_contact_counts(rows, contacts, target, eligibility):
    """Unaudited proxy successes remain unknown under the contact criterion."""
    target = str(Path(target).resolve())
    audits = {c['run_id']: c for c in contacts if str(Path(c['source_root']).resolve()) == target}
    result = []
    for item in eligibility:
        state, step = item['state'], item['checkpoint']
        admitted = set(item['no_prior_protected_contact_repeats'])
        for condition in ['reference'] + list(INTERVENTION_CANDIDATES):
            selected = [r for r in rows if r['state'] == state and r['branch_step'] == step
                        and r['extra_budget'] == step and r['condition'] == condition]
            if not selected:
                continue
            valid = [r for r in selected if r['repeat'] in admitted]
            successes = [r for r in valid if r['safe_success']]
            audited = [r for r in successes if r['run_id'] in audits]
            without_contact = [r for r in audited
                               if audits[r['run_id']]['completed_without_robot_target_protected_contact']]
            result.append({'state': state, 'checkpoint': step, 'condition': condition,
                           'suffix_actions': 300, 'official_proxy_n': len(selected),
                           'official_proxy_safe_complete': sum(r['safe_success'] for r in selected),
                           'no_prior_contact_n': len(valid),
                           'no_prior_contact_proxy_safe_complete': len(successes),
                           'audited_proxy_successes': len(audited),
                           'completed_without_protected_contact': len(without_contact),
                           'unknown_contact_successes': len(successes) - len(audited),
                           'scope': 'conditional on physically untouched prefixes; unknown is not failure'})
    return result


def initial_reference_contacts(rows, contacts, target, states=None, repeats=range(VALIDATION_REPEATS)):
    """Require a physical replay for each fresh validation, rather than extrapolating."""
    states = [s['id'] for s in STATES] if states is None else states
    target = str(Path(target).resolve())
    audits = {c['run_id']: c for c in contacts if str(Path(c['source_root']).resolve()) == target}
    selected = [r for r in rows if r['state'] in states and r['condition'] == 'reference'
                and r['branch_step'] is None and r['validation']]
    expected = {(state, repeat) for state in states for repeat in repeats}
    if len(selected) != len(expected) or {(r['state'], r['repeat']) for r in selected} != expected:
        raise RuntimeError('Fresh reference validation coverage incomplete')
    for row in selected:
        if row['run_id'] not in audits:
            raise RuntimeError('Fresh reference contact replay missing: ' + row['run_id'])
        contact = audits[row['run_id']]
        if (contact['source_success'], contact['source_safe_success']) != (row['success'], row['safe_success']):
            raise RuntimeError('Reference contact replay outcome disagrees')
    result = []
    for state in states:
        group = [r for r in selected if r['state'] == state]
        result.append({'state': state, 'n': len(group),
                       'task_complete': sum(r['success'] for r in group),
                       'official_proxy_safe_complete': sum(r['safe_success'] for r in group),
                       'both_safe_complete': sum(r['safe_success'] and
                            audits[r['run_id']]['completed_without_robot_target_protected_contact'] for r in group),
                       'scope': 'each fresh execution replayed; fixed initial state and deterministic reference, not probability certification'})
    return result
