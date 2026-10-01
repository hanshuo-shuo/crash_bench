import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import INITIAL_COMMON_FIELDS, execution_digest, validate_rows, paired_effect, continuation_comparisons
from protocol import STATES, CANDIDATES, seed_for, BRANCH_CONDITIONS, BRANCH_BASELINE

def row(condition='raw', repeat=0, success=False, collided=False, step=None, extra=0):
    s = STATES[0]
    name = '%s_%s_%d_%s_%d' % (s['id'], condition, repeat, step, extra)
    seed = seed_for(s, repeat)
    return {'run_id': name, 'state': s['id'], 'role': s['role'], 'repeat': repeat, 'seed': seed,
            'condition': condition, 'validation': False, 'success': success, 'collided': collided,
            'safe_success': success and not collided, 'max_obstacle_l1_m': .01 if collided else 0,
            'end_step': max(200, step or 0), 'extra_budget': extra, 'branch_step': step,
            'rng_reset': {'seed': seed, 'run_id': name, 'requests': 0},
            'initial_policy_input_sha256': 'input', 'first_chunk_sha256': 'chunk',
            'initial_execution_fingerprint': 'physics', 'prefix_verified': True,
            'checkpoint_fingerprint': 'prefix%d' % (step or 0), 'prefix_condition': BRANCH_BASELINE if step is not None else None,
            'code_commit': 'fixed', 'upstream_commit': '2457feed5968ae803926e178c8ce8243b9ecdcf9'}

class AnalysisTests(unittest.TestCase):
    def test_physics_digest_not_just_qpos_images_and_aegis_factor_separate(self):
        payload = {k: 'fixed' for k in INITIAL_COMMON_FIELDS}
        payload['aegis'] = {'geometry': 'raw'}
        baseline = execution_digest(payload)
        for key in ['qvel_sha256', 'ctrl_sha256', 'warmstart_sha256', 'action_queue', 'policy_rng']:
            other = copy.deepcopy(payload); other[key] = 'changed'
            self.assertNotEqual(baseline, execution_digest(other))
        other = copy.deepcopy(payload); other['aegis'] = {'geometry': 'corrected'}
        self.assertEqual(baseline, execution_digest(other))
        self.assertNotEqual(execution_digest(payload, True), execution_digest(other, True))
        del other['ctrl_sha256']
        with self.assertRaises(KeyError): execution_digest(other)

    def test_missing_physical_evidence_or_false_reset_and_unverified_prefix_stop(self):
        a, b = row(), row('geometry')
        validate_rows([a, b])
        for field, value in [('initial_execution_fingerprint', None), ('first_chunk_sha256', 'other'), ('seed', 0)]:
            changed = copy.deepcopy(b); changed[field] = value
            with self.assertRaises(ValueError): validate_rows([a, changed])
        changed = row('aegis',step=50); changed['prefix_verified'] = False
        with self.assertRaises(ValueError): validate_rows([changed])
        changed = copy.deepcopy(b); changed['rng_reset']['run_id'] = 'other'
        with self.assertRaises(ValueError): validate_rows([changed])
        with self.assertRaises(ValueError): validate_rows([a, b], require_initial_complete=True)

    def test_identity_and_geometry_effects_are_separate_paired_counts(self):
        data = [row('raw'), row('identity'), row('geometry', success=True)]
        self.assertEqual(paired_effect(data, STATES[0]['id'], 'raw', 'identity')['net_count'], 0)
        self.assertEqual(paired_effect(data, STATES[0]['id'], 'raw', 'geometry')['net_count'], 1)

    def test_equal_budget_late_drop_uses_same_survivor_seeds_and_missing_not_failure(self):
        data = []
        for repeat in [0, 1]:
            for c in BRANCH_CONDITIONS:
                data.append(row(c, repeat, c in ['reference','nominal'], step=0))
        # Only repeat0 has a late safe checkpoint; repeat1 must not enlarge early denominator.
        for extra in [0, 250]:
            for c in BRANCH_CONDITIONS:
                data.append(row(c, 0, False, step=250, extra=extra))
        result = continuation_comparisons(data, STATES[0]['id'])[0]
        self.assertEqual(result['same_surviving_prefix_repeats'], [0])
        self.assertEqual(result['early_any_candidate_witness'], 1)
        self.assertEqual(result['late_any_candidate_witness'], 0)
        self.assertEqual(result['opportunity_losses'], 1)
        self.assertEqual(result['extra_budget_gains'], 0)
        privileged_only=[dict(r,safe_success=(r['condition']=='reference'),success=(r['condition']=='reference')) for r in data]
        only=continuation_comparisons(privileged_only,STATES[0]['id'])[0]
        self.assertEqual(only['early_any_candidate_witness'],0)
        self.assertEqual(only['late_any_candidate_witness'],0)
        self.assertEqual(only['reference_late_witness'],1)
        incomplete = [r for r in data if not (r['branch_step'] == 250 and r['extra_budget'] == 250 and r['condition'] == 'reference')]
        result = continuation_comparisons(incomplete, STATES[0]['id'])[0]
        self.assertEqual(result['same_surviving_prefix_repeats'], [])
        self.assertEqual(result['opportunity_losses'], 0)

if __name__ == '__main__': unittest.main()
