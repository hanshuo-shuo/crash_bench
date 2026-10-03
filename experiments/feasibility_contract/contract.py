"""Pure standard-library evidence rules; no simulator dependency."""
import math

UPSTREAM = '2457feed5968ae803926e178c8ce8243b9ecdcf9'
PREFIX_ROOT = '/projects/p33100/siosio/crashbench_safelibero/feasibility/20261001T215021Z_initial_327205258fd4'
CAPS = {'obvious_negative': -.05, 'obvious_positive_candidate': .15,
        'near_negative': -.005, 'near_positive_candidate': .005}
SOURCE_HASHES = {
 'objects/site_object.py': '4907bf2550945c395b85fd92725763ada4da533ad50966eb179e637dabeeebf6',
 'object_states/base_object_states.py': 'ca6f53123336b6487bdb4feb02534ddb3a0627a4b847fe89da4460520ef01366',
 'predicates/base_predicates.py': '319137e95330dbb22a266a8dceb8110f18413c2bf17461550222228c4bc310a1',
 'problems/libero_floor_manipulation.py': '38fce9925399651977033647525a64c28f50d45a110d72ecf35efa92054e9234',
}

def state(task, episode, split, prefix_repeat=None):
    target = {1: 'chocolate_pudding_1', 2: 'milk_1'}[task]
    row = dict(id='object_I_t%d_e%d' % (task, episode), suite='safelibero_object',
               level='I', task=task, episode=episode, target=target, goal='basket_1',
               goal_site='basket_1_contain_region', split=split, seed=7, budget=300)
    row['layout_group'] = row['id']
    if prefix_repeat is not None:
        row.update(id=row['id']+'_r%d_t252'%prefix_repeat, prefix_repeat=prefix_repeat,
                   prefix_steps=252, prefix_root=PREFIX_ROOT)
    return row

SMOKE = [state(2, 0, 'development_exposed')]
PILOT = [state(2, e, 'layout_holdout') for e in [6, 7, 8]] + [
         state(1, e, 'task_holdout') for e in [6, 7]] + [
         state(2, 5, 'external_exposed', r) for r in [0, 3]]

def bounds(site_pos, rotation, size):
    extent = [abs(sum(rotation[i][j]*size[j] for j in range(3))) for i in range(3)]
    lower = [site_pos[i]-extent[i] for i in range(3)]
    upper = [site_pos[i]+extent[i] for i in range(3)]
    lower[2] -= .01
    return lower, upper

def predicate(target, site_pos, rotation, size):
    lower, upper = bounds(site_pos, rotation, size)
    return all(lower[i] < target[i] < upper[i] for i in range(3))

def legal_action(action):
    return len(action) == 7 and all(math.isfinite(x) and -1 <= x <= 1 for x in action)

def exclusion(cap, supported):
    if supported and cap is not None and math.isfinite(cap) and cap <= -1e-6:
        return dict(label='infeasible', rule='strict_In_implies_ingress_gt_zero',
                    cap_m=cap, contradiction='ingress > 0 and ingress <= cap',
                    scope='SC-INGRESS-v1; pinned native point-containment predicate')
    return None

def decide(witness, certificate):
    if witness and certificate:
        raise RuntimeError('Contradictory feasible witness and exclusion certificate')
    return 'feasible' if witness else ('infeasible' if certificate else 'unknown')

def witness_at(records, cap, budget, prefix_safe=True):
    safe = bool(prefix_safe)
    for row in records:
        if row['step'] > budget:
            break
        safe = safe and row['protected_contact_count'] == 0
        if cap is not None:
            safe = safe and row['max_ingress_m'] <= cap
        if safe and row['native_success'] and row['synchronized_success']:
            return row['step']
    return None
