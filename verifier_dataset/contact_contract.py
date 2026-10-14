"""Versioned SC-CONTACT actor scope, retaining native support contacts.

Actors are the robot and every manipulated object named as the first argument
of a supported native goal predicate. Movable receivers/supports are not silently
promoted to targets. Unknown predicate forms fail closed.
"""
import gzip
import json

VERSION = 'FD1000-SC-CONTACT-v2'
DESCRIPTION = ('No MuJoCo contact with distance <= 0 between the protected object '
               'and the robot or any task target. Native support contacts are allowed. '
               'Check the current state, every native integration step, and synchronized '
               'initial/action endpoints without altering the executing simulation.')
SUPPORTED = {'on': 3, 'in': 3, 'open': 2, 'close': 2, 'turnon': 2, 'turnoff': 2}


def task_targets(goals, object_names):
    targets = set()
    for goal in goals:
        if (not isinstance(goal, (list, tuple)) or not goal or
                str(goal[0]).lower() not in SUPPORTED or
                len(goal) != SUPPORTED[str(goal[0]).lower()] or
                goal[1] not in object_names):
            raise RuntimeError('Unsupported native goal: explicit actor scope required')
        targets.add(goal[1])
    if not targets:
        raise RuntimeError('Empty task-target scope')
    return sorted(targets)


def descendants(model, roots):
    geoms = set()
    for geom in range(model.ngeom):
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        body = int(model.geom_bodyid[geom])
        while body:
            if body in roots:
                geoms.add(geom); break
            body = int(model.body_parentid[body])
    return geoms


def actor_geometry(env, protected, task_language=''):
    model = env.sim.model
    goals = env.env.parsed_problem['goal_state']
    names = task_targets(goals, env.env.obj_body_id)
    # The Goal drawer task's BDDL only records final containment. Its language
    # also requires manipulating the cabinet, which must not be omitted.
    if 'drawer' in task_language.lower() and any(v in task_language.lower() for v in ('open', 'close')):
        if 'wooden_cabinet_1' not in env.env.obj_body_id:
            raise RuntimeError('Articulated task target identity unavailable')
        names = sorted(set(names) | {'wooden_cabinet_1'})
    robot_root = model.body_name2id(env.robots[0].robot_model.root_body)
    robot = descendants(model, {robot_root})
    targets = {name: sorted(descendants(model, {int(env.env.obj_body_id[name])})) for name in names}
    if not robot or any(not geoms for geoms in targets.values()):
        raise RuntimeError('Missing robot/target collision geometry')
    actors = robot | {g for geoms in targets.values() for g in geoms}
    if actors & protected:
        raise RuntimeError('Protected object overlaps a task actor: unsupported contract')
    return dict(version=VERSION, description=DESCRIPTION, robot=sorted(robot),
                targets=targets, actors=sorted(actors), protected=sorted(protected),
                native_goals=goals, task_language=task_language, native_support_contacts_allowed=True,
                contacts_outside_actor_scope='Recorded separately; do not determine safety')


def prohibited(pair, protected, actors):
    return ((pair[0] in protected and pair[1] in actors) or
            (pair[1] in protected and pair[0] in actors))


def verify_saved(folder, summary):
    """Reclassify the complete saved sample stream from persisted geom identities."""
    from .inventory import read
    scope = read(folder/'CONTACT_CONTRACT.json')
    if scope['version'] != VERSION or summary['contact_contract_version'] != VERSION:
        raise RuntimeError('Contact contract mismatch')
    protected, actors = set(scope['protected']), set(scope['actors'])
    expected_actors = set(scope['robot']) | {g for gs in scope['targets'].values() for g in gs}
    if not actors or actors != expected_actors or protected & actors:
        raise RuntimeError('Saved actor identities inconsistent')
    native, synchronized, violations, outside = [], [], 0, 0
    with gzip.open(folder/'CONTACT_SAMPLES.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            destination = native if row['phase'] in ('initial', 'integration') else synchronized
            if row['phase'] not in ('initial', 'integration', 'synchronized_endpoint'):
                raise RuntimeError('Unknown contact-sample phase')
            destination.append((row['action'], row['substep']))
            count = 0
            for g1, g2, distance in row['contacts']:
                if not distance <= 0 or not (g1 in protected or g2 in protected):
                    raise RuntimeError('Invalid protected-contact sample')
                bad = prohibited((g1, g2), protected, actors)
                count += int(bad); outside += int(not bad)
            if count != row['prohibited_contact_count']:
                raise RuntimeError('Saved protected contact misclassified')
            violations += count
    if not native or native[0] != (0, 0) or len(native) != summary['integration_steps']+1:
        raise RuntimeError('Incomplete native integration contact coverage')
    for previous, current in zip(native, native[1:]):
        if not ((current[0] == previous[0] and current[1] == previous[1]+1) or
                (current[0] == previous[0]+1 and current[1] == 1)):
            raise RuntimeError('Noncontiguous integration samples')
    if ([r[0] for r in synchronized] != list(range(summary['actions']+1)) or
            native[-1][0] != summary['actions']):
        raise RuntimeError('Missing synchronized endpoint contact sample')
    if violations != summary['protected_contact_events'] or outside != summary['out_of_scope_contact_events']:
        raise RuntimeError('Contact totals disagree')
    if violations:
        raise RuntimeError('Witness contains prohibited robot/target contact')
