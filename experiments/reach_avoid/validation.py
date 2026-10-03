"""Persisted goal, horizon and label checks independent of online bookkeeping."""
import importlib.util
from pathlib import Path
from certificate import classify, certify, inscribed_radius

spec = importlib.util.spec_from_file_location('ra_native_contract',
    Path(__file__).resolve().parents[1]/'feasibility_contract/contract.py')
native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)


def goal(pose):
    return native.predicate(pose['target'],pose['site_position'],pose['site_rotation'],pose['site_size'])


def verify_steps(steps, summary, fixed_goal, certificate, horizon_T):
    if not steps or [x['step'] for x in steps] != list(range(len(steps))):
        raise RuntimeError('Noncontiguous/duplicate step records; exactly one initial step 0 is required')
    if summary['steps'] != len(steps)-1 or summary['execution_horizon_T'] != horizon_T or len(steps)-1 > horizon_T:
        raise RuntimeError('Witness step count/horizon mismatch')
    for row in steps:
        if row['step'] and not native.legal_action(row['action']):raise RuntimeError('Illegal persisted action')
        ng = goal(row['native_pose']); sg = goal(row['synchronized_pose'])
        target = row['synchronized_pose']['target']
        fg = all(lo < x < hi for lo,x,hi in zip(fixed_goal[0],target,fixed_goal[1]))
        expected = dict(native_success=ng,synchronized_success=sg,fixed_goal_success=fg,
            contract_success=bool(ng and sg and fg),safe_success=bool(ng and sg and fg and row['safe_history']))
        if any(row[k] != value for k,value in expected.items()):raise RuntimeError('Persisted goal predicate mismatch')
    last = steps[-1]
    if last['safe_history'] != summary['safe_history'] or last['safe_success'] != summary['safe_success']:
        raise RuntimeError('Terminal success/safety summary mismatch')
    wanted = classify(last['safe_success'] and summary['initial_valid'],len(steps)-1,horizon_T,certificate)
    if not summary['initial_valid']:wanted='unknown'
    if summary['label'] != wanted:raise RuntimeError('Persisted label logic mismatch')
    expected_outcome = ('invalid' if not summary['initial_valid'] else 'collision' if not last['safe_history']
        else 'safe_completion' if last['safe_success'] else 'safe_timeout')
    if summary['outcome'] != expected_outcome:raise RuntimeError('Persisted outcome mismatch')
    if expected_outcome=='safe_timeout' and summary['steps']!=horizon_T:raise RuntimeError('Safe timeout must reach full execution horizon')


def verify_certificate(geometry, material, fixture, initial_step, initial_valid, recorded):
    source = material['source']
    radius = inscribed_radius(source['half_sizes'],source['root_local']) if source else 0.
    if abs(radius-material['radius']) > 1e-12:raise RuntimeError('Stored material radius mismatch')
    result=certify(geometry['boxes'],fixture['lower'],fixture['upper'],
        initial_step['synchronized_pose']['target'],geometry['fixed_goal'],radius,
        initial_valid,geometry['static'],source is not None,True)
    if result['label']!=recorded['label'] or result['conditions']!=recorded['conditions']:
        raise RuntimeError('Persisted certificate mismatch')
    return result
