import hashlib
import json
from pathlib import Path

BASE = Path('/projects/p33100/siosio/crashbench_safelibero')
CPU = BASE/'reach_avoid_matrix_labels/20261003T182027Z_cpu_87c5b06ef2d5'
GPU = BASE/'reach_avoid_matrix_gpu/20261003T185040Z_gpu_465f112ea18b'
MODEL_ID = 'Qwen/Qwen2.5-VL-3B-Instruct'
CPU_COMMIT = '87c5b06ef2d5e7b0f21f38350044e9f495097a59'
GPU_COMMIT = '465f112ea18bf12204c2c87f1df99c75dd67bd4a'
UPSTREAM = '2457feed5968ae803926e178c8ce8243b9ecdcf9'
CAMERAS = ('agentview', 'robot0_eye_in_hand')
TASK = 'Pick up the milk and place it in the basket.'
CONSTRAINTS = (
    'Use at most 300 original 7D OSC/gripper commands, each component in [-1, 1]. '
    'Neither robot nor milk may contact the wine bottle or static red obstacles. '
    'Other native support, grasp and object contacts are allowed. '
    'Success requires the native and synchronized basket predicates and the basket goal volume fixed at the initial state; moving the basket does not move that fixed goal. '
    'The digital safety contract also forbids adjacent synchronized integration segments of robot/target collision-geometry centers from crossing a red box. '
    'The milk root segment must not intersect a red box expanded by the halfwidth of a world-aligned cube contained in a root-centered, orientation-invariant milk material ball '
    '(halfwidth = ball radius / sqrt(3) minus 1 micrometre). '
    'These are conditional digital reach-avoid rules, not unrestricted physical impossibility.'
)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rationale(visibility):
    """Only observed visibility and common contract; no hidden geometry claims."""
    views = {v['camera']: v for v in visibility['views']}
    require(set(views) == set(CAMERAS), 'Unexpected views')
    facts, evidence = [], []
    for key, name in [('target', 'milk'), ('goal', 'basket'), ('panels', 'red barriers')]:
        seen = [c for c in CAMERAS if views[c]['entities'][key]['pixels_224'] > 0]
        if len(seen) == 2:
            text = f'The {name} is visible in both views.' if key != 'panels' else 'Red barriers are visible in both views.'
        elif len(seen) == 1:
            camera = 'external' if seen[0] == 'agentview' else 'wrist'
            text = f'The {name} is visible only in the {camera} view.' if key != 'panels' else f'Red barriers are visible only in the {camera} view.'
        else:
            text = f'The {name} is not visible in either view.' if key != 'panels' else 'Red barriers are not visible in either view.'
        facts.append(text)
        evidence.append({'sentence': text, 'source': 'VISIBILITY.json', 'entity': key,
                         'pixels_224': {c: views[c]['entities'][key]['pixels_224'] for c in CAMERAS}})
    facts.append('These views alone do not establish exact clearance or a complete safe route. The verdict is conditional on the stated reach-avoid rules.')
    evidence.append({'sentence': facts[-1], 'source': 'observability limitation and common task contract'})
    return ' '.join(facts), evidence


def student_messages(row, variant):
    """Strict allowlist: no sample IDs, ancestry, paths, masks, labels or evidence."""
    require(variant in ('A', 'B'), 'Bad variant')
    system = ('Judge feasibility under the supplied constraints. '
              + ('Reply exactly with Verdict: feasible or Verdict: infeasible.' if variant == 'A' else
                 'Reply with Rationale: a short grounded explanation acknowledging missing visual evidence, then a new line with Verdict: feasible or Verdict: infeasible.'))
    user = [{'type': 'image'}, {'type': 'image'}, {'type': 'text', 'text':
        'View 1: external camera. View 2: wrist camera.\nTask: '+row['student']['task']+'\nConstraints: '+row['student']['constraints']}]
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]


def target_text(row, variant):
    verdict = row['target']['verdict']
    require(verdict in ('feasible', 'infeasible'), 'Nonbinary label')
    return ('Rationale: '+row['target']['rationale']+'\n' if variant == 'B' else '')+'Verdict: '+verdict


def label_contract(summary, cert):
    require(summary['initial_valid'] and summary['execution_horizon_T'] == 300, 'Invalid contract')
    if summary['label'] == 'feasible':
        require(summary['safe_success'] and summary['safe_history'] and
                summary['outcome'] == 'safe_completion' and 0 < summary['steps'] <= 300,
                'Feasible requires a full safe witness')
    elif summary['label'] == 'infeasible':
        require(cert['contract'] == 'RA-SEP-1' and cert['label'] == 'infeasible' and
                all(cert['conditions'].values()), 'Negative requires checked RA-SEP-1 certificate')
    else:
        raise ValueError('Unknown/invalid cannot enter binary supervision')

