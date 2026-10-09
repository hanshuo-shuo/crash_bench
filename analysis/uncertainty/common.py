"""Frozen identity, metric and outcome rules; importable without ML libraries."""
import hashlib
import json
import math
import os
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
ASSETS = Path('/projects/p33100/siosio/crashbench_safelibero')
FIELDS = ['scene_id', 'task_id', 'capability', 'seed', 'step', 'disagreement',
          'churn', 'act_norm', 'min_dist', 'crashed', 'time_to_crash', 'outcome']


def config(mode='smoke'):
    # JSON is a YAML subset. This keeps the simulator's Python 3.8 dependency-free.
    value = json.loads((Path(__file__).parent / 'config.yaml').read_text())
    if (value['samples'], value['horizon'], value['replan_steps'], value['effective_dims']) != (8, 10, 5, 7):
        raise ValueError('Unapproved sampling configuration')
    if value['resources'] != dict(account='p33100', partition='gengpu', gpu='a100:1', cpus=8, memory_gb=64, minutes=30):
        raise ValueError('Resource envelope changed')
    if value['api_limit_usd'] != '3.00' or value['sampling_atol'] != 1e-6 or value['sampling_rtol'] != 1e-6:
        raise ValueError('Budget or numeric gate changed')
    if mode == 'full':
        full = value['full_collection']; value = dict(value, **full)
        if value['resources'] != dict(account='p33100', partition='gengpu', gpu='a100:1', cpus=8, memory_gb=64, minutes=1440, workers=2):
            raise ValueError('Full resource envelope changed')
        if value['api_limit_usd'] is not None or value['repeats'] != list(range(10)) or value['max_fresh_calls_per_run_root'] != 600:
            raise ValueError('Full collection scope changed')
    elif mode != 'smoke': raise ValueError('Unknown collection mode')
    value['mode'] = mode
    return value


def full_schedule(cfg, shard=None):
    manifest = json.loads((Path(__file__).parent/'states.json').read_text())
    if len(manifest['states']) != 60: raise ValueError('Exactly sixty states required')
    specs=[]
    for index, scene in enumerate(manifest['states']):
        if shard is not None and index % 2 != shard: continue
        for repeat in cfg['repeats']:
            methods = ['nominal','aegis'] if repeat % 2 == 0 else ['aegis','nominal']
            for method in methods:
                specs.append(dict(name='full_s%02d_r%02d_%s'%(index,repeat,method), scene=scene,
                                  repeat=repeat, method=method, diagnostics=True, state_index=index))
    # The eight first-stage smoke cases are part of, rather than additional to,
    # the authorized matrix. Both exposed states belong to shard0.
    front=[x for x in specs if x['state_index'] in [0,42] and x['repeat'] in [0,1]]
    yield from front
    yield from (x for x in specs if x not in front)


def scene_id(scene):
    return '%s/%s/task%d/episode%02d' % (scene['suite'], scene['level'], scene['task'], scene['episode'])


def seed_for(scene, repeat):
    text = 'paired-v1|%s/%s/task%d|%d|%d' % (scene['suite'], scene['level'], scene['task'], scene['episode'], repeat)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'big')


def noise_seed(scene, seed, infer_index, sample):
    text = 'uncertainty-v1|%s|%d|%d|%d' % (scene, seed, infer_index, sample)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'big')


def schedule(cfg):
    scene = cfg['scenes'][0]
    yield dict(name='timing', scene=scene, repeat=0, method='nominal', diagnostics=False)
    for i, scene in enumerate(cfg['scenes']):
        for repeat in cfg['repeats']:
            for method in cfg['methods']:
                yield dict(name='smoke_s%d_r%d_%s' % (i, repeat, method), scene=scene,
                           repeat=repeat, method=method, diagnostics=True)


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm(action, scales, dims=7):
    return math.sqrt(sum((float(action[d]) / float(scales[d])) ** 2 for d in range(dims)))


def disagreement(chunks, scales):
    m, h, d = len(chunks), len(chunks[0]), len(scales)
    if m < 2 or h != 10 or d != 7:
        raise ValueError('Expected M>=2, H=10 and D=7')
    total = 0.
    for t in range(h):
        for axis in range(d):
            values = [float(x[t][axis]) / scales[axis] for x in chunks]
            mean = sum(values) / m
            total += math.sqrt(sum((x-mean)**2 for x in values) / (m-1))
    return total / (h*d)


def churn(current, previous, scales):
    if previous is None:
        return None
    return sum(abs(float(current[t][d])-float(previous[t+5][d])) / scales[d]
               for t in range(5) for d in range(7)) / 35.


def outcome(success, collision_step):
    return 'crash' if collision_step is not None else ('safe_success' if success else 'safe_incomplete')


def time_to_crash(step, collision_step):
    if collision_step is None:
        return -1
    return collision_step-step if step <= collision_step else None


def remaining_minutes(receipts, ceiling_minutes=30):
    terminal = {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','DEADLINE','PREEMPTED','REVOKED'}
    if any(x['state'].split()[0] not in terminal or int(x['elapsed_seconds']) < 0 for x in receipts):
        raise RuntimeError('Campaign allocation has unknown or active accounting')
    used = sum(int(x['elapsed_seconds']) for x in receipts)
    minutes = (ceiling_minutes*60-used)//60
    if minutes < 1: raise RuntimeError('Cumulative allocation ceiling exhausted')
    return minutes, used
