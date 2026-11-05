"""P4 open-loop runner and summary (docs/generative_verifier/HIL_MECHANISM_NEURONS.md sections 4.1, 5 and 6).

Every state is a stored switch-point policy input of HIL suite stage 1 (P0's selection), read under prompt A (base) and
prompt B (donor) on the SW input, as in the open-loop pilot. Nothing runs the simulator, trains or calls an API.

  run_neurons.py screen ROOT --checkpoint DIR                         (GPU) attribution on the discovery states -> screen/<id>/
  run_neurons.py select ROOT                                          (CPU) per-pair rankings and random sets -> NEURONS.json
  run_neurons.py measure ROOT --shard K --shards N --checkpoint DIR   (GPU) every state -> states/<id>/
  run_neurons.py summarize ROOT                                       (CPU) tool check, N0-N2, tables -> SUMMARY.json, and the
                                                                      discovery-only profiles for the explanations -> PROFILES_DISCOVERY.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent/'feasibility')); sys.path.insert(0, str(HERE.parents[1]))
import metrics as M  # noqa: E402
import neuron_readings as NR  # noqa: E402
import run_open_loop as O  # noqa: E402

GOAL_PROMPTS = ('open the middle drawer of the cabinet', 'put the bowl on the stove', 'put the wine bottle on top of the cabinet',
                'open the top drawer and put the bowl inside', 'put the bowl on top of the cabinet', 'push the plate to the front of the stove',
                'put the cream cheese in the bowl', 'turn on the stove', 'put the bowl on the plate', 'put the wine bottle on the rack')
read, write, sha, provenance = O.read, O.write, O.sha, O.provenance


def prompts(init):
    """The 14 profile prompts of a state (protocol 4.1): the ten LIBERO-Goal tasks, empty, random, STOP, RETURN."""
    from verifier_dataset import hil_suite as S
    from verifier_dataset import language_ablation as A
    return [('goal%d' % i, p) for i, p in enumerate(GOAL_PROMPTS)] + [('empty', ''), ('random', A.random_prompt(init)),
                                                                     ('stop', S.STOP), ('return', S.RETURN)]


def load(root):
    root = Path(root); plan = read(root/'PLAN.json')
    if sha(root/'STATES.json') != plan['states_sha256']:
        raise RuntimeError('STATES.json changed')
    states = {s['id']: s for s in read(root/'STATES.json')['states']}
    return root, plan, states


def policy_and_patcher(checkpoint):
    from openpi.policies import policy_config
    from openpi.training import config
    import neurons as N
    policy = policy_config.create_trained_policy(config.get_config('pi05_libero'), checkpoint)
    if policy._is_pytorch_model:
        raise RuntimeError('JAX policy required')
    return policy, N.NeuronPatcher(policy)


def save(folder, row, arrays):
    folder.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        np.save(folder/(name + '.npy'), value)
    write(folder/'ROW.json', row)
    write(folder/'HASHES.json', {p.name: sha(p) for p in sorted(folder.iterdir()) if p.name != 'HASHES.json'})


def verified(folder):
    for name, digest in read(folder/'HASHES.json').items():
        if sha(folder/name) != digest:
            raise RuntimeError('Output changed: %s/%s' % (folder, name))
    return read(folder/'ROW.json')


# ---------------------------------------------------------------- GPU: screening
def screen(root, checkpoint):
    import jax
    from serve import digest
    root, plan, states = load(root)
    policy, patcher = policy_and_patcher(checkpoint); done = []
    for sid in plan['discovery_ids']:
        state = states[sid]; folder = root/'screen'/sid; started = time.monotonic()
        if (folder/'HASHES.json').exists():
            raise RuntimeError('State already screened: %s' % sid)
        data_a, data_b, _, _ = O.inputs(state)
        score, total = patcher.neuron_atp(data_a, data_b, plan['window'])
        row = dict(id=sid, pair=state['pair'], init=state['init'], phase=state['phase'], total=total, finite=bool(np.isfinite(score).all()),
                   input_b_exact=digest(data_b) == state['b_first']['input_sha256'], score_max=float(np.nanmax(score)),
                   score_min=float(np.nanmin(score)), elapsed_seconds=time.monotonic() - started, **provenance())
        save(folder, row, dict(SCORE=score.astype(np.float32))); done.append(sid)
        write(root/'PROGRESS_screen.json', dict(done=len(done), total=len(plan['discovery_ids']), last=sid))
    write(root/'SCREEN_COMPLETE.json', dict(passed=True, states=done, jax_devices=[str(d) for d in jax.devices()], **provenance()))


# ---------------------------------------------------------------- CPU: selection
def select(root):
    root, plan, states = load(root)
    if not read(root/'SCREEN_COMPLETE.json')['passed']:
        raise RuntimeError('Screening did not pass')
    hooks = plan['hooks']; scores = {}
    rows = [verified(root/'screen'/sid) for sid in plan['discovery_ids']]
    for row in rows:
        scores.setdefault(row['pair'], []).append(np.load(root/'screen'/row['id']/'SCORE.npy'))
    pairs = {}; union = set(); width = None
    for pair in sorted(scores):
        top = NR.rank(np.stack(scores[pair]), hooks)[:NR.K_GRID[-1]]
        width = np.stack(scores[pair]).shape[-1]
        if len({(l, n) for l, n, _ in top}) != len(top):
            raise RuntimeError('Duplicate neurons in the ranking of %s' % pair)
        pairs[pair] = dict(states=len(scores[pair]), top=[list(t) for t in top],
                           random={str(k): [list(x) for x in NR.random_set(top, k, pair, width)] for k in NR.K_GRID})
        union |= {l * width + n for l, n, _ in top[:NR.TOP_PROFILE]}
    out = dict(pairs=pairs, union=sorted(union), width=width, hooks=hooks, k_grid=list(NR.K_GRID), discovery_ids=plan['discovery_ids'],
               ranking_ok=all(r['finite'] for r in rows), **provenance())
    write(root/'NEURONS.json', out)
    print(json.dumps({p: v['top'][:5] for p, v in pairs.items()}), flush=True)
    return out


# ---------------------------------------------------------------- GPU: measurement
def masks(neurons, pair, depth, width, hooks):
    import neurons as N
    info = neurons['pairs'][pair]; top = [(l, n) for l, n, _ in info['top']]
    none = np.zeros((depth, width), bool); every = N.layer_mask(depth, width, hooks)
    lnone = np.zeros(depth, bool); lall = np.isin(np.arange(depth), hooks)
    patch = dict(noop=(none, lnone), full=(every, lall), attn=(none, lall), mlp=(every, lnone))
    for k in NR.K_GRID:
        patch['top%d' % k] = (N.mask_of(top[:k], depth, width), lnone)
        patch['rand%d' % k] = (N.mask_of(info['random'][str(k)], depth, width), lnone)
    norm = {('top%d' % k): N.mask_of(top[:k], depth, width) for k in NR.K_GRID}; norm['all'] = every
    rho = {}
    for l in hooks:
        rho['mlp_L%02d' % l] = (N.layer_mask(depth, width, [l]), lnone)
        rho['attn_L%02d' % l] = (none, np.arange(depth) == l)
    for i in range(NR.TOP_PROFILE):
        rho['single%d' % (i + 1)] = (N.mask_of(top[i:i + 1], depth, width), lnone)
    stack = lambda d, names: (np.stack([d[n][0] for n in names]), np.stack([d[n][1] for n in names]))
    return stack(patch, NR.patch_names()), np.stack([norm[n] for n in NR.norm_names()]), list(rho), stack(rho, list(rho)), top


def measure_state(policy, patcher, state, plan, neurons):
    from serve import digest
    started = time.monotonic()
    depth, width, hooks = patcher.depth, patcher.mlp, plan['hooks']
    data_a, data_b, _, _ = O.inputs(state)
    seeds = [(state['seed'] + j) % O.SEED_SPAN for j in range(plan['seeds'])]
    (pm, pa), norm_sel, rho_names, (rm, ra), top = masks(neurons, state['pair'], depth, width, hooks)
    rho, acts, g_norm, norms, gap = patcher.neuron_patch(data_a, data_b, seeds, pm, pa, norm_sel, plan['window'])
    rho_more = patcher.neuron_rho(data_a, data_b, rm, ra, plan['window'])
    score, total = patcher.neuron_atp(data_a, data_b, plan['window'])
    native = O.native(policy, data_a, seeds[0])
    effects = M.state_effects(acts, list(NR.patch_names()))
    floor = effects['floor']
    distance = lambda x, y: float(np.linalg.norm(M.path(x) - M.path(y)))
    union = np.asarray(neurons['union'])
    stats, maps = [], {}
    for name, text in prompts(state['init']):
        s, m = patcher.neuron_profile(dict(data_a, prompt=text), union)
        stats.append(s)
        if text in (state['prompt_a'], state['prompt_b']):
            maps['A' if text == state['prompt_a'] else 'B'] = m
    atp = dict(total=total, **{'single%d' % (i + 1): float(score[l, n]) for i, (l, n) in enumerate(top[:NR.TOP_PROFILE])},
               **{'top%d' % k: float(sum(score[l, n] for l, n in top[:k])) for k in NR.K_GRID})
    row = dict(id=state['id'], pair=state['pair'], phase=state['phase'], init=state['init'], b_done=state['b_done'], seeds=seeds,
               input_b_exact=digest(data_b) == state['b_first']['input_sha256'],
               rho=dict(zip(NR.patch_names(), map(float, rho)), **dict(zip(rho_names, map(float, rho_more)))),
               D=dict(zip(NR.norm_names(), map(float, norms))), g_norm=g_norm, hook_gap=gap,
               noop_max_abs=float(np.abs(acts[2] - acts[0]).max()), noise_floor=floor, defined=effects['defined'], effect=effects['state'],
               mech_native_ratio=distance(acts[0, 0], native) / floor if floor > 0 else None, atp=atp,
               profile_prompts=[n for n, _ in prompts(state['init'])], elapsed_seconds=time.monotonic() - started, **provenance())
    if set(maps) != {'A', 'B'}:
        raise RuntimeError('Prompts A and B must be among the profile prompts: %s' % state['id'])
    arrays = dict(ACTIONS=acts.astype(np.float32), PROFILE=np.stack(stats).astype(np.float32),
                  MAPS=np.stack([maps['A'], maps['B']]).astype(np.float16))
    return row, arrays


def measure(root, shard, shards, checkpoint):
    import jax
    root, plan, states = load(root)
    neurons = read(root/'NEURONS.json')
    if sha(root/'NEURONS.json') != read(root/'NEURONS.sha256.json')['sha256']:
        raise RuntimeError('NEURONS.json changed')
    mine = [sid for i, sid in enumerate(plan['state_ids']) if i % shards == shard]
    policy, patcher = policy_and_patcher(checkpoint); done = []
    for sid in mine:
        folder = root/'states'/sid
        if (folder/'HASHES.json').exists():
            raise RuntimeError('State already measured: %s' % sid)
        row, arrays = measure_state(policy, patcher, states[sid], plan, neurons)
        save(folder, row, arrays); done.append(sid)
        write(root/('PROGRESS_%d.json' % shard), dict(done=len(done), total=len(mine), last=sid, last_seconds=row['elapsed_seconds']))
    write(root/('SHARD_%d_COMPLETE.json' % shard), dict(passed=True, states=done, jax_devices=[str(d) for d in jax.devices()], **provenance()))


# ---------------------------------------------------------------- CPU: summary
def discovery_profiles(root, plan, rows, neurons):
    """What the explanation writer sees: per neuron of the union, the mean activation per prompt and phase over the discovery
    inits of its own pairs, and the image rows where its B-prompt map peaks at f = 0. Nothing from the reading inits."""
    width = neurons['width']; owner = {}
    for pair, info in neurons['pairs'].items():
        for rank, (l, n, s) in enumerate(info['top'][:NR.TOP_PROFILE]):
            owner.setdefault(l * width + n, []).append(dict(pair=pair, rank=rank + 1, score=s))
    out = {}
    for j, flat in enumerate(neurons['union']):
        pairs = [o['pair'] for o in owner[flat]]
        mine = [r for r in rows if r['pair'] in pairs and r['init'] in NR.DISCOVERY_INITS]
        table = {}
        for r in mine:
            prof = np.load(root/'states'/r['id']/'PROFILE.npy')[:, j, 0]
            for name, value in zip(r['profile_prompts'], prof):
                table.setdefault('%s f=%s' % (r['pair'], r['phase']), {}).setdefault(name, []).append(float(value))
        peaks = []
        for r in mine:
            if r['phase'] == 0:
                b = np.load(root/'states'/r['id']/'MAPS.npy')[1, j].astype(float)
                peaks += [int(t) for t in np.argsort(-b)[:3]]
        out[str(flat)] = dict(layer=flat // width, unit=flat % width, owners=owner[flat],
                              mean=dict((c, {k: float(np.mean(v)) for k, v in d.items()}) for c, d in sorted(table.items())),
                              peak_rows=sorted(set(peaks)))
    return out


def summarize(root):
    root, plan, states = load(root)
    for k in range(plan['shards']):
        if not read(root/('SHARD_%d_COMPLETE.json' % k))['passed']:
            raise RuntimeError('Shard %d did not pass' % k)
    neurons = read(root/'NEURONS.json')
    if sha(root/'NEURONS.json') != read(root/'NEURONS.sha256.json')['sha256']:
        raise RuntimeError('NEURONS.json changed')
    screen = [verified(root/'screen'/sid) for sid in plan['discovery_ids']]
    rows = [verified(root/'states'/sid) for sid in plan['state_ids']]
    pairs = sorted({r['pair'] for r in rows}); phases = sorted({r['phase'] for r in rows})
    tool, detail = NR.tool_reading(rows, screen, neurons['ranking_ok'])
    ok = tool == 'tool_ok'
    route, route_cells = NR.route_reading(rows, pairs)
    sparse, per = NR.sparsity_reading(rows, pairs)
    used = NR.k_used(per)
    fade, fade_cells = NR.fade_reading(rows, used)
    summary = dict(
        mode=plan['mode'], tool=tool, tool_detail=detail, states=len(rows), reading_inits=list(NR.READING_INITS),
        readings=dict(N0=route if ok else None, N1=sparse if ok else None, N2=fade if ok else None),
        route_cells=route_cells, sparsity=per, k_used=used, fade_cells=fade_cells,
        described=NR.described(rows, used, pairs, phases), layer_curves=NR.layer_curves(rows, pairs, plan['hooks']),
        attribution=NR.attribution_agreement(rows),
        effects={'%s f=%s' % (p, f): {n: M.bootstrap_median([r['effect'][n] for r in NR.cell(rows, p, f)]) for n in NR.patch_names()[3:]}
                 for p in pairs for f in phases},
        top_overlap={'%s|%s' % (a, b): len({tuple(x[:2]) for x in neurons['pairs'][a]['top'][:100]} & {tuple(x[:2]) for x in neurons['pairs'][b]['top'][:100]})
                     for i, a in enumerate(pairs) for b in pairs[i + 1:]},
        top_layers={p: np.bincount([x[0] for x in neurons['pairs'][p]['top'][:used[p]]], minlength=len(plan['hooks'])).tolist() for p in pairs},
        seconds_per_state=float(np.median([r['elapsed_seconds'] for r in rows])), **provenance())
    write(root/'SUMMARY.json', summary)
    write(root/'PROFILES_DISCOVERY.json', discovery_profiles(root, plan, rows, neurons))
    print(json.dumps(dict(tool=tool, tool_detail=detail, readings=summary['readings'], k_used=used), indent=1), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['screen', 'select', 'measure', 'summarize'])
    parser.add_argument('root'); parser.add_argument('--shard', type=int, default=0); parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--checkpoint'); args = parser.parse_args()
    if args.action == 'screen':
        screen(args.root, args.checkpoint)
    elif args.action == 'select':
        out = select(args.root); write(Path(args.root)/'NEURONS.sha256.json', dict(sha256=sha(Path(args.root)/'NEURONS.json')))
    elif args.action == 'measure':
        measure(args.root, args.shard, args.shards, args.checkpoint)
    else:
        summarize(args.root)
